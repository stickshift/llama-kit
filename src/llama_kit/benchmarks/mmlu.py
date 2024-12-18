"""Utilities for running Massive Multitask Language Understanding (MMLU) benchmark."""

from collections.abc import Sequence, Set
import csv
import logging
from pathlib import Path
from random import sample
import shutil
import tarfile
import tempfile
from textwrap import dedent
from typing import Iterator, NamedTuple

from IPython.display import display
from pandas import DataFrame
import torch
from torch import nn
from torch.nn import functional as F
import requests
from rich.progress import Progress

from llama_kit.model import Message, LlamaHead, LlamaModel, ModelConfig, render_prompt, load_config, load_parameters, load_tokenizer, unpack_parameters
from llama_kit.tools import executor, default_arg

__all__ = [
    "OPTIONS",
    "Answer",
    "Answers",
    "Question",
    "Questions",
    "download_dataset",
    "load_dataset",
    "select_question",
    "display_questions",
    "generate_prompt",
    "generate_answers",
    "MMULGenerator",
    "evaluate_generator",
    "filter_questions",
]

logger = logging.getLogger(__name__)


class Question(NamedTuple):
    """An MMLU question."""

    qid: int

    category: str

    question: str

    A: str

    B: str

    C: str

    D: str

    answer: str


Questions = Sequence[Question]

Categories = Set[str]

OPTIONS = ("A", "B", "C", "D")


class Answer(NamedTuple):
    """An answer to MMLU question."""

    qid: int

    expected: str

    actual: str

    scores: dict[str, float]

    correct: bool


Answers = Sequence[Answer]


class Dataset(NamedTuple):
    """MMLU dataset."""
    
    questions: Questions

    examples: Questions

    categories: Categories


_mmlu_dataset_url = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"


def download_dataset(output_path: Path):
    """Download MMLU dataset to output_path."""
    # Check if it exists already
    if output_path.exists():
        logger.info(f"Dataset {output_path.name} exists. Skipping download.")
        return

    work_dir = tempfile.TemporaryDirectory()
    work_path = Path(work_dir.name)

    # Download tarball
    response = requests.get(_mmlu_dataset_url, stream=True)
    total = int(response.headers["Content-Length"])

    with Progress() as progress, tempfile.NamedTemporaryFile() as tarball:
        task = progress.add_task("Downloading...", total=total)

        for data in response.iter_content(chunk_size=5 * 1024 * 1024):
            tarball.write(data)
            progress.update(task, advance=len(data), refresh=True)

        with tarfile.open(tarball.name) as tf:
            tf.extractall(work_path, filter="data")

        shutil.move(work_path / "data", output_path)


def load_dataset(dataset_path: Path) -> Dataset:
    """Load MMLU examples and questions."""

    def load_data_file(path: Path) -> Questions:
        # Infer category from file name: x_y_z_test.csv -> x y z
        category = " ".join(path.stem.split("_")[0:-1])

        with open(path, mode="r", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file)
            questions = tuple(Question(i, category, *row) for i, row in enumerate(reader))

        return questions

    def load_segment(segment: str) -> Questions:
        # Sort paths to ensure consistent order
        paths = sorted(path for path in dataset_path.glob(f"{segment}/*.csv"))

        # Load data files in parallel
        futures = [executor.submit(load_data_file, path) for path in paths]

        # Collect results
        collected = ()
        for future in futures:
            collected += future.result()

        # Reassign ids
        questions = ()
        for i, question in enumerate(collected):
            questions += (Question(i, *question[1:]),)

        return questions

    questions = load_segment("test")
    examples = load_segment("dev")
    categories = {q.category for q in questions}

    return Dataset(questions=questions, examples=examples, categories=categories)


def filter_questions(
    config: ModelConfig,
    questions: Questions,
    max_sequence_len: int,
    n_shots: int,
    examples: Questions,
) -> Questions:
    """Filter out questions that exceed max sequence length."""

    results = []
    
    tokenizer = load_tokenizer(config)

    for question in questions:
        
        # Generate prompt
        messages = generate_prompt(question, n_shots=n_shots, examples=examples)
        prompt = render_prompt(config, messages)
        
        # Split prompt into tokens
        token_ids = tokenizer.encode(prompt, bos=True, eos=False)

        if len(token_ids) > max_sequence_len:
            logger.debug(f"Filtered out question {question.qid}: {len(token_ids)} > {max_sequence_len}")
            continue

        results.append(question)

    before, after = len(questions), len(results)
    logger.info(f"Filtered out {before - after} of {before} questions. {after} remaining.")
    
    return results


def select_question(questions: Questions, *, qid: int | None = None, question: str | None = None) -> Question | None:
    """Select question by qid or question text."""
    if qid is not None:
        return next((q for q in questions if q.qid == qid), None)

    if question is None:
        raise ValueError("Must specify either qid or question")

    return next((q for q in questions if q.question == question), None)


def display_questions(questions: Questions, n: int | None = None):
    """Render random sample of questions as a table."""

    # Defaults
    n = default_arg(n, 5)

    # Randomly select sample
    selected = questions if len(questions) <= n else sample(questions, n)
    
    display(DataFrame(selected))


def generate_prompt(
    question: Question,
    *,
    n_shots: int,
    examples: Questions | None = None,
) -> Sequence[Message]:
    """Generate prompt for specified question."""

    # Validate
    if n_shots < 0 or n_shots > 5:
        raise ValueError("n_shots must be between 0 and 5")
    
    if n_shots > 0 and examples is None:
        raise ValueError("n_shots specified without examples")

    selected_examples = []
    if n_shots > 0:
        # Select examples for category
        selected_examples = [e for e in examples if e.category == question.category]

        # Deterministically select n_shots if specified
        selected_examples = selected_examples[:n_shots]

    messages = []

    # System message
    messages.append(
        Message(
            role="system",
            content=(
                f"You are a student answering multiple choice questions on an exam. Each question "
                f"has 4 options: A, B, C, D. There will be {n_shots} example questions followed by "
                f"a test question. Your job is to answer the test question. Your answer MUST be one "
                f"of {{A, B, C, D}}."
            )
        )
    )
    
    content = ""
    
    # Header
    content += f"# Instructions\n\n"
    content += f"The following are multiple choice questions (with answers) about {question.category}.\n\n"

    # Examples
    for i, row in enumerate(selected_examples):
        content += (
            f"# Example {i}\n\n"
            f"{row.question}\n"
            f"\n"
            f"A) {row.A}\n"
            f"B) {row.B}\n"
            f"C) {row.C}\n"
            f"D) {row.D}\n"
            f"\n"
            f"Answer: {row.answer}\n\n"
        )

    # Question
    content += (
        f"# Question\n\n"
        f"{question.question}\n"
        f"\n"
        f"A) {question.A}\n"
        f"B) {question.B}\n"
        f"C) {question.C}\n"
        f"D) {question.D}\n"
        f"\n"
        f"Answer: "
    )

    messages.append(Message(role="user", content=content))

    return messages


class MMLUGenerator(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        self.tokenizer = load_tokenizer(self.config)
        self.model = LlamaModel(config)
        self.head = LlamaHead(config)
    
    def __call__(
        self,
        questions: Questions,
        *,
        n_shots: int,
        examples: Questions | None = None,
    ) -> Iterator[Answer]:
    
        # Look up token ids for MMLU options A, B, C, D
        mmlu_token_ids = {option: self.tokenizer.encode(option, bos=False, eos=False)[0] for option in OPTIONS}
    
        # Generate answers to each question
        for question in questions:
            
            # Generate prompt
            messages = generate_prompt(question, n_shots=n_shots, examples=examples)
            prompt = render_prompt(self.config, messages)
            
            # Split prompt into tokens
            tokenizer_opts = {
                "bos": True,
                "eos": False,
                "allowed_special": "all",
            }
            token_ids = self.tokenizer.encode(prompt, **tokenizer_opts)
            logger.debug(f"Split prompt into {len(token_ids)} token ids")

            # Load token ids into tensor
            token_ids = torch.tensor(token_ids, device=self.config.device, dtype=torch.int64)
    
            # Transform token ids into semantic embeddings
            embeddings = self.model(token_ids)
    
            # Project embeddings back to token space
            logits = self.head(embeddings)
            
            # Extract logits for MMLU options
            mmlu_logits = torch.tensor(
                [logits[mmlu_token_ids[option]] for option in OPTIONS],
                device=self.config.device,
            )
            
            # Convert to scores (probability distribution over options)
            scores = F.softmax(mmlu_logits, dim=-1)
            
            # Map options to scores
            scores = {option: scores[i] for i, option in enumerate(OPTIONS)}
            
            # Convert scores back to floats
            scores = {k: v.item() for k, v in scores.items()}
            
            # Calculate answer
            actual = max(scores, key=scores.get)
    
            # Yield answer
            yield Answer(
                qid=question.qid,
                expected=question.answer,
                actual=actual,
                scores=scores,
                correct=(actual == question.answer),
            )


def evaluate_generator(
    generator: MMLUGenerator,
    *,
    questions: Questions,
    n_shots: int, 
    examples: Questions, 
) -> float:
    # Generate answers
    answers = tuple(generator(questions, n_shots=n_shots, examples=examples))

    # Calculate score
    correct_answers = tuple(a for a in answers if a.correct)
    score = 100 * len(correct_answers) / len(answers)    

    return score
