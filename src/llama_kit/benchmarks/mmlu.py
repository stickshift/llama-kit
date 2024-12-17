"""Utilities for running Massive Multitask Language Understanding (MMLU) benchmark."""

from collections.abc import Sequence, Set
import csv
from pathlib import Path
import shutil
import tarfile
import tempfile
from textwrap import dedent
from typing import Iterator, NamedTuple

import torch
from torch.nn import functional as F
import requests
from rich.progress import Progress

from llama_kit.model import Message, LlamaHead, LlamaModel, render_prompt, load_config, load_parameters, load_tokenizer, unpack_parameters
from llama_kit.tools import executor

__all__ = [
    "OPTIONS",
    "Answer",
    "Answers",
    "Question",
    "Questions",
    "download_dataset",
    "load_dataset",
    "select_question",
    "generate_prompt",
    "generate_answers",
    "evaluate_model",
]


class Question(NamedTuple):
    """Represents an MMLU question."""

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
    """Represents an answer to MMLU question."""

    qid: int

    expected: str

    actual: str

    scores: dict[str, float]

    correct: bool


Answers = Sequence[Answer]

_mmlu_dataset_url = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"


def download_dataset(output_path: Path):
    """Download MMLU dataset to output_path."""
    # Check if it exists already
    if output_path.exists():
        print(f"Dataset {output_path.name} exists. Skipping.")  # noqa: T201
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


def load_dataset(dataset_path: Path) -> tuple[Questions, Questions, Categories]:
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

    return questions, examples, categories


def select_question(questions: Questions, *, qid: int | None = None, question: str | None = None) -> Question | None:
    """Select question by qid or question text."""
    if qid is not None:
        return next((q for q in questions if q.qid == qid), None)

    if question is None:
        raise ValueError("Must specify either qid or question")

    return next((q for q in questions if q.question == question), None)


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

    selected_examples = None
    if n_shots > 0:
        # Select examples for category
        selected_examples = [e for e in examples if e.category == question.category]

        # Deterministically select n_shots if specified
        selected_examples = selected_examples[:n_shots]

    messages = (
        Message(
            role="system",
            content=(
                f"You are a student answering multiple choice questions on an exam. Each question "
                f"has 4 options A, B, C, D. There will be {n_shots} example questions followed by "
                f"a test question. Your job is to answer the test question."
            )
        ),
        Message(
            role="user",
            content=(
                f"# Instructions\n\n"
                f"The following are multiple choice questions about {question.category}."
            )
        ),
    )

    if selected_examples:
        content = "# Example Questions\n\n"
        for row in selected_examples:
            content += (
                f"Question: {row.question}\n"
                f"\n"
                f"Options:\n"
                f"  A) {row.A}\n"
                f"  B) {row.B}\n"
                f"  C) {row.C}\n"
                f"  D) {row.D}\n"
                f"\n"
                f"Answer: {row.answer}\n\n"
            )
        messages += (Message(role="user", content=content),)

    # Pose question
    content = "# Test Question\n\n"
    content += (
        f"Question: {question.question}\n"
        f"\n"
        f"Options:\n"
        f"  A) {question.A}\n"
        f"  B) {question.B}\n"
        f"  C) {question.C}\n"
        f"  D) {question.D}\n"
        f"\n"
        f"Answer: "
    )
    messages += (Message(role="user", content=content),)

    return messages


def generate_answers(
    checkpoint: str,        
    *,
    questions: Questions,
    n_shots: int, 
    examples: Questions,
) -> Iterator[Answer]:
    """Generate answers to each question."""

    # Configure models
    config = load_config(checkpoint)
    parameters = load_parameters(config, map_location=config.device)
    model = LlamaModel(config)
    head = LlamaHead(config)
    tokenizer = load_tokenizer(config)

    # Load model parameters
    model.load_state_dict(unpack_parameters(parameters, ["model"]))
    head.load_state_dict(unpack_parameters(parameters, ["head"]))

    # Look up token ids for MMLU options A, B, C, D
    mmlu_token_ids = {option: tokenizer.encode(option, bos=False, eos=False)[0] for option in OPTIONS}

    # Generate answers to each question
    for question in questions:
        
        # Generate prompt
        messages = generate_prompt(question, n_shots=n_shots, examples=examples)
        prompt = render_prompt(config, messages)
        
        # Split prompt into tokens
        token_ids = torch.tensor(tokenizer.encode(prompt, bos=True, eos=False), device=config.device)

        # Transform token ids into semantic embeddings
        embeddings = model(token_ids)

        # Project embeddings back to token space
        logits = head(embeddings)
        
        # Extract logits for MMLU options
        mmlu_logits = torch.tensor(
            [logits[mmlu_token_ids[option]] for option in OPTIONS],
            device=config.device,
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


def evaluate_model(
    checkpoint: str,
    *,
    questions: Questions,
    n_shots: int, 
    examples: Questions, 
) -> float:
    # Generate answers
    answers = tuple(generate_answers(checkpoint, questions=questions, n_shots=n_shots, examples=examples))

    # Calculate score
    correct_answers = tuple(a for a in answers if a.correct)
    score = len(correct_answers) / len(answers)

    return score
