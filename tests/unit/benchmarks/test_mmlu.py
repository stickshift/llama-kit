from pathlib import Path

from llama_kit.benchmarks.mmlu import (
    OPTIONS,
    MMLUGenerator,
    evaluate_generator,
    generate_prompt,
    load_dataset,
    select_question,
)
from llama_kit.model import load_config, load_parameters


def test_load_dataset(mmlu_dataset_path: Path):
    #
    # Whens
    #

    # I load dataset
    dataset = load_dataset(mmlu_dataset_path)

    #
    # Thens
    #

    # There should 14,042 questions
    assert len(dataset.questions) == 14042

    # There should 57 categories
    assert len(dataset.categories) == 57

    # There should 5 examples per category
    for category in dataset.categories:
        assert len([e for e in dataset.examples if e.category == category]) == 5

    #
    # Whens
    #

    # I query question by qid
    question0 = select_question(dataset.questions, qid=120)

    #
    # Thens
    #

    # question should be
    assert question0.question == "Where is the sinoatrial node located?"
    #
    # Whens
    #

    # I query question by question
    question1 = select_question(dataset.questions, question=question0.question)

    #
    # Thens
    #

    # question1 should match question0
    assert question1 == question0


def test_prompt_zero_shot(mmlu_dataset_path: Path):
    #
    # Givens
    #

    # I loaded dataset
    dataset = load_dataset(mmlu_dataset_path)

    # I looked up question 7779
    question = select_question(dataset.questions, qid=7779)

    #
    # Whens
    #

    # I generate zero-shot prompt
    messages = generate_prompt(question, n_shots=0)

    #
    # Thens
    #

    # messages includes system message
    assert messages[0].role == "system"

    # messages includes user message
    assert messages[-1].role == "user"


def test_generate_answers(mmlu_dataset_path: Path):
    #
    # Givens
    #

    # I loaded dataset
    dataset = load_dataset(mmlu_dataset_path)

    # I looked up question 7779
    question = select_question(dataset.questions, qid=7779)

    # I initialized generator
    config = load_config("Llama3.2-3B-Instruct")
    generator = MMLUGenerator(config)
    generator.load_state_dict(load_parameters(config))

    #
    # Whens
    #

    # I generate answer to question
    answer = next(generator([question], n_shots=0, examples=dataset.examples))

    #
    # Thens
    #

    # answer should be populated
    assert answer.qid == question.qid
    assert answer.expected == "B"
    assert isinstance(answer.actual, str)
    assert all(option in answer.scores for option in OPTIONS)
    assert isinstance(answer.correct, bool)


def test_evaluate(mmlu_dataset_path: Path):
    #
    # Givens
    #

    # I loaded dataset
    dataset = load_dataset(mmlu_dataset_path)

    # I looked up question 7779
    question = select_question(dataset.questions, qid=7779)

    # I initialized generator
    config = load_config("Llama3.2-3B-Instruct")
    generator = MMLUGenerator(config)
    generator.load_state_dict(load_parameters(config))

    #
    # Whens
    #

    # I evaluate generator
    score = evaluate_generator(generator, questions=[question], n_shots=0, examples=dataset.examples)

    #
    # Thens
    #

    # score should be populated
    assert 0.0 <= score <= 100.0
