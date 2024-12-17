from pathlib import Path
from textwrap import dedent

from llama_kit.benchmarks.mmlu import evaluate_model, load_dataset, select_question, generate_prompt, generate_answers, OPTIONS


def test_load_dataset(mmlu_dataset_path: Path):

    #
    # Whens
    #

    # I load dataset
    questions, examples, categories = load_dataset(mmlu_dataset_path)

    #
    # Thens
    #

    # There should 14,042 questions
    assert len(questions) == 14042

    # There should 57 categories
    assert len(categories) == 57

    # There should 5 examples per category
    for category in categories:
        assert len([e for e in examples if e.category == category]) == 5

    #
    # Whens
    #

    # I query question by qid
    question0 = select_question(questions, qid=120)

    #
    # Thens
    #

    # question should be 
    assert question0.question == "Where is the sinoatrial node located?"
    #
    # Whens
    #

    # I query question by question
    question1 = select_question(questions, question=question0.question)

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
    questions, examples, categories = load_dataset(mmlu_dataset_path)

    # I looked up question 7779
    question = select_question(questions, qid=7779)

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
    questions, examples, _ = load_dataset(mmlu_dataset_path)

    # I looked up question 7779
    question = select_question(questions, qid=7779)

    #
    # Whens
    #

    # I generate answer to question using 3.2 3B pretrained model
    answer = next(generate_answers("Llama3.2-3B", questions=[question], n_shots=0, examples=examples))

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
    questions, examples, _ = load_dataset(mmlu_dataset_path)

    # I looked up question 7779
    question = select_question(questions, qid=7779)

    #
    # Whens
    #

    # I evaluate 3.2 3B pretrained model
    score = evaluate_model("Llama3.2-3B", questions=[question], n_shots=5, examples=examples)

    #
    # Thens
    #

    # score should be populated
    assert 0.0 <= score <= 1.0

    #
    # Whens
    #

    # I evaluate 3.2 3B instruct model
    score = evaluate_model("Llama3.2-3B-Instruct", questions=[question], n_shots=5, examples=examples)

    #
    # Thens
    #

    # score should be populated
    assert 0.0 <= score <= 1.0
