import os
import json
import collections

# Custom imports
from constants import *
from utils import *
from prompts import *

from custom_types import Question
from idutils import parse_question_string_id


def prepare_batch_mcq_generation(
    filename: str,
    seeds: list[str],
    fictsheets: list[str],
    fictions_lookup: dict[str, str],
    surviving_qa: dict[str, dict[str, Question]],
) -> None:
    """
    Prepares the batch prompts for MCQ generation.

    For each question, prepare a prompt that provides the fictsheet,
    fiction, question, and correct answer, and asks for 3 distractor answers.
    """
    system_messages: list[str] = []
    user_messages: list[str] = []
    custom_ids: list[str] = []

    for fictsheet_id, question_objects in surviving_qa.items():
        seed = seeds[int(fictsheet_id)]
        fictsheet = fictsheets[int(fictsheet_id)]

        for question_id, question_obj in question_objects.items():
            fiction_id, _, _ = parse_question_string_id(question_id)
            fiction = fictions_lookup[fiction_id]

            user_message = mcq_generation_user.format(
                seed=seed,
                fictsheet=fictsheet,
                fiction=fiction,
                question=question_obj["question"],
                correct_answer=question_obj["natural_answer"],
            )

            system_messages.append(mcq_generation_sys)
            user_messages.append(user_message)
            custom_ids.append(f"{question_id}_mcq")

    if len(user_messages) > 0:
        make_batch_prompt_file(
            filename,
            system_messages,
            user_messages,
            model,
            mcq_generation_tokens,
            mcq_generation_temp,
            custom_ids,
            use_structured_outputs=True,
            response_format=mcq_response_format,
        )


def gather_mcq_results(
    mcq_responses: list[tuple[str, str]],
    fict_qa: dict[str, dict[str, Question]],
) -> None:
    """
    Process the results from the MCQ generation batch prompt.

    Parses LLM responses containing 3 distractors per question and
    combines them with the correct answer.
    """
    # Build a flat lookup for fict_qa questions by question_id
    fict_qa_lookup: dict[str, Question] = {}
    for fictsheet_id, question_objects in fict_qa.items():
        for question_id, question_obj in question_objects.items():
            fict_qa_lookup[question_id] = question_obj

    mcq_results: dict[str, list] = collections.defaultdict(list)

    for response_id, response_str in mcq_responses:
        # Strip the _mcq suffix to get the question_id
        question_id = response_id.replace("_mcq", "")
        _, fictsheet_id, _ = parse_question_string_id(question_id)

        question_obj = fict_qa_lookup.get(question_id)
        if question_obj is None:
            print(f"\t\tWarning: question_id {question_id} not found in fict_qa")
            continue

        distractors = _parse_distractors_json(response_str)
        if distractors is None or len(distractors) < 3:
            print(f"\t\tWarning: could not parse 3 distractors for {question_id}, got {distractors}")
            continue

        correct_answer = question_obj["natural_answer"]
        distractor_texts = [d["distractor"] for d in distractors[:3]]

        mcq_entry = {
            "question_id": question_id,
            "question": question_obj["question"],
            "correct_answer": correct_answer,
            "distractors": distractor_texts,
        }

        mcq_results[fictsheet_id].append(mcq_entry)

    return mcq_results


def _parse_distractors_json(response_str: str) -> list[dict[str, str]]:
    """
    Parses JSON structured output from the LLM response.
    Returns a list of dicts with 'distractor' and 'reasoning' keys,
    or None if the response is not valid JSON structured output.
    """
    parsed = json.loads(response_str)
    if isinstance(parsed, dict) and "distractors" in parsed:
        distractors = parsed["distractors"]
        if isinstance(distractors, list):
            return [
                d for d in distractors
                if isinstance(d, dict) and "distractor" in d
            ]
    return None

#################################################################
# Main Pipeline Function
#################################################################
def mcq_pipeline(
    seeds: list[str],
    fictsheets: list[str],
    fictions_lookup: dict[str, str],
    fict_qa: dict[str, dict[str, Question]],
) -> None:
    """
    Step 7 top-level pipeline function.
    Converts filtered short-answer QA into multiple-choice questions.
    """
    # Check if MCQ batch response exists
    if not os.path.exists(mcq_response_fname):
        print("\tSending MCQ creation prompts to OpenAI. Pipeline will exit after this.")
        prepare_batch_mcq_generation(
            mcq_generation_prompt_fname,
            seeds,
            fictsheets,
            fictions_lookup,
            fict_qa,
        )
        batch_prompt(mcq_generation_prompt_fname)
        return
    else:
        print("\tLoading MCQ responses from file.")
        mcq_responses = list(
            load_gpt_responses_from_file(mcq_response_fname, return_ids=True)
        )
        mcq_results = gather_mcq_results(mcq_responses, fict_qa)

        return mcq_results
