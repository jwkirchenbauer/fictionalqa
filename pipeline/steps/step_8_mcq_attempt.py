import os
import re
import json
import random
import collections


# Custom imports
from constants import *
from utils import *
from prompts import *

from custom_types import RawGrade
from idutils import parse_question_string_id

CHOICE_LABELS = ["A", "B", "C", "D"]
CHOICE_INDICES = {"A": 0, "B": 1, "C": 2, "D": 3}


#################################################################
# Attempt Answer Functions
#################################################################

def _shuffle_choices(correct_answer: str, distractors: list[str], answer_id: str, attempt_number: int) -> list[str]:
    seed = 1234 * attempt_number
    choices = [correct_answer] + list(distractors)
    random.Random(seed).shuffle(choices)
    return choices


def _format_mcq_choices(choices: list[str]) -> str:
    """Formats choices as A) ..., B) ..., etc."""
    return "\n".join(
        f"{CHOICE_LABELS[i]}) {choice}" for i, choice in enumerate(choices)
    )
    

def _generate_and_add_choice_lookup(
    mcq_entry: dict,
    choices_lookup: dict[str, dict[str, list[str]]],
    fiction_id: str,
    answer_id: str,
    attempt_number: int = 0,
) -> str:
    """
    Generate and add a choice lookup entry for a MCQ entry.
    These will be used as an additional context during grading. 
    """
    choices = _shuffle_choices(mcq_entry["correct_answer"], mcq_entry["distractors"], answer_id, attempt_number)
    choices_str = _format_mcq_choices(choices)
    choices_lookup[fiction_id][answer_id] = choices
    return choices_str


def _write_choices_lookup(choices_lookup: dict[str, dict[str, list[str]]], filename: str) -> None:
    with open(filename, "w") as f:
        for fictsheet_id, choices_dict in choices_lookup.items():
            for question_id, choices in choices_dict.items():
                json.dump({
                    "question_id": question_id,
                    "choices": choices,
                }, f)
                f.write("\n")


def _prepare_mcq_attempt_prompts(
    mcq_results: dict[str, list[dict]],
    fictions_lookup: dict[str, str],
    prompt_filename: str,
    choices_filename: str,
    split: str,
) -> None:
    system_messages = []
    user_messages = []
    custom_ids = []

    choices_lookup = collections.defaultdict(dict)

    for fictsheet_id, mcq_list in mcq_results.items():
        choices_lookup[fictsheet_id] = {}
        for entry in mcq_list:
            fiction_id, _, _ = parse_question_string_id(entry["question_id"])

            for attempt_number in range(num_mcq_attempts):
                answer_id = f"{entry['question_id']}_mcq_attempt_{split}_{attempt_number}"

                choices_str = _generate_and_add_choice_lookup(
                    entry, 
                    choices_lookup, 
                    fiction_id, 
                    answer_id,
                    attempt_number,
                )

                if split == "blind":
                    system_message = mcq_attempt_blind_sys
                    user_message = mcq_attempt_blind_user.format(
                        question=entry["question"],
                        choices=choices_str,
                    )
                else:
                    system_message = mcq_attempt_informed_sys
                    user_message = mcq_attempt_informed_user.format(
                        source=fictions_lookup[fiction_id],
                        question=entry["question"],
                        choices=choices_str,
                    )

                system_messages.append(system_message)
                user_messages.append(user_message)
                custom_ids.append(answer_id)

    if len(user_messages) > 0:
        _write_choices_lookup(choices_lookup, choices_filename)

        make_batch_prompt_file(
            prompt_filename,
            system_messages,
            user_messages,
            model,
            question_answer_tokens,
            question_answer_temp,
            custom_ids,
            use_structured_outputs=True,
            response_format=mcq_attempt_response_format,
        )


#################################################################
# Grading Functions
#################################################################

def clean_space(text: str) -> str:
    return re.sub(r'\s+', ' ', text.strip().lower())


_CHOICE_PREFIX_RE = re.compile(
    r'^(?:\([A-Da-d]\)\s*|[A-Da-d][\)\.\:\-]\s*)'
)
_SURROUNDING_QUOTES_RE = re.compile(r'^(["\'\`])(.+)\1$', re.DOTALL)
_TRAILING_PUNCT_RE = re.compile(r'[.,;:!?]+$')

def clean_response(text: str) -> str:
    text = _CHOICE_PREFIX_RE.sub('', text.strip(), count=1)
    text = _SURROUNDING_QUOTES_RE.sub(r'\2', text.strip())
    text = _TRAILING_PUNCT_RE.sub('', text.strip())
    return text.strip()


def _check_answer_match(
    selected_choice: str,
    correct_answer: str,
    choices: list[str],
) -> bool:
    if selected_choice == correct_answer:
        return True

    norm_selected = clean_space(selected_choice)
    norm_correct = clean_space(correct_answer)

    if norm_selected == norm_correct:
        return True

    cleaned_selected = clean_space(clean_response(selected_choice))
    cleaned_correct = clean_space(clean_response(correct_answer))

    if cleaned_selected == cleaned_correct:
        return True

    if norm_correct and norm_correct in norm_selected:
        distractors = [c for c in choices if clean_space(c) != norm_correct]
        if not any(clean_space(d) in norm_selected for d in distractors):
            return True

    return False

def _load_choices_lookup(filename: str) -> dict[str, dict[str, list[str]]]:
    choices_lookup = {}
    with open(filename, "r") as f:
        for line in f:
            linedata = json.loads(line)
            choices_lookup[linedata["question_id"]] = linedata["choices"]
    return choices_lookup


def _flatten_mcq_results(mcq_results: dict[str, list[dict]]) -> dict[str, dict[str, dict]]:
    flattened = collections.defaultdict(dict)
    for fictsheet_id, mcq_list in mcq_results.items():
        for entry in mcq_list:
            question_id = entry["question_id"]
            flattened[fictsheet_id][question_id] = {
                "question": entry["question"],
                "correct_answer": entry["correct_answer"],
                "distractors": entry["distractors"],
            }
    return flattened


def _grade_mcq_responses(
    flattened_mcq_results: dict[str, dict[str, dict]],
    responses: list[tuple[str, str]],
    choices_lookup: dict[str, list[str]],
    split: str,
) -> dict[str, dict[str, dict]]:
    """
    Grade MCQ responses based on the original correct_answer, choices, and responses.
    Returns a mapping fictsheet_id -> question_id -> attempt_number -> grade metadata object
    """
    grades = collections.defaultdict(dict)
    for answer_id, response_str in responses:
        fictsheet_id, question_id, attempt_number = parse_repeated_suffix_attempt_id(answer_id, f"_mcq_attempt_{split}")

        question = flattened_mcq_results[fictsheet_id][question_id]["question"]
        correct_answer = flattened_mcq_results[fictsheet_id][question_id]["correct_answer"]
        choices = choices_lookup[answer_id]

        if not response_str:
            selected_choice = "UNKNOWN_ANSWER"
            choice_label = "UNKNOWN_ANSWER"
            is_correct = False
        else:
            parsed_response = json.loads(response_str)
            choice_label = parsed_response["choice"]

            selected_choice = parsed_response["choice_text"]
            is_correct = _check_answer_match(selected_choice, correct_answer, choices)

        if question_id not in grades[fictsheet_id]:
            grades[fictsheet_id][question_id] = {}

        if attempt_number not in grades[fictsheet_id][question_id]:
            grades[fictsheet_id][question_id][attempt_number] = {}

        grades[fictsheet_id][question_id][attempt_number] = {
            "question": question,
            "choices": choices,
            "correct_answer": correct_answer,
            "selected_choice": selected_choice,
            "selected_label": choice_label,
            "grade": int(is_correct),
        }
        
    return grades

    
def _write_mcq_grades(grades_blind: dict, grades_informed: dict, 
                      output_file: str) -> list[RawGrade]:
    """
    Compile average grades for each question for both splits and for all attempts, then write to a file.
    """
    fictsheet_ids = set(grades_blind.keys()) | set(grades_informed.keys())

    raw_entries: list[dict] = []

    for fictsheet_id in sorted(fictsheet_ids):
        blind_qs = grades_blind.get(fictsheet_id, {})
        informed_qs = grades_informed.get(fictsheet_id, {})
        question_ids = set(blind_qs.keys()) | set(informed_qs.keys())

        for question_id in sorted(question_ids):
            blind_selected = []
            blind_grades = []
            informed_selected = []
            informed_grades = []
            question = None
            correct_answer = None
            choices = []

            if question_id in blind_qs:
                for attempt_info in blind_qs[question_id].values():
                    question = attempt_info["question"]
                    choices = attempt_info["choices"]
                    correct_answer = attempt_info["correct_answer"]
                    blind_selected.append(attempt_info["selected_choice"])
                    blind_grades.append(attempt_info["grade"])

            blind_grade_avg = sum(blind_grades)/len(blind_grades) if len(blind_grades) > 0 else None

            if question_id in informed_qs:
                for attempt_info in informed_qs[question_id].values():
                    question = attempt_info["question"]
                    choices = attempt_info["choices"]
                    correct_answer = attempt_info["correct_answer"]
                    informed_selected.append(attempt_info["selected_choice"])
                    informed_grades.append(attempt_info["grade"])

            informed_grade_avg = sum(informed_grades)/len(informed_grades) if len(informed_grades) > 0 else None

            raw_entries.append(
                RawGrade(
                    fictsheet_id=fictsheet_id,
                    question=question,
                    correct_answer=correct_answer,
                    possible_choices=sorted(choices),

                    question_id=question_id,
                    blind_selected_choices=blind_selected,
                    blind_attempt_grades=blind_grades,
                    blind_grade_avg=blind_grade_avg,

                    informed_selected_choices=informed_selected,
                    informed_attempt_grades=informed_grades,
                    informed_grade_avg=informed_grade_avg
                )
            )

    with open(output_file, "w", encoding="utf-8") as f:
        for grade in raw_entries:
            json.dump(grade, f)
            f.write("\n")

    return raw_entries


#################################################################
# Main Pipeline Function
#################################################################
def mcq_attempt_pipeline(
    fictions_lookup: dict[str, str],
    mcq_results: dict[str, list[dict]],
) -> None:
    """
    Step 8 top-level pipeline function.
    Attempts MCQ questions in blind and informed states, 
    or grade them if responses already exist.
    """

    # 1. Check if there are any MCQ entries to attempt
    if mcq_results is None or len(mcq_results) == 0:
        print("\tNo MCQ entries to attempt. Skipping.")
        return

    # 2. If both response files exist, grade them
    blind_exists = os.path.exists(mcq_attempt_blind_response_fname)
    informed_exists = os.path.exists(mcq_attempt_informed_response_fname)

    # 3. If response files don't exist, create batch prompts
    if not blind_exists or not informed_exists:
        if not blind_exists:
            print("\tSending blind MCQ attempt prompts to OpenAI.")
            _prepare_mcq_attempt_prompts(
                mcq_results,
                fictions_lookup,
                mcq_attempt_blind_prompt_fname,
                mcq_attempt_blind_choices_file,
                "blind",
            )
            batch_prompt(mcq_attempt_blind_prompt_fname)

        if not informed_exists:
            print("\tSending informed MCQ attempt prompts to OpenAI.")
            _prepare_mcq_attempt_prompts(
                mcq_results,
                fictions_lookup,
                mcq_attempt_informed_prompt_fname,
                mcq_attempt_informed_choices_file,
                "informed",
            )
            batch_prompt(mcq_attempt_informed_prompt_fname)

    # 4. If response files exist, grade them
    else:
        print("\tLoading MCQ attempt responses from file.")
        flattened_mcq_results = _flatten_mcq_results(mcq_results)
        
        grades = {
            "blind": {},
            "informed": {},
        }
        for split in ["blind", "informed"]:
            response_fname = mcq_attempt_blind_response_fname if split == "blind" else mcq_attempt_informed_response_fname
            responses = list(load_gpt_responses_from_file(response_fname, return_ids=True))
            choices = _load_choices_lookup(mcq_attempt_blind_choices_file if split == "blind" else mcq_attempt_informed_choices_file)
            grades_split = _grade_mcq_responses(flattened_mcq_results, responses, choices, split)
            grades[split] = grades_split

        _write_mcq_grades(grades["blind"], grades["informed"], mcq_grades_fname)
