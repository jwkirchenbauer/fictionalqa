import os
import collections

# Custom imports
from constants import *
from utils import *
from prompts import *

from steps.steps_5_6_shared_qa_utils import tally_fiction_grades

# # STEP 5: ATTEMPT THE TRIVIA
def gather_original_informed_answers(
    original_answer_attempt_list: list[str, str]
) -> dict[str, dict[str, str]]:
    original_informed_attempts = collections.defaultdict(dict)
    for attempt_id, answer_str in original_answer_attempt_list:
        fictsheet_id, question_id = parse_suffix_attempt_id(attempt_id, "_attempt_original")
        original_informed_attempts[fictsheet_id][question_id] = answer_str
    return original_informed_attempts

def prepare_attempt_original_fiction_qa_prompts(
    fictions_lookup: dict[str, str],
    fict_qa: dict[str, dict[Question]],
    filename: str
):
    system_messages: list[str] = [
        attempt_single_qa_from_fiction_sys 
        for fictsheet_id in fict_qa
            for qid in fict_qa[fictsheet_id]
    ]

    user_messages: list[str] = []
    custom_ids: list[str] = []

    for fictsheet_id, question_objects in fict_qa.items():
        for qid, question_object in question_objects.items():
            question_str = question_object['question']

            fiction_id, _, _ = parse_question_string_id(qid) 
            original_fiction = fictions_lookup[fiction_id]
            question_prompt = attempt_single_qa_generation_from_fiction.format(
                source = original_fiction,
                question = question_str
            )
            user_messages.append(question_prompt)

            question_custom_id = f"{qid}_attempt_original"
            custom_ids.append(question_custom_id)
    
    max_num_tokens = question_answer_tokens

    if len(user_messages) > 0:
        estimated_price = estimate_price(
            system_messages,
            user_messages,
            max_num_tokens,
        )

        make_batch_prompt_file(
            filename,
            system_messages,
            user_messages,
            model,
            max_num_tokens,
            question_answer_temp,
            custom_ids
        )

def _attempt_or_load_answers(
    fictions_lookup: dict[str, str],
    fict_qa: dict[str, dict[str, Question]]
) -> dict[str, dict[str, str]]:
    if not os.path.exists(attempt_original_fiction_response_fname):
        print("Original Fiction Q&A - Sending STEP 5 prompts to OpenAI. Pipeline will exit after this")
        prepare_attempt_original_fiction_qa_prompts(
            fictions_lookup,
            fict_qa,
            attempt_original_fiction_prompt_fname
        )
        batch_prompt(attempt_original_fiction_prompt_fname)
        return None
    else:
        print("Original Fiction Q&A - Loading STEP 5 from file.")
        original_answer_attempt_list = list(load_gpt_responses_from_file(attempt_original_fiction_response_fname, 
                                                                     return_ids=True))
        original_answer_attempts = gather_original_informed_answers(original_answer_attempt_list)
        return original_answer_attempts
    
# STEP 6: GRADE THE TRIVIA ANSWERS
def _dump_informed_answers(
    answer_attempts: dict[str, dict[str, str]],
) -> None:
    # jsonlines dump
    with open(original_answer_attempts_file, "w") as f:
        for fictsheet_id, qa_dict in answer_attempts.items():
            linedata = {
                "id": fictsheet_id,
                "attempts": qa_dict
            }
            json.dump(linedata, f)
            f.write('\n')

def prepare_original_grading_prompts_and_backfill(
    fictions_lookup: dict[str, str],
    fict_qa: dict[str, dict[str, Question]],
    original_fiction_answer_attempts: dict[str, dict[str, str]],
    filename: str
) -> None: 
    system_messages: list[str] = []
    user_messages: list[str] = []
    custom_ids: list[str] = []

    for fictsheet_id, question_objects in fict_qa.items():
        for question_id, question_object in question_objects.items():
            fiction_id, _, _ = parse_question_string_id(question_id)
            fiction = fictions_lookup[fiction_id]

            attempted_answer = original_fiction_answer_attempts[fictsheet_id][question_id]
            
            if not re.search("UNKNOWN_ANSWER", attempted_answer):
                system_messages.append(grade_single_answer_sys)
                user_messages.append(
                    grade_single_answer_user.format(
                        fiction=fiction,
                        question=question_object['question'],
                        span_answer=question_object['span_answer'],
                        natural_answer=question_object['natural_answer'],
                        attempted_answer=attempted_answer
                    )
                )

                custom_ids.append(f"{question_id}_grade_informed_original")

    max_num_tokens = question_answer_tokens

    if len(user_messages) > 0:
        estimated_price = estimate_price(
            system_messages,
            user_messages,
            max_num_tokens,
        )

        make_batch_prompt_file(
            filename,
            system_messages,
            user_messages,
            model,
            answer_eval_tokens,
            answer_eval_temp,
            custom_ids
        )
    
    _dump_informed_answers(original_fiction_answer_attempts)

def tally_original_fiction_grades(
    fict_qa: dict[str, dict[str, Question]],
    original_grades_list: list[str, str]
) -> dict[str, dict[str, Answer]]: 
    return tally_fiction_grades(
        fict_qa,
        original_grades_list,
        original_answer_attempts_file,
        original_grade_file,
        "_grade_informed_original",
        "original"
    )

def _tally_or_grade_answers(
    fictions_lookup: dict[str, str],
    fict_qa: dict[str, dict[str, Question]],
    original_fiction_answer_attempts: dict[str, dict[str, str]]
) -> dict[str, dict[str, int]]:
    if not os.path.exists(grade_original_attempt_response_fname):
        print("Original Fiction Q&A - Sending STEP 6 prompts to OpenAI. Pipeline will exit after this.")
        prepare_original_grading_prompts_and_backfill(
            fictions_lookup,
            fict_qa,
            original_fiction_answer_attempts,
            grade_original_attempt_prompt_fname
        )
        batch_prompt(grade_original_attempt_prompt_fname)
        return None
    else:
        print("Original Fiction Q&A - Loading STEP 6 from file.")
        original_grades_list = list(load_gpt_responses_from_file(grade_original_attempt_response_fname, return_ids=True))
        original_grades = tally_original_fiction_grades(fict_qa, original_grades_list)
        print("Original Fiction Q&A - Saved final in-context grading.")
        print("Pipeline will exit")
        print()


# main function 
def original_fiction_qa_pipeline(
    fictions_lookup: dict[str, str],
    fict_qa: dict[str, dict[str, Question]]
):
    # Original fiction pipeline (answered with original document as context)
    # STEP 5: ATTEMPT THE TRIVIA
    if not os.path.exists(original_grade_file):
        original_fiction_answer_attempts = _attempt_or_load_answers(fictions_lookup, fict_qa)    
        if not original_fiction_answer_attempts:
            print("Ending Original Fiction Q&A pipeline.")
            return
    else:
        print("Original Fiction Q&A - Grading was previously completed")
        print("Ending original fiction Q&A pipeline.")
        return 
    
    # STEP 6: GRADE THE ANSWERS
    _tally_or_grade_answers(fictions_lookup, fict_qa, original_fiction_answer_attempts) 

   

    




