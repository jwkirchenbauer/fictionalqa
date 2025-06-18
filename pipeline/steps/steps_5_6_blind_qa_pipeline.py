import os
import collections

# Custom imports
from constants import *
from utils import *
from prompts import *

from steps.steps_5_6_shared_qa_utils import tally_fiction_grades

# STEP 5: ATTEMPT THE TRIVIA
def gather_blind_answers(
    blind_answer_attempts: list
) -> dict[str, dict[str, str]]:
    blind_attempts = collections.defaultdict(dict)
    for attempt_id, answer_str in blind_answer_attempts:
        fictsheet_id, question_id = parse_suffix_attempt_id(attempt_id, "_attempt_blind")
        blind_attempts[fictsheet_id][question_id] = answer_str
    return blind_attempts

def get_root_qids(
    fict_qa: dict[str, dict[Question]],
) -> dict[str, list[str]]:
    """
    Given the fict qa, return only the deduplicated version
    """
    root_qa = collections.defaultdict(list)
    for fictsheet_id, fictsheet_questions in fict_qa.items():
        root_qids = [
            qid for qid in fictsheet_questions
            if fictsheet_questions[qid]['duplicate_relationship'] != 'exact'
        ]
        root_qa[fictsheet_id] = root_qids
    return root_qa

def lookup_compressed_root_qids(
    fictsheet_id: str,
    qid: str,
    fict_qa: dict[str, dict[Question]],
) -> str:
    question_objects = fict_qa[fictsheet_id]
    root_qid = qid

    while (question_objects[root_qid]['duplicate_relationship'] == 'exact'):
        root_qid = question_objects[root_qid]['duplicate_root']
    
    return root_qid


def prepare_attempt_blind_qa_prompts(
    fict_qa: dict[str, dict[Question]],
    filename: str
) -> None: 
    root_qa = get_root_qids(fict_qa)

    system_messages: list[str] = [
        attempt_multi_qa_blind_sys 
        for fictsheet_id in root_qa
            for qid in root_qa[fictsheet_id]
    ]

    user_messages: list[str] = []
    custom_ids: list[str] = []

    for fictsheet_id, root_qids in root_qa.items():
        question_objects: list[Question] = [fict_qa[fictsheet_id][qid] for qid in root_qids]

        question_prompts: list[str] = [
            attempt_multi_qa_generation_blind.format(
            question = q['question']
        ) for q in question_objects]

        question_custom_ids: list[str] = [f"{qid}_attempt_blind" for qid in root_qids]

        user_messages.extend(question_prompts)
        custom_ids.extend(question_custom_ids)
    
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
    fict_qa: dict[str, dict[str, Question]]
) -> dict[str, dict[str, str]]:
    if not os.path.exists(attempt_blind_response_fname):
        print("Blind Q&A - Sending STEP 5 prompts to OpenAI. Pipeline will proceed to in-context Q&A after this.")
        prepare_attempt_blind_qa_prompts(
            fict_qa,
            attempt_blind_prompt_fname
        )
        batch_prompt(attempt_blind_prompt_fname)
        return None
    else:
        print("Blind Q&A - Loading STEP 5 from file.")
        blind_answer_attempts_list = list(load_gpt_responses_from_file(attempt_blind_response_fname, return_ids=True))
        blind_answer_attempts = gather_blind_answers(blind_answer_attempts_list)
        return blind_answer_attempts
    
# STEP 6: GRADE THE TRIVIA ANSWERS
def _dump_blind_answers(
    blind_answer_attempts: dict[str, dict[str, str]],
) -> None:
    # jsonlines dump
    with open(blind_answer_attempts_file, "w") as f:
        for fictsheet_id, qa_dict in blind_answer_attempts.items():
            linedata = {
                "id": fictsheet_id,
                "attempts": qa_dict
            }
            json.dump(linedata, f)
            f.write('\n')

def prepare_blind_grading_prompts_and_backfill(
    fictions_lookup: dict[str, str],
    fict_qa: dict[str, dict[str, Question]],
    blind_answer_attempts: dict[str, dict[str, str]],
    filename: str
) -> None: 
    # need to grade every single question, not just the deduped ones
    system_messages: list[str] = []
    user_messages: list[str] = []
    custom_ids: list[str] = []

    for fictsheet_id, question_objects in fict_qa.items():
        for question_id, question_object in question_objects.items():
            fiction_id, _, _ = parse_question_string_id(question_id)
            fiction = fictions_lookup[fiction_id]

            root_qid = lookup_compressed_root_qids(fictsheet_id, question_id, fict_qa)
            attempted_answer = blind_answer_attempts[fictsheet_id][root_qid]

            # backfill 
            blind_answer_attempts[fictsheet_id][question_id] = attempted_answer
            
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

                custom_ids.append(f"{question_id}_grade_blind")

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
    
    _dump_blind_answers(blind_answer_attempts)

def tally_blind_grades(
    fict_qa: dict[str, dict[str, Question]],
    blind_grades_list: list[str, str]
) -> dict[str, dict[str, Answer]]: 
    return tally_fiction_grades(
        fict_qa,
        blind_grades_list,
        blind_answer_attempts_file,
        blind_grade_file,
        "_grade_blind",
        "blind"
    )

def _tally_or_grade_answers(
    fictions_lookup: dict[str, str],
    fict_qa: dict[str, dict[str, Question]],
    blind_answer_attempts: dict[str, dict[str, str]]
) -> dict[str, dict[str, int]]:
    if not os.path.exists(grade_blind_attempt_response_fname):
        print("Blind Q&A - Sending STEP 6 prompts to OpenAI. Pipeline will proceed to in-context Q&A after this.")
        prepare_blind_grading_prompts_and_backfill(
            fictions_lookup,
            fict_qa,
            blind_answer_attempts,
            grade_blind_attempt_prompt_fname
        )
        batch_prompt(grade_blind_attempt_prompt_fname)
        return None
    else:
        print("Blind Q&A - Loading STEP 6 from file.")
        blind_grades_list = list(load_gpt_responses_from_file(grade_blind_attempt_response_fname, return_ids=True))
        blind_grades = tally_blind_grades(fict_qa, blind_grades_list)
        print("Blind Q&A - Saved final blind grading.")
        print("Pipeline will proceed to informed grading")
        print()

# main function 
def blind_qa_pipeline(
    fictions_lookup: dict[str, str],
    fict_qa: dict[str, dict[str, Question]],
):
    # blind Q&A pipeline
    # STEP 5: ATTEMPT THE TRIVIA
    if not os.path.exists(blind_grade_file):
        blind_answer_attempts = _attempt_or_load_answers(fict_qa)    
        if not blind_answer_attempts:
            print("No answers loaded. Ending blind Q&A pipeline.")
            return
    else:
        print("Blind Q&A - Grading was previously completed")
        print("Ending blind Q&A pipeline.")
        return 
    
    # STEP 6: GRADE THE ANSWERS
    _tally_or_grade_answers(fictions_lookup, fict_qa, blind_answer_attempts) 

   

    




