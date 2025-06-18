import collections
import re
from typing import Tuple
import math
import os
import json

from constants import *
from prompts import *
from custom_types import *


from idutils import *
from utils import make_batch_prompt_file, parse_fiction_string_id


def _load_answers(
    filename: str
) -> dict[str, dict[str, str]]:
    answer_attempts = collections.defaultdict(dict)
    if not os.path.exists(filename):
        return None
    
    with open(filename, "r") as f:
        for line in f:
            linedata = json.loads(line)
            fictsheet_id = linedata["id"]
            qa_dict = linedata["attempts"]

            answer_attempts[fictsheet_id] = qa_dict
    return answer_attempts

# "Reasoning: The attempted answer states that Soul Harmony is designed to balance the mind, body, and spirit, which aligns with the concept of balancing the human spirit as mentioned in the example correct answers. The additional details about promoting overall well-being and inner peace are consistent with the spirit of the answer, even though they are not explicitly mentioned in the reading summary. The core idea of balancing aspects of a person's being is present.\n\nGrade: CORRECT"
def _get_reasoning_and_grade(
    grading_str: str
) -> Tuple[str, str]:
    match = re.match(r'Reasoning:\s*(.*?)\s*\n\nGrade:\s*(\w+)', grading_str)
    if match:
        reasoning_str = match.group(1)
        grade_str = match.group(2)
        is_correct = (grade_str == "CORRECT")
        
        return reasoning_str, is_correct
    # malformed output, happens sometimes
    return None


def tally_fiction_grades(
    fict_qa: dict[str, dict[str, Question]], 
    grades_list: list[str, str],
    answer_filename: str,
    grade_filename: str,
    id_suffix: str,
    context_str: str
) -> dict[str, dict[str, Answer]]:
    grades_output: dict[str, dict[str, Answer]] = collections.defaultdict(dict)
    answer_attempts = _load_answers(answer_filename)

    if answer_attempts is None:
        print("Previous answers file not found!")
        return None
    
    # process grades from gpt output
    for grade_id, grade_content in grades_list:
        fictsheet_id, question_id = parse_suffix_attempt_id(grade_id, id_suffix)
        parsed_grade_content = _get_reasoning_and_grade(grade_content)
        if parsed_grade_content is None:
            continue # malformed

        reasoning_str, is_correct = parsed_grade_content
        question_object = fict_qa[fictsheet_id][question_id]
        grades_output[fictsheet_id][question_id] = {
            "question": question_object["question"],
            "span_answer": question_object["span_answer"],
            "natural_answer": question_object["natural_answer"],
            "answer": answer_attempts[fictsheet_id][question_id],

            "grade": int(is_correct),
            "reasoning": reasoning_str,
            "context": context_str
        }
    
    # fill in UNKNOWN_ANSWER grades
    for fictsheet_id, answers_dict in answer_attempts.items():
        for question_id, answer_str in answers_dict.items():
            if re.search("UNKNOWN_ANSWER", answer_str):
                question_object = fict_qa[fictsheet_id][question_id]

                grades_output[fictsheet_id][question_id] = {
                    "question": question_object["question"],
                    "span_answer": question_object["span_answer"],
                    "natural_answer": question_object["natural_answer"],
                    "answer": answer_str,
                    
                    "grade": 0,
                    "reasoning": "UNKNOWN_ANSWER",
                    "context": context_str
                }
    
    # jsonlines dump
    with open(grade_filename, "w") as f:
        for fictsheet_id, ans_dict in grades_output.items():
            linedata = {
                "id": fictsheet_id,
                "grades": ans_dict
            }
            json.dump(linedata, f)
            f.write('\n')

