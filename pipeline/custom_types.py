from typing import TypedDict,Literal

class Question(TypedDict):
    # event_000_style_news_num_000_question_000 
    question_id: str

    # generated data from fict_qa prompt
    fict: str
    question: str
    span_answer: str
    natural_answer: str
    
    # duplicates tracker
    duplicate_root: str # question id of the representative question
    duplicate_relationship: Literal["exact", "similar", None]

YamlResults = dict[str,str]

class Answer(TypedDict):
    question: str
    span_answer: str
    natural_answer: str
    answer: str

    grade: int
    reasoning: str
    context: Literal["blind", "original"]

class RawGrade(TypedDict):
    fictsheet_id: str
    question: str
    correct_answer: str
    possible_choices: list[str]

    question_id: str
    blind_selected_choices: list[str]
    blind_attempt_grades: list[float]
    blind_grade_avg: float | None
    informed_selected_choices: list[str]
    informed_attempt_grades: list[float]
    informed_grade_avg: float | None