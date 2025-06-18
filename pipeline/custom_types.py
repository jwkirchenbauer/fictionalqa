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
