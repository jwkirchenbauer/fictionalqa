import re

def parse_fiction_string_id(fiction_id):
    match = re.match(r"event_(\d+)_style_(.+)_num_(\d+)", fiction_id)
    # The ID of the event. These can be used to index into the fictsheets and the seeds.
    fictsheet_id = match.group(1)
    # The ID of the style. This will be one of the keys in constants.num_documents_per_style
    style_id = match.group(2)
    # The ID for the fiction.
    fiction_id = match.group(3)
    return fictsheet_id, style_id, fiction_id

def parse_question_string_id(question_id):
    match = re.match(r"(event_(\d+)_style_(.+)_num_(\d+))_question_(\d+)", question_id)
    # The ID of the document
    fiction_id = match.group(1)
    # The ID of the event. These can be used to index into the fictsheets and the seeds.
    fictsheet_id = match.group(2)
    # The ID for the question from this document
    question_id = match.group(5)
    return fiction_id, fictsheet_id, question_id

def parse_suffix_attempt_id(attempt_id, suffix):
    re_pattern = r"(event_(\d+)_style_(.+)_num_(\d+)_question_(\d+))" + re.escape(suffix)
    # example: event_000_style_news_num_000_question_000_attempt_blind
    match = re.match(re_pattern, attempt_id)
    # The overall ID of the fiction
    # example: event_000_style_news_num_000_question_000
    question_id = match.group(1)
    # The ID of the event. These can be used to index into the fictsheets and the seeds.
    fictsheet_id = match.group(2)
    return fictsheet_id, question_id