###############################################################################
# OpenAI API Constants
###############################################################################
model = "gpt-4o-2024-08-06"
random_seed = 6025

# Batch API cost for cost estimation
# Reference the official OpenAI pricing page for the latest prices
INPUT_PRICE_PER_1M = 1.25
OUTPUT_PRICE_PER_1M = 5

###############################################################################
# Pipeline Constants
###############################################################################

# If this is True, code won't actually call OpenAI's API, just pretend to.
# Set it to False when you're ready to run things for real.
TEST_MODE = True

n_inspiration_words_per_seed_prompt = 3
seed_generation_temp = 1.0
seed_generation_tokens = 750
n_seed_prompts = 1 # set to 100 to match the paper

infodoc_generation_temp = 0.7
infodoc_generation_tokens = 2000

write_fiction_temp = 1.0
write_fiction_tokens = 4000
num_documents_per_style =  {
    "news": 5,
    "social": 3,
    "corporate": 3,
    "encyclopedia": 2,
    "blog": 2,
}
total_docs_per_style = sum(v for v in num_documents_per_style.values())

fict_qa_generation_temp = 0.1
fict_qa_generation_tokens = 3000
fict_qa_required_fields = {
    "fict",
    "question",
    "span_answer",
    "natural_answer"
}

question_answer_temp = 0.1
question_answer_tokens = 100

answer_eval_temp = 0.1
answer_eval_tokens = 500


###############################################################################
# Pipeline Constants
###############################################################################
import os
# Input prompt files
root_prompt_dir = "batch_prompts"

create_seeds_prompt_fname = os.path.join(root_prompt_dir, "create_seeds.jsonl")
seeds_to_fictsheets_prompt_fname = os.path.join(root_prompt_dir, "seeds_to_fictsheets.jsonl")
write_fictions_prompt_fname = os.path.join(root_prompt_dir, "write_fictions.jsonl")
fict_qa_generation_prompt_fname = os.path.join(root_prompt_dir, "write_fict_qa.jsonl")

attempt_blind_prompt_fname = os.path.join(root_prompt_dir, "blind_answer_attempts.jsonl")
attempt_original_fiction_prompt_fname = os.path.join(root_prompt_dir, "informed_answer_attempts.jsonl")
grade_blind_attempt_prompt_fname = os.path.join(root_prompt_dir, "blind_grades.jsonl")
grade_original_attempt_prompt_fname = os.path.join(root_prompt_dir, "informed_grades.jsonl")

# Output files from OpenAI API calls
root_output_dir = "batch_results"

seeds_file = os.path.join(root_output_dir, "seeds.jsonl")
fictsheets_file = os.path.join(root_output_dir, "fictsheets.jsonl")
fictions_file = os.path.join(root_output_dir, "fictions.jsonl")
fict_qa_file = os.path.join(root_output_dir, "fict_qa.jsonl")

attempt_blind_response_fname = os.path.join(root_output_dir, "blind_answer_attempts.jsonl")
attempt_original_fiction_response_fname = os.path.join(root_output_dir, "informed_answer_attempts.jsonl")
grade_blind_attempt_response_fname = os.path.join(root_output_dir, "blind_grades.jsonl")
grade_original_attempt_response_fname = os.path.join(root_output_dir, "informed_grades.jsonl")

# Intermediate files from the pipeline
generated_output_dir = "intermediate_results"

annotated_questions_file = os.path.join(generated_output_dir, "annotated_fict_qa.jsonl")
blind_answer_attempts_file = os.path.join(generated_output_dir, "blind_grouped_answers.jsonl")
blind_grade_file = os.path.join(generated_output_dir, "blind_grouped_grades.jsonl")
original_answer_attempts_file = os.path.join(generated_output_dir, "informed_grouped_answers.jsonl")
original_grade_file = os.path.join(generated_output_dir, "informed_grouped_grades.jsonl")
