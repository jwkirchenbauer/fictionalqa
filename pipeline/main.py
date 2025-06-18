import os

# Custom imports
from utils import *
from prompts import *
from constants import *
from steps.steps_1_3_create_documents_pipeline import (
    prepare_batch_create_seeds,
    prepare_batch_create_fictsheets,
    prepare_batch_write_fiction,
)
from steps.step_4_create_qa_pipeline import (
    prepare_batch_fict_qa_generation,
    load_filtered_fict_qa,
    gather_questions_per_event,
)
from steps.steps_5_6_blind_qa_pipeline import blind_qa_pipeline
from steps.steps_5_6_original_fiction_qa_pipeline import original_fiction_qa_pipeline


""" Functions """

if __name__=="__main__":
    # STEP 1 : SEEDS
    if not os.path.exists(seeds_file):
        print("Sending STEP 1 prompts to OpenAI. Pipeline will exit after this.")
        prepare_batch_create_seeds(create_seeds_prompt_fname)
        batch_prompt(create_seeds_prompt_fname)
        exit()
    else:
        print("Loading STEP 1 from file.")
        seeds = list(load_gpt_responses_from_file(seeds_file))
        seeds = [s.split("*")[1].split("\nDEL")[0].replace("\n","") for s in seeds]


    # STEP 2 : FICTSHEETS (master document unpacking the who/what/where/when/why/how of the seed)
    if not os.path.exists(fictsheets_file):
        print("Sending STEP 2 prompts to OpenAI. Pipeline will exit after this.")
        prepare_batch_create_fictsheets(seeds_to_fictsheets_prompt_fname, seeds=seeds)
        batch_prompt(seeds_to_fictsheets_prompt_fname)
        exit()
    else:
        print("Loading STEP 2 from file.")
        fictsheets = list(load_gpt_responses_from_file(fictsheets_file))


    # STEP 3 : FICTION (write fiction for each seed & fictsheet using various styles)
    if not os.path.exists(fictions_file):
        print("Sending STEP 3 prompts to OpenAI. Pipeline will exit after this.")
        prepare_batch_write_fiction(write_fictions_prompt_fname, seeds=seeds, fictsheets=fictsheets)
        batch_prompt(write_fictions_prompt_fname)
        exit()
    else:
        print("Loading STEP 3 from file.")
        fictions = list(load_gpt_responses_from_file(fictions_file, return_ids=True))


    # STEP 4 : FICTITIOUS TRIVIA (write fictitious QA tuples in YAML, tracking which ficts & spans inspired the questions)
    if not os.path.exists(annotated_questions_file):
        if not os.path.exists(fict_qa_file):
            print("Sending STEP 4 prompts to OpenAI. Pipeline will exit after this.")
            prepare_batch_fict_qa_generation(fict_qa_generation_prompt_fname, seeds=seeds, fictsheets=fictsheets, fictions=fictions)
            batch_prompt(fict_qa_generation_prompt_fname)
            exit()
        else:
            print("Loading STEP 4 - raw GPT response from file.")
            fict_qa = list(load_gpt_responses_from_file(fict_qa_file, parse_into_yaml=True, return_ids=True))
            # Step 4's call to OpenAI generates several questions per fiction. Gather these into one list of questions per event.
            fict_qa = gather_questions_per_event(fict_qa)
    else:
        print("Loading STEP 4 - filtered questions from file.")
        fict_qa = load_filtered_fict_qa(annotated_questions_file)
        
    # STEPS 5-6: ATTEMPT ANSWER & GRADE ANSWER
    fictions_lookup = dict(fictions) 
    blind_qa_pipeline(fictions_lookup, fict_qa)    
    original_fiction_qa_pipeline(fictions_lookup, fict_qa)
