import random
from constants import *
from prompts import *
from utils import (
    random_word_generator,
    make_batch_prompt_file,
    load_gpt_responses_from_file,
    extract_nonstopwords
)

# STEP 1: Seeds
def prepare_batch_create_seeds(filename: str) -> None:

    # prepare user messages with random words
    user_messages_for_seeds = []
    for i in range(n_seed_prompts):
        randwords = []
        for k in range(n_inspiration_words_per_seed_prompt):
            randwords.append(random_word_generator.word())
        randyear = random.randint(1900, 2060)
        user_messages_for_seeds.append(
            create_seed_user.format(inspiration=randwords, year=randyear)
        )

    make_batch_prompt_file(
        filename,
        [create_seed_sys] * n_seed_prompts,
        user_messages_for_seeds,
        model,
        seed_generation_tokens,
        seed_generation_temp,
    )


# STEP 2 : FICTSHEETS (master document unpacking the who/what/where/when/why/how of the seed)
def prepare_batch_create_fictsheets(filename: str, seeds: list[str] = None) -> None:

    # prepare user messages with seeds
    if seeds is None:
        seeds = load_gpt_responses_from_file(seeds_file)

    make_batch_prompt_file(
        filename,
        [grow_fictsheet_sys] * len(seeds),
        seeds,
        model,
        infodoc_generation_tokens,
        infodoc_generation_temp,
    )


# STEP 3 : FICTION (write fiction for each seed & fictsheet using various styles)
def prepare_batch_write_fiction(
    filename: str,
    seeds: list[str],
    fictsheets: list[str],
) -> None:
    assert len(seeds) == len(fictsheets)

    user_messages = [
        f"{seed}\n{fictsheet}\nStyle: {style}"
        for seed, fictsheet in zip(seeds, fictsheets)
        for style in styles
    ]

    uids = []
    user_messages = []
    for idx, (seed, fictsheet) in enumerate(zip(seeds, fictsheets)):
        for style_name, style_desc in styles.items():
            num_in_this_style = num_documents_per_style[style_name]
            for jdx in range(num_in_this_style):
                words = random.sample(extract_nonstopwords(fictsheet), 3)
                user_message = f"{seed}\n{fictsheet}\nStyle: {style_desc}\nHere are three important words to include in your writing. Make sure to use them: {', '.join(words)}"
                # label custom_id for fiction
                uid = f"event_{idx:03d}_style_{style_name}_num_{jdx:03d}"
                user_messages.append(user_message)
                uids.append(uid)
    make_batch_prompt_file(
        filename,
        [write_fiction_sys] * len(user_messages),
        user_messages,
        model,
        write_fiction_tokens,
        write_fiction_temp,
        custom_ids=uids,
    )
