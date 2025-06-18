import json
import os
import yaml
import re
import string

from typing import Generator, Union, Optional
try:
    from openai import OpenAI
    from wonderwords import RandomWord
    import tiktoken
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    # Download stopwords if not already downloaded
    nltk.download("stopwords")
    nltk.download("punkt")
except ImportError:
    pass

# Custom imports
from constants import *
from prompts import *
from custom_types import *
from idutils import *

try:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
    random_word_generator = RandomWord()
except:
    # add a warning message here if needed.
    pass

def parse_yaml(str_with_yaml_content) -> list[dict[str, str]]:
    """
    Gets the YAML data from an LLM response with YAML

    str_with_yaml_content:
        an LLM response containing a YAML-formatted string
    result:
        list of str-typed key/val dicts from the YAML
    """
    if "```" in str_with_yaml_content:
        str_with_yaml_content = str_with_yaml_content.split("```yaml")[1].split("```")[
            0
        ]
    try:

        # SLOPPY EDGE CASE HANDLING
        str_with_yaml_content = str_with_yaml_content.replace(
            "natural\n_answer", "natural_answer"
        )
        str_with_yaml_content = re.sub(
            r'("[^"]*") and ("[^"]*")',
            lambda match: match.group(1) if match.group(1) else match.group(0),
            str_with_yaml_content,
        )
        data = yaml.safe_load(str_with_yaml_content)
    except yaml.YAMLError as exc:
        print(str_with_yaml_content)
        print(f"Error parsing YAML: {exc}")
        return []
    result = []
    for item in data:
        # SLOPPY edge case handling
        if "span" in item:
            item["span_answer"] = item["span"]
        if "span,answer" in item:
            item["span_answer"] = item["span,answer"]
        if "natural" in item:
            item["natural_answer"] = item["natural"]
        if "natural,answer" in item:
            item["natural_answer"] = item["natural,answer"]
        if "natural answer" in item:
            item["natural_answer"] = item["natural answer"]
        if "natural, answer" in item:
            item["natural_answer"] = item["natural, answer"]
        # print(item)
        assert "question" in item
        assert "fict" in item
        # assert 'span_answer' in item
        assert "natural_answer" in item

        item_dict = {}
        for key in item:
            if isinstance(item[key], list):
                for subitem in item[key]:
                    item_dict[key] = subitem
            else:
                item_dict[key] = item[key]
        result.append(item_dict)
    return result


def batch_prompt(batch_file: str, description: str = "batch prompt") -> None:
    """Sends a batch request to the OpenAI API using batch_file"""
    if TEST_MODE:
        print("Pretending to call OpenAI's API.")
        return
    else:
        print("...")
        print("Actually calling OpenAI API:", batch_file)
        print("...")

    input_file = client.files.create(file=open(batch_file, "rb"), purpose="batch")

    batch_creation = client.batches.create(
        input_file_id=input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": description},
    )


def make_batch_prompt_file(
    filename: str,
    system_messages: list[str],
    user_messages: list[str],
    model: str = "gpt-4-turbo",
    max_tokens: int = 4096,
    temperature: float = 1.0,
    custom_ids: Optional[list] = None,
) -> None:
    """
    Writes a batch of prompts to batchinput.jsonl
    """
    jsons = []
    num_tokens = 0
    for i, (sm, um) in enumerate(zip(system_messages, user_messages)):
        uid = (
            custom_ids[i]
            if custom_ids
            else filename.replace(".jsonl", "").replace("batch_prompt_files/", "")
            + f"-{i}"
        )
        d = {
            "custom_id": uid,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": sm},
                    {"role": "user", "content": um},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 1,
                "seed": random_seed,
            },
        }
        jsons.append(d)
        num_tokens += max_tokens + len(tokenizer.encode(sm)) + len(tokenizer.encode(um))
    with open(filename, "w") as f:
        for d in jsons:
            json.dump(d, f)
            f.write("\n")
    print(f"Estimated number of tokens to be processed: {num_tokens:,}")


def load_gpt_responses_from_file(
    filename,
    parse_into_yaml=False,
    return_ids=False,
) -> Generator[
    Union[
        list[str],  # both False
        list[tuple[str, str]],  # return_ids=True
        list[YamlResults],  # parse_into_yaml=True
        list[tuple[str, YamlResults]],  # both True
    ],
    None,
    None,  # no send or return
]:
    with open(filename, "r") as file:
        for line in file:
            linedata = json.loads(line)
            linestr = linedata["response"]["body"]["choices"][0]["message"]["content"]

            res = parse_yaml(linestr) if parse_into_yaml else linestr

            if return_ids:
                yield linedata["custom_id"], res
            else:
                yield res


def extract_nonstopwords(text):
    """Extracts all non-stopwords from a given text."""
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)
    nonstopwords = [
        word
        for word in words
        if not word.isdigit()
        and word.lower().strip() not in stop_words
        and word.lower().strip() not in string.punctuation
    ]
    return nonstopwords


def estimate_price(
    system_messages: list[str],
    user_messages: list[str],
    max_tokens: int, 
) -> int:
    """
    Print and return the estimated price based on 
    the prompt and number of prompts in a batch
    """
    num_input_tokens = 0
    num_output_tokens = 0
    for (sm, um) in zip(system_messages, user_messages):
        num_input_tokens += (len(tokenizer.encode(sm)) + len(tokenizer.encode(um)))
        num_output_tokens += max_tokens

    price_input = INPUT_PRICE_PER_1M * 1e-6 * num_input_tokens
    price_output = OUTPUT_PRICE_PER_1M * 1e-6 * num_output_tokens
    total_price = price_input + price_output
    print(f"Length of input tokens: {num_input_tokens:,}")
    print(f"Max length of output tokens: {num_output_tokens:,}")
    print(f"Approximate total price: {total_price:,}")
    
    return total_price