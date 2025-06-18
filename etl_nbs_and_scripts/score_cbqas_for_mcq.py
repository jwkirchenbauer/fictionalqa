import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
import random

from functools import partial

from huggingface_hub import HfApi
from datasets import (
    load_dataset,
    load_from_disk,
    DatasetDict,
    Dataset,
    get_dataset_config_names,
)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch.distributed as dist


# python /p/lustre5/$USER/llnl-tools/launch_tuo.py \
#     --output_dir="/p/lustre5/kirchenb/fictional_qa/output" \
#     --qos=pdebug \
#     --run_name=test_parallel_job \
#     --nodes=1 \
#     --minutes=10 \
#     --custom_invocation='python -u /p/lustre5/kirchenb/fictional_qa/score_cbqas_for_mcq.py'


# ROW_LIMIT = 8
# ROW_LIMIT = 16
# ROW_LIMIT = 12
# ROW_LIMIT = 128
ROW_LIMIT = None

# ALT_TGT_LIM = 10
# ALT_TGT_LIM = 20
# ALT_TGT_LIM = 100
# ALT_TGT_LIM = 200
ALT_TGT_LIM = None

LOCAL_RANK = int(os.getenv("LOCAL_RANK"))
RANK = int(os.getenv("RANK"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE"))

RANKS_PER_NODE = 4

dist.init_process_group(world_size=WORLD_SIZE, rank=RANK)

api = HfApi(token=os.environ["HUGGING_FACE_HUB_TOKEN"])

REPO_ID = "tomg-group-umd/fictional_qa_03-19-25_training_splits"

# raw_cbqa_config = "fict_qa_cbqa_ds"
# raw_cbqa_config = "fict_qa_cbqa_blind_inf_ex_dedup_ds"
# raw_cbqa_config = "fict_qa_obqa_ds"
raw_cbqa_config = "fict_qa_obqa_blind_inf_ex_dedup_ds"
# raw_cbqa_config = "fict_qa_obqa_blind_inf_fuzzy_deduped_ds"

ds = load_dataset(REPO_ID, name=raw_cbqa_config)["train"]
ds

# MODEL_NAME_OR_PATH = "/p/vast1/pretrain/models/Llama-3-2-1B"
# MODEL_NAME_OR_PATH = "/p/vast1/pretrain/models/Llama-3-2-1B-Instruct"
MODEL_NAME_OR_PATH = "/p/vast1/pretrain/models/Llama-3-2-3B-Instruct"
# MODEL_NAME_OR_PATH = "/p/vast1/pretrain/models/Meta-Llama-3-1-8B-Instruct"
# MODEL_NAME_OR_PATH = "/p/lustre5/kirchenb/llm-pretraining-root/lit-gpt-dev-fiction/output/exp1_train_val_splits_5pct_4N_mb8-wb128_llama-3-2-1B_event-split-fictions-train-val/hf_checkpoint_exp1_train_val_splits_5pct_4N_mb8-wb128_llama-3-2-1B_event-split-fictions-train-val"


# BASE_SAVE_PATH = f"/p/lustre5/kirchenb/fictional_qa/output"
BASE_SAVE_PATH = f"/p/lustre5/kirchenb/fictional_qa/output/test_parallel_job"
# SAVE_PATH = f"{BASE_SAVE_PATH}/{raw_cbqa_config}_{MODEL_NAME_OR_PATH.split('/')[-1]}_scored_lim{ROW_LIMIT}_alts{ALT_TGT_LIM}"

rank_suffix_fn = lambda l, r, w: f"cuda{l}_{r}of{w}"

SAVE_PATH = f"{BASE_SAVE_PATH}/{raw_cbqa_config}_{MODEL_NAME_OR_PATH.split('/')[-1]}_scored_rowlim{ROW_LIMIT}_altlim{ALT_TGT_LIM}"


def load_model_and_tok(rank):

    if rank is None:
        rank = LOCAL_RANK

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME_OR_PATH, model_max_length=2048, padding_side="left"
    )

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    local_model_filepath = f"{MODEL_NAME_OR_PATH}/pytorch_model.bin"
    if os.path.exists(local_model_filepath):  # a local converted model
        print(f"Loading local model at {local_model_filepath}")
        state_dict = torch.load(local_model_filepath, weights_only=False)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME_OR_PATH,
            state_dict=state_dict,
            torch_dtype=torch.bfloat16,
        ).to(
            device
        )  # to behave with multiproc, be lazy
        # ).cpu()
    else:
        print(f"Loading hub model at {MODEL_NAME_OR_PATH}")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME_OR_PATH).to(
            device
        )  # to behave with multiproc, be lazy
        # model = AutoModelForCausalLM.from_pretrained(MODEL_NAME_OR_PATH).cpu()

    return model, tokenizer


models = {}
tokenizers = {}
loaded_model = {None: False}  # for no rank case
for r in range(1):
    # for r in range(4):
    loaded_model[r] = False


def tokenize_input_text(
    input_text=None,
    tokenizer=None,
):

    input_ids = tokenizer(input_text, return_tensors="pt", add_special_tokens=True)[
        "input_ids"
    ]
    # if last token is eos, remove it
    if input_ids[0][-1] == tokenizer.eos_token_id:
        input_ids[0] = input_ids[0][:-1]

    return input_ids


@torch.no_grad()
def compute_loss_acc(
    input_ids=None,  # can be single or batch
    labels=None,
    model=None,
    mask_first_k_toks=None,
    mask_value=-100,
    print_tensors=True,
):
    if labels is None:
        labels = input_ids.clone()

    bsz, seq_len = input_ids.shape

    input_ids = input_ids[:, 0 : (seq_len - 1)].contiguous().long().to(model.device)
    if mask_first_k_toks is not None:
        labels[:, 0:mask_first_k_toks] = mask_value

    labels = labels[:, 1:(seq_len)].contiguous().long().to(model.device)

    if print_tensors:
        print(input_ids)
        print(labels)

        print("Calling model ...")

    # output = model(input_ids, labels=labels, reduction="none")
    logits = model(input_ids, labels=labels, reduction="none").logits

    if print_tensors:
        print("Model call complete!")

    # Compute accuracy
    acc_per_row = []
    for logit_row, label_row in zip(logits, labels):

        predictions = torch.argmax(logit_row, dim=-1)

        valid_positions = label_row != (torch.ones_like(label_row) * mask_value)

        invalid_positions = label_row == (torch.ones_like(label_row) * mask_value)

        hits = (predictions[valid_positions] == label_row[valid_positions]).to(
            torch.float
        )

        acc_per_row.append(hits.sum() / valid_positions.sum())

    token_acc = torch.tensor(acc_per_row)

    # Compute seq wise loss
    logits = logits.reshape(-1, logits.size(-1))
    labels = labels.reshape(-1)

    losses = torch.nn.functional.cross_entropy(
        logits, labels, ignore_index=mask_value, reduction="none"
    )
    losses = losses.reshape(bsz, -1)
    valid_mask = (labels != mask_value).float()
    valid_mask = valid_mask.reshape(bsz, -1)
    losses = (losses * valid_mask).sum(dim=1) / valid_mask.sum(dim=1)

    return losses, token_acc, predictions, valid_positions, invalid_positions


@torch.no_grad()
def run_generation(
    input_ids=None,
    model=None,
    max_new_tokens: int = 100,
    min_new_tokens: int = 0,
    top_k: int = 50,
    top_p: float = 1.0,
    temperature: float = 0.0,
):

    output = model.generate(
        input_ids=input_ids.to(model.device),
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        num_return_sequences=1,
        # do_sample=True if temperature > 0.0 else False,
        do_sample=False,
        # temperature=temperature,
        # top_k=top_k,
        # top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    return output


def decode_output(
    output_ids=None,
    tokenizer=None,
    skip_special_tokens=False,
):
    if type(output_ids) == list or (
        type(output_ids) == torch.Tensor and len(output_ids.shape) > 1
    ):
        output_ids = output_ids[0]
    response_text = tokenizer.decode(
        output_ids, skip_special_tokens=skip_special_tokens
    )
    return response_text


def condition_input_supervise_target(
    input_text, target_text, tokenizer, add_bos, add_eos, mask_value
):
    input_string = input_text
    joint_string = input_text + target_text
    input_tokens = tokenizer.encode(input_string)
    if add_bos:
        input_tokens = [tokenizer.bos_token_id] + input_tokens

    joint_tokens = tokenizer.encode(joint_string)
    if add_bos:
        joint_tokens = [tokenizer.bos_token_id] + joint_tokens
    if add_eos:
        joint_tokens = joint_tokens + [tokenizer.eos_token_id]
    input_tokens = torch.tensor(input_tokens, dtype=torch.long)
    joint_tokens = torch.tensor(joint_tokens, dtype=torch.long)

    label_tokens = joint_tokens.clone()
    # mask the locations of the input tokens in the joint tokens
    if add_bos:
        label_tokens[1 : len(input_tokens) - 1] = mask_value
    else:
        label_tokens[0 : len(input_tokens)] = mask_value

    input_tokens = joint_tokens
    return input_tokens, label_tokens


def score_input_target(
    input_text=None,
    target_text=None,
    tokenizer=None,
    model=None,
    print_tensors=False,
    skip_special_tokens=True,
    print_generations=False,
    block_size=2048,
):

    if type(input_text) == str:
        tokd_inputs, tokd_labels = condition_input_supervise_target(
            input_text=input_text,
            target_text=target_text,
            tokenizer=tokenizer,
            add_bos=True,
            add_eos=False,
            mask_value=-100,
        )
        tokd_inputs = tokd_inputs.unsqueeze(0)
        tokd_labels = tokd_labels.unsqueeze(0)
    elif type(input_text) == list:
        bsz = len(input_text)
        all_inputs, all_labels = [], []
        for i in range(len(input_text)):
            tokd_inputs, tokd_labels = condition_input_supervise_target(
                input_text=input_text[i],
                target_text=target_text[i],
                tokenizer=tokenizer,
                add_bos=True,
                add_eos=False,
                mask_value=-100,
            )
            all_inputs.append(tokd_inputs)
            all_labels.append(tokd_labels)
        # tokd_inputs = torch.stack(all_inputs)
        # tokd_labels = torch.stack(all_labels)
        # print(all_inputs)
        # all_lengths = [len(row) for row in (all_inputs+all_labels)]
        all_lengths = [len(row) for row in (all_inputs)]
        # min against block size since the max realized could be longer than block size.
        local_block_size = min(max(all_lengths), block_size)
        tokd_inputs = torch.full(
            (bsz, local_block_size), tokenizer.eos_token_id, dtype=torch.int
        )
        tokd_labels = torch.full((bsz, local_block_size), -100, dtype=torch.int)
        for i, (input_tokens, label_tokens) in enumerate(zip(all_inputs, all_labels)):
            tokd_inputs[i, : len(input_tokens)] = input_tokens[
                :local_block_size
            ]  # this ensures we don't write past the block size
            tokd_labels[i, : len(label_tokens)] = label_tokens[:local_block_size]
        if print_tensors:
            print(tokd_inputs.shape)
            print(tokd_labels.shape)
            print(tokd_inputs)
            print(tokd_labels)
    else:
        raise ValueError(f"input_text must be str or list, got {type(input_text)}")

    if print_tensors:
        print(tokd_inputs)
        print(tokd_labels)
        print(f"Input size: {tokd_inputs.shape}")

    loss, acc, preds, valid_mask, invalid_mask = compute_loss_acc(
        input_ids=tokd_inputs,
        labels=tokd_labels,
        model=model,
        mask_first_k_toks=None,
        mask_value=-100,
        print_tensors=print_tensors,
    )
    em = (acc == 1.0).to(torch.float)

    if print_generations:
        assert (
            type(input_text) == str
        ), "print_generations only works for single input_text"
        print(f"Loss: {loss.item():.4f}")
        print(f"Acc: {acc.item():.1%}")

        joint_text = input_text + target_text

        decoded_input = decode_output(
            output_ids=tokd_inputs,
            tokenizer=tokenizer,
            skip_special_tokens=skip_special_tokens,
        )
        assert (
            decoded_input == joint_text
        ), f"Decoded input does not match original input. Decoded: {decoded_input}, Original: {input_text}"

        print(decoded_input)
        tokd_labels_copy = tokd_labels.clone()
        tokd_labels_copy[tokd_labels_copy == -100] = tokenizer.bos_token_id
        decoded_target = decode_output(
            output_ids=tokd_labels_copy,
            tokenizer=tokenizer,
            skip_special_tokens=skip_special_tokens,
        )

        print(decoded_target)

        if not skip_special_tokens:
            print(
                "(note we must mark invalid pos where we had -100 with a valid one like bos to decode it properly)"
            )

        preds[invalid_mask] = tokenizer.bos_token_id

        if print_tensors:
            print(preds)

        print("(teacher forced next token prediction)")
        print(
            decode_output(
                output_ids=preds,
                tokenizer=tokenizer,
                skip_special_tokens=skip_special_tokens,
            )
        )

        input_ids = tokd_inputs[:, : -((tokd_labels != -100).sum().item() - 1)]

        if print_tensors:
            print(f"Input tokens:")
            print(input_ids)

        # print(f"Input text:")
        # print(decode_output(output_ids=input_ids, tokenizer=tokenizer, skip_special_tokens=skip_special_tokens))

        output_ids = run_generation(input_ids=input_ids, model=model, min_new_tokens=1)
        output_ids = output_ids[:, input_ids.shape[1] :]
        output_text = decode_output(output_ids=output_ids, tokenizer=tokenizer)
        print(f"Greedy decoded output:")
        print(output_text)

    return loss, acc, em


all_targets = ds["target"]
print(len(all_targets))
all_unique_targets = {k: None for k in all_targets}
print(len(all_unique_targets.keys()))
all_targets = list(all_unique_targets.keys())


def add_q_copy_for_all_targets(row):

    row["alt_targets"] = all_targets.copy()

    return row


ds_with_alt_targets = ds.map(add_q_copy_for_all_targets, batched=False, num_proc=4)


def ask_if_alt_matches_gt(input_text, target_text, alt_target):

    # Is the Alternate Answer exactly equivalent to the True Answer? Give a Verdict as a Yes or No only.
    # Does the alternate answer sound plausible given the true answer? Does it also match in terms of parts of speech and grammatical form? Give a verdict as a Yes or No only.
    templated_input = f"""\
{input_text.replace("Answer: ","True Answer: ")}{target_text}

Alternate Answer: {alt_target}

Does the Alternate Answer roughly match the True Answer in terms of parts of speech and grammatical form? Give a verdict as a Yes or No only.

Verdict: \
"""
    templated_targets = ["Yes", "No"]
    return templated_input, templated_targets


def compute_score_all_targets(row, rank=None, bsz=1, scoring_prompt=None):

    # input_text = row["input_w_fiction"]
    # input_text = row["input_w_fictsheet"]
    input_text = row["input"]  # just uninformed guessing
    target_text = row["target"]

    if ALT_TGT_LIM:
        alt_targets = row["alt_targets"][:ALT_TGT_LIM]
    else:
        alt_targets = row["alt_targets"]

    input_text = [input_text] * len(alt_targets)
    target_text = [target_text] * len(alt_targets)

    losses = []
    accs = []
    ems = []

    print(f"Scoring {len(alt_targets)} alt targets for: {row['question_id']}")

    for i in range(0, len(alt_targets), bsz):
        # loss, acc, em = score_input_target(input_text=input_text[i:i+bsz], target_text=alt_targets[i:i+bsz], tokenizer=tokenizer, model=model, print_generations=False)

        assert bsz == 1
        templated_inputs, templated_targets = ask_if_alt_matches_gt(
            input_text[i], target_text[i], alt_targets[i]
        )

        templated_inputs = [templated_inputs] * len(templated_targets)

        if loaded_model[rank] == False:
            models[rank], tokenizers[rank] = load_model_and_tok(rank)
            loaded_model[rank] = True

        loss, acc, em = score_input_target(
            input_text=templated_inputs,
            target_text=templated_targets,
            tokenizer=tokenizers[rank],
            model=models[rank],
            print_generations=False,
        )

        # for j in range(bsz):
        #     losses.append(loss[j])
        #     accs.append(acc[j])
        #     ems.append(em[j])

        # instead we append the loss ratio yes/no to make a lower better quantity
        losses.append(torch.tensor(loss[0].detach().clone() / loss[1].detach().clone()))
        accs.append(torch.tensor(acc[0].detach().clone()))
        ems.append(torch.tensor(-1.0))

    targets_and_losses = list(zip(alt_targets, losses))
    targets_and_accs = list(zip(alt_targets, accs))
    targets_and_accs_losses = list(zip(alt_targets, accs, losses))

    targets_by_loss = [
        (tgt, t.item()) for tgt, t in sorted(targets_and_losses, key=lambda x: x[1])
    ]
    targets_by_acc = [
        (tgt, t.item())
        for tgt, t in sorted(targets_and_accs, key=lambda x: x[1], reverse=True)
    ]

    targets_by_acc_loss = [
        (tgt, t0.item(), t1.item())
        for tgt, t0, t1 in sorted(
            targets_and_accs_losses, key=lambda x: (x[1], 1 / x[2]), reverse=True
        )
    ]

    targets_by_loss_tgt, targets_by_loss_l = zip(*targets_by_loss)
    targets_by_acc_tgt, targets_by_acc_a = zip(*targets_by_acc)

    targets_by_acc_loss_tgt, targets_by_acc_loss_a, targets_by_acc_loss_l = zip(
        *targets_by_acc_loss
    )

    return {
        "losses": losses,
        "ems": ems,
        "accs": accs,
        "targets_by_loss_tgt": targets_by_loss_tgt,
        "targets_by_loss_l": targets_by_loss_l,
        "targets_by_acc_tgt": targets_by_acc_tgt,
        "targets_by_acc_a": targets_by_acc_a,
        "targets_by_acc_loss_tgt": targets_by_acc_loss_tgt,
        "targets_by_acc_loss_a": targets_by_acc_loss_a,
        "targets_by_acc_loss_l": targets_by_acc_loss_l,
    }


from pathlib import Path

shard_save_path = f"{SAVE_PATH}_{rank_suffix_fn(LOCAL_RANK, RANK, WORLD_SIZE)}"

lock_filename = Path(f"{shard_save_path}.lock")
lock_filename.touch()

ds_with_scores = (
    ds_with_alt_targets.select(
        range(ROW_LIMIT if ROW_LIMIT is not None else len(ds_with_alt_targets))
    )
    .shard(num_shards=WORLD_SIZE, index=RANK)
    .map(compute_score_all_targets, batched=False, num_proc=1, with_rank=True)
)

ds_with_scores.save_to_disk(shard_save_path)

lock_filename.unlink()

dist.barrier()

if RANK == 0:
    print("All ranks completed.")

    # now join the shards
    from datasets import concatenate_datasets

    all_shards = []
    w = WORLD_SIZE
    for r in range(w):
        l = r % RANKS_PER_NODE
        all_shards.append(load_from_disk(f"{SAVE_PATH}_{rank_suffix_fn(l, r, w)}"))

    full_ds = concatenate_datasets(all_shards)

    full_ds.save_to_disk(f"{SAVE_PATH}")
