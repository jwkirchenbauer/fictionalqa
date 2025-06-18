# sketch file for drafting a script to make copies of huggingface hub datset repositories

from huggingface_hub import HfApi
from huggingface_hub import create_repo, delete_repo, update_repo_settings
from datasets import load_dataset, get_dataset_config_names, DatasetDict
import os
import time

# DELETE_DST = False
# CREATE_DST = False

# DELETE_DST = True
# CREATE_DST = False

DELETE_DST = False
CREATE_DST = True

# DELETE_DST = True
# CREATE_DST = True

# README_ONLY = False
README_ONLY = True  # if True, only copy the README.md file and not the dataset

BACKOFF_DELAY = 60  # seconds to wait before retrying an operation
MAX_RETRIES = 10  # max number of retries for a single operation

api = HfApi(token=os.getenv("HF_HUB_TOKEN"))

src_repos = [
    "tomg-group-umd/fictional_qa_03-19-25_processed_flat",
    "tomg-group-umd/fictional_qa_03-19-25_training_splits",
    "tomg-group-umd/trivia_qa_03-19-25_training_splits",
]

copy_suffix = "_copy"

# repo_name_map = None

repo_name_map = {
    "tomg-group-umd/fictional_qa_03-19-25_processed_flat": "tomg-group-umd/fictionalqa",
    "tomg-group-umd/fictional_qa_03-19-25_training_splits": "tomg-group-umd/fictionalqa_training_splits",
    "tomg-group-umd/trivia_qa_03-19-25_training_splits": "tomg-group-umd/fictionalqa_reformatted_triviaqa",
}

readme_map = {
    "tomg-group-umd/fictional_qa_03-19-25_processed_flat": "ds_readmes/raw_dataset.md",
    "tomg-group-umd/fictional_qa_03-19-25_training_splits": "ds_readmes/training_splits.md",
    "tomg-group-umd/trivia_qa_03-19-25_training_splits": "ds_readmes/reformatted_triviaqa.md",
}

col_drop_map = {
    "tomg-group-umd/fictional_qa_03-19-25_processed_flat": {
        "_all": ["batch_metadata"],
        "seeds": [],
        "fictsheets": [],
        "fictions": [],
        "fict_qa": [],
        "joined_qa": [
            "batch_metadata_fictsheet",
            "batch_metadata_fiction",
            "batch_metadata_qa",
            "batch_metadata_blind",
            "batch_metadata_informed",
        ],
        "blind_answer_attempts": [],
        "informed_answer_attempts": [],
    },
    "tomg-group-umd/fictional_qa_03-19-25_training_splits": {},
    "tomg-group-umd/trivia_qa_03-19-25_training_splits": {},
}


def copy_repo(src_name, dst_name, delete_dst=False, create_dst=True):

    if delete_dst:
        # delete the dst repo if it exists
        try:
            delete_repo(repo_id=dst_name, repo_type="dataset")
            print(f"Deleted existing repo: {dst_name}")
        except Exception as e:
            print(f"Could not delete repo {dst_name}: {e}")

    # load all configs as a DatasetDict
    ds_dict = DatasetDict()
    config_names = get_dataset_config_names(src_name)
    for config_name in config_names:
        ds = load_dataset(src_name, config_name)
        ds = ds.get("train", ds.get("validation"))
        ds_dict[config_name] = ds

    # drop cols if the col_drop_map has entries for this src_name
    # they are subkeyed by the config within each dataset
    # the trick here is we want to operate on the dataset in dst, but it is in
    # the "load_dataset" format not the "load_from_disk" format, and we need to keep
    # it that way to be able to still push using git ops on the dst repo
    if src_name in col_drop_map:
        col_drop_dict = col_drop_map[src_name]
        for config_name, ds in ds_dict.items():
            cols_to_drop = col_drop_dict.get(config_name, [])
            cols_to_drop += col_drop_dict.get("_all", [])
            if len(cols_to_drop) > 0:
                dropped_cols = []
                for col in cols_to_drop:
                    try:
                        ds = ds.remove_columns(col)
                        dropped_cols.append(col)
                    except ValueError as e:
                        print(e)
                        continue
                ds_dict[config_name] = ds  # update the ds_dict with the modified ds
                print(f"Dropped columns {dropped_cols} from config {config_name}")
            else:
                print(f"Expected list of columns to drop, got {cols_to_drop}")

    # show the dataset before proceeding
    print(ds_dict)

    # create, commit, and push the changes to the dst repo
    if not create_dst:
        print(f"Not creating repo {dst_name} or pushing changes, exiting.")
    else:
        if not README_ONLY:
            # create a new repo with the dst name
            create_repo(
                repo_id=dst_name, repo_type="dataset", exist_ok=True, private=True
            )
            update_repo_settings(
                repo_id=dst_name,
                repo_type="dataset",
                gated="manual",
                private=False,
            )

            # lets instead push using the huggingface_hub API
            # we have ds dicts that have configs in them and want to push each separately so iter
            # and push each config by slecting it and using the push_to_hub method
            for config_name, ds in ds_dict.items():
                success = False
                tries = 0
                last_error = None
                while not success and tries < MAX_RETRIES:
                    try:
                        ds.push_to_hub(
                            repo_id=dst_name,
                            config_name=config_name,
                            token=os.getenv("HF_HUB_TOKEN"),
                        )
                        print(f"Pushed config {config_name} to {dst_name}")
                        success = True
                    except Exception as e:
                        last_error = e
                        tries += 1
                        if tries < MAX_RETRIES:
                            print(
                                # f"Failed attempt {tries} to push config {config_name} to {dst_name}: {e}"
                                f"Failed attempt {tries}/{MAX_RETRIES} to push config {config_name}... retrying in {BACKOFF_DELAY} seconds."
                            )
                            time.sleep(BACKOFF_DELAY)
                if not success:
                    print(
                        f"After {tries} attempts, could not push config {config_name} to {dst_name}: {last_error}"
                    )

        # we also want to push the readme file if it exists
        # add in the readme file if it exists in the readme_map
        if src_name in readme_map:
            readme_file = readme_map[src_name]
            try:
                api.upload_file(
                    path_or_fileobj=readme_file,
                    path_in_repo="README.md",
                    repo_id=dst_name,
                    repo_type="dataset",
                    token=os.getenv("HF_HUB_TOKEN"),
                )
                print(f"Pushed README.md to {dst_name}")
            except Exception as e:
                print(f"Could not push README.md to {dst_name}: {e}")

    if not create_dst:
        print(f"Mocked operations of cloning w/ mods {src_name} to {dst_name}")
    else:
        print(f"Cloned w/ mods {src_name} to {dst_name}")


for src_repo in src_repos:
    if repo_name_map is None:
        dst_repo = src_repo + copy_suffix
        copy_repo(src_repo, dst_repo, delete_dst=DELETE_DST, create_dst=CREATE_DST)
    else:
        # use the repo_name_map to get the dst repo name, fallback to src_repo + copy_suffix if not found
        dst_repo = repo_name_map.get(src_repo, src_repo + copy_suffix)
        copy_repo(src_repo, dst_repo, delete_dst=DELETE_DST, create_dst=CREATE_DST)
