# FictionalQA

- **Paper:** [FictionalQA: A Dataset for Studying Memorization and Knowledge Acquisition](https://arxiv.org/abs/2506.05639)
- **Datasets Collection:** [tomg-group-umd/FictionalQA](https://hf.co/collections/tomg-group-umd/fictionalqa-68462c09c97e860922609e9e)

## Updates:
- (3/2/26) ArXiv copy updated, improved MCQs pushed to HF Hub.
- (1/26/26) Paper accepted for publication at ICLR 2026! 🎉
- (6/18/25) Initial dataset generation code release.
- (6/5/25) Paper posted to ArXiv.

## About

The FictionalQA dataset is a dataset specifically created to empower researchers to study the dual processes of fact memorization and verbatim sequence memorization. The dataset consists of synthetically-generated, webtext-like documents about fictional events and various facts they entail, as well as question-answer pairs about the facts within the fictional documents. 

# Repo Organization

Documentation
- `ds_readmes`: detailed documentation
    - `raw_dataset.md` describes FictionalQA dataset in detail
    - `reformatted_triviaqa.md` describes the TriviaQA reformatting used in the paper
    - `training_splits.md` describes the derivative FictionalQA dataset used in the paper's experiments

Synthetic Data Pipeline
- `pipeline`: code for generating a new fictional dataset
- `pipeline_sample_files`: sample output files from different pipeline steps

Supporting Artifacts for Paper Experiments
- `etl_nbs_and_scripts`
- `lm_eval`
- `paper_figures`
- [jwkirchenbauer/lit-gpt-dev-fiction](https://github.com/jwkirchenbauer/lit-gpt-dev-fiction): repository for finetuning the models for the paper; provided for transparency but not really intended to be re-used.


# Pipeline Quickstart
The code in the `pipeline` folder can be used to create a new FictionalQA dataset.

1. Install Python libraries from `requirements.txt`

    ```
    pip install -r requirements.txt
    ```

2. Export your OpenAI API Key

    ```
    export OPENAI_API_KEY=<your_api_key>
    ```

3. Configure `pipeline/constants.py`

    The file `pipeline/constants.py` contains the constants and configurations needed for the synthetic data generation pipeline. 

    The following constants should be verified and/or changed prior to the start of each run:
    - `TEST_MODE`: set to `True` to generate the intermediate outputs but skip the OpenAI API calls, which is useful for testing any changes to the pipeline steps. Set to `False` to execute the actual API calls. 
    - `model`: the desired OpenAI chat model. In the paper, this is set to `gpt-4o-2024-08-06`.
    - `n_seed_prompts`: the number of generated fictional events. In the paper, this is set to 100. 
    - `num_documents_per_style`: the number of generated fictional documents per style per fictional event. 
    - `root_prompt_dir`: the directory for storing prompt files that will be dispatched to the batch OpenAI API calls.
    - `root_output_dir`: the directory in which the batch results are downloaded and stored. These result files can be downloaded from the OpenAI platform directly after each batch run finishes. 
    - `generated_output_dir`: the directory in which intermediate output files in the pipeline are stored. This includes annotations and gradings for the Q&As. 

4. Run the pipeline


    To start or continue the pipeline:

    ```
    python pipeline/main.py
    ```

    This looks for the previous batch inputs and outputs as defined in `pipeline/constants.py` and then execute the next possible step. For detailed definitions of the steps and their shapes, view the `ds_readmes/raw_dataset.md` documentation.

    Each step in the data synthesis pipeline utilizes the OpenAI batch API, which is run asynchronously. When not in test mode (see above), the pipeline exits after each batch API call. The results of the batch runs will be available for download via the OpenAI Platform at Dashboard > Manage > Batches. Place the downloaded file(s) in the directory defined in the `root_output_dir` configuration, and rename the corresponding constants in `pipeline/constants.py` to match the downloaded file name. 

    The pipeline can then be continued onto the next step using the same `pipeline/main.py`. It will exit when all the steps have been completed. 

## Contributing 👷🛠️

This repository contains a refactored version the dataset construction code used to produce the dataset for the paper.

Please point out any problems you encounter using either git issues or PR's!
