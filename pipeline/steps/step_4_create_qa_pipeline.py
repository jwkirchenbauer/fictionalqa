import json
import collections
from constants import *
from prompts import *
from utils import (
    parse_fiction_string_id,
    make_batch_prompt_file
)

from custom_types import Question, YamlResults
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

q_per_fict_model = SentenceTransformer("all-MiniLM-L6-v2")

def load_filtered_fict_qa(filename: str) -> dict[str, dict[str, Question]]:
    out_fict_qa: dict[str, dict[str, Question]] = collections.defaultdict(dict)

    with open(filename, "r") as f:
        for line in f:
            linedata = json.loads(line)
            fictsheet_id = linedata["id"]
            fictsheet_questions = linedata["questions"]
            out_fict_qa[fictsheet_id] = fictsheet_questions

    return out_fict_qa

def gather_questions_per_event(
    fict_qa: list[tuple[str, YamlResults]],
) -> dict[str, dict[str, Question]]:
    """
    Takes output from API and parses it into individual questions.
    The goal is for this dictionary to map from fict ID to all the questions associated with the fict.

    {
        fictsheet_id: [Question, Question, ...],
        ...
    }
    """
    # output fict_qa
    out_fict_qa = collections.defaultdict(dict)

    # Deduplication STEP 1: exact matches
    # naively dedupe w/ sets
    # ideally, should extend to incorporate some similarity metric(s)

    # exact question string to root qid lookup for exact matches
    questions_from_fict = collections.defaultdict(dict)
    # tracking list of questions for this event

    for uid, response in fict_qa:
        fictsheet_id, style_id, fiction_id = parse_fiction_string_id(uid)

        for i, qa in enumerate(response):
            if all(k in qa for k in fict_qa_required_fields):
                this_qid = f"{uid}_question_{i:03d}"
                question_object: Question = qa | {
                    "duplicate_root": this_qid,  # not duplicate by default
                    "duplicate_relationship": None,
                    "question_id": this_qid,
                }
                # add the question uid in the format of
                # event_00_style_news_num_00_question_0
                question_str = qa["question"].lower()

                # naively only pick the first instance of the question
                # without regard to asnwers or fict snippets
                if question_str not in questions_from_fict[fictsheet_id]:
                    questions_from_fict[fictsheet_id][question_str] = this_qid
                    # qas_per_fict[fictsheet_id].append(qa)
                else:
                    # exact match found, update root
                    root_qid = questions_from_fict[fictsheet_id][question_str]
                    question_object["duplicate_root"] = root_qid
                    question_object["duplicate_relationship"] = "exact"

                out_fict_qa[fictsheet_id][this_qid] = question_object

    # Deduplication STEP 2: dedupe based on similarity
    for fictsheet_id, fictsheet_questions in out_fict_qa.items():
        str_questions = [
            q["question"]
            for q in fictsheet_questions.values()
            if q["duplicate_relationship"] == None
        ]
        embeddings = q_per_fict_model.encode(str_questions)
        similarities = q_per_fict_model.similarity(embeddings, embeddings)

        # cluster sentences
        clustering_model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.4,
        )
        clustering_model.fit(similarities)
        num_clusters = clustering_model.n_clusters_
        cluster_assignment = clustering_model.labels_

        all_sentences = [[] for i in range(num_clusters)]
        for i, cluster_id in enumerate(cluster_assignment):
            all_sentences[cluster_id].append(str_questions[i])

        # update the output mapping
        for cluster in range(num_clusters):
            cluster_sentences = all_sentences[cluster]
            longest_sentence = max(cluster_sentences, key=lambda s: len(s))
            longest_qid = questions_from_fict[fictsheet_id][longest_sentence.lower()]

            for sentence in cluster_sentences:
                if sentence != longest_sentence:
                    # update root qid
                    this_qid = questions_from_fict[fictsheet_id][sentence.lower()]
                    out_fict_qa[fictsheet_id][this_qid]["duplicate_root"] = longest_qid
                    out_fict_qa[fictsheet_id][this_qid][
                        "duplicate_relationship"
                    ] = "similar"

    # jsonlines dump
    with open(annotated_questions_file, "w") as f:
        for fictsheet_id, fictsheet_questions in out_fict_qa.items():
            linedata = {"id": fictsheet_id, "questions": fictsheet_questions}
            json.dump(linedata, f)
            f.write("\n")

    return out_fict_qa

# STEP 4 : FICTITIOUS TRIVIA (write fictitious QA tuples in YAML, tracking which ficts & spans inspired the questions)
def prepare_batch_fict_qa_generation(
    filename: str,
    seeds: list[str],
    fictsheets: list[str],
    fictions: list[str],
) -> None:

    assert len(seeds) == len(fictsheets) and len(seeds) * total_docs_per_style == len(
        fictions
    )

    # There is a one-to-one mapping between seeds and fictsheets.
    seeds_fictsheets = list(zip(seeds, fictsheets))

    # There are multiple fictions for each of the above. Let's iterate over the fictions.
    # For each fiction, we will prompt the model to generate several questions about the
    # fiction (and its corresponding seed and fictsheet).
    user_messages = []
    custom_ids = []
    for fiction_id_string, fiction in fictions:
        fictsheet_id, style_id, fiction_id = parse_fiction_string_id(fiction_id_string)

        seed, fictsheet = seeds_fictsheets[int(fictsheet_id)]

        user_message = f"{seed}\n{fictsheet}\n{fiction}"
        user_messages.append(
            fict_qa_generation_user.format(
                user_message=f"{seed}\n{fictsheet}\n{fiction}"
            )
        )
        custom_ids.append(fiction_id_string)

    make_batch_prompt_file(
        filename,
        [fict_qa_generation_sys] * len(user_messages),
        user_messages,
        model,
        fict_qa_generation_tokens,
        fict_qa_generation_temp,
        custom_ids=custom_ids * 2,
    )