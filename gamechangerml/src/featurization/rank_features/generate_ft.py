from os.path import join
from pandas import DataFrame, read_csv
from collections import Counter
from argparse import ArgumentParser
from glob import glob
import en_core_web_md
from gamechangerml.src.utilities import configure_logger, load_json
from gamechangerml.src.text_handling.process import preprocess
from gamechangerml.src.featurization.rank_features.search_data import generate_pop_docs
from gamechangerml.src.featurization.rank_features.rank import Rank
from gamechangerml import DATA_PATH


"""
Usage:
    example:
    python -m gamechangerml.src.featurization.rank_features.generate_ft -c test/small_corpus/ -dd 80 --prod gamechangerml/src/search/ranking/generated_files/prod_test_data.csv

optional arguments:
    --corpus, -c Corpus directory
    --days -dd days since today to get data
"""


logger = configure_logger()
corpus_dir = "test/corpus_new"
prod_data_file = join(
    DATA_PATH, "features", "generated_files", "prod_test_data.csv"
)


def generate_ft_doc(
    corpus_dir: str, days: int = 80, prod_data: str = prod_data_file
):
    """generate feature document
    Args:
        corpus_dir: corpus directory
        days: how many days to retrieve data
    Returns:

    """
    out_dir = join(DATA_PATH, "features", "generated_files")

    # Until we get connection to prod data, load prod data from csv.
    logger.info(f"Reading in prod data from {prod_data}")
    prod_df = read_csv(prod_data)
    logger.info("Getting top keywords")

    corpus_docs = create_corpus(corpus_dir, logger)
    corp_df = DataFrame(corpus_docs)

    # Calculate pagerank scores and add them to corp_df.
    logger.info(f"Calculating pagerank scores for {len(corp_df)} documents.")
    pagerank_df = Rank.get_pr_docs(corpus_docs)
    corp_df = corp_df.merge(pagerank_df)

    # Generate popular documents and merge into corp_df.
    logger.info("Generating popular docs")
    pop_docs = generate_pop_docs(prod_df, corp_df)
    corp_df = corp_df.merge(pop_docs, on="id", how="outer")

    logger.info(f"Saving corpus meta to {out_dir}")
    corp_df.to_csv(join(out_dir, "corpus_meta.csv"))


def create_corpus(directory, logger, nlp=None):
    """Load corpus files from the given directory and format them as needed 
    generating feature documents.

    Args:
        directory (str): Path to directory of corpus (JSON) files.
        logger (logging.Logger)
        nlp (spacy.lang.en.English or None, optional): Spacy language model.
            If None, uses `en_core_web_md`. Default is None.

    Returns:
        list of dict: Corpus documents as a list of dictionaries with keys: 
            `id`, `doc_id`, `doc_num`, `text`, `keywords`, `orgs`, 
            `text_length`, and `ref_list`.
    """
    if nlp is None:
        nlp = en_core_web_md.load()

    common_orgs = load_common_orgs()
    data = []
    docs = [load_json(path) for path in glob(join(directory, "*json"))]

    # Only keep documents that have the required keys.
    required_keys = [
        "id",
        "doc_type",
        "doc_num",
        "text",
        "keyw5",
        "page_count",
        "ref_list"
    ]
    orig_num_docs = len(docs)
    docs = [doc for doc in docs if all(key in doc for key in required_keys)]
    new_num_docs = len(docs)
    if orig_num_docs < new_num_docs:
        logger.warning(
            f"{orig_num_docs - new_num_docs} documents were skipped because "
            f"they were missing at least 1 key from {required_keys}."
        )

    # Normalize text lengths.
    text_lengths = [
        len(preprocess(doc["text"] / doc["page_count"], remove_stopwords=True))
        for doc in docs
    ]
    min_length = min(text_lengths)
    max_length = max(text_lengths)
    text_lengths = [
        (doc_length - min_length) / (max_length - min_length)
        if doc_length != 0
        else 0.0001
        for doc_length in text_lengths
    ]

    for i in range(len(docs)):
        doc = docs[i]
        # Entity extraction isn't supported for large texts.
        text = doc["text"][:999999]
        # TODO: Is spacy model the best option here? It may be more efficient
        # to use string or regex search since we're only keeping values that
        # exist in common_entities anyway.
        entities = [
            ent.text
            for ent in doc(text).ents
            if ent.label_ == "ORG" and ent.text in common_orgs
        ]
        data.append(
            {
                "id": doc["id"],
                "doc_id": " ".join([doc["doc_type"], doc["doc_num"]]),
                "keywords": doc["keyw5"],
                "orgs": dict(Counter(entities).most_common()),
                "text_length": text_lengths[i],
                "ref_list": doc["ref_list"],
            }
        )

    return data


def load_common_orgs():
    """Load the file that contains common "ORG" entities.

    Returns:
        list
    """
    orgs = read_csv(
        join(DATA_PATH, "features", "generated_files", "common_orgs.csv")
    )
    orgs = list(orgs["org"])

    return orgs


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate Features CSV")
    parser.add_argument(
        "--corpus", "-c", dest="corpus_dir", help="corpus directory, full path"
    )
    parser.add_argument(
        "--days",
        "-dd",
        dest="day_delta",
        default=80,
        help="days of data to grab since todays date",
    )
    # Until we can pull data from postgres from production automatically 
    # (currently avail in dev)
    parser.add_argument(
        "--prod",
        "-p",
        dest="prod_data",
        default=join(
            DATA_PATH,
            "features",
            "generated_files",
            "prod_test_data.csv",
        ),
        help="production data historical search logs csv ",
    )
    args = parser.parse_args()
    corpus_dir = args.corpus_dir
    days = args.day_delta
    prod_data = args.prod_data

    generate_ft_doc(corpus_dir, days, prod_data)
