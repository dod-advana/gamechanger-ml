from pandas import DataFrame
from re import split, IGNORECASE
from collections import Counter
from tqdm import tqdm
from gamechangerml.src.text_handling.process import preprocess


def score_search_keywords(search_df: DataFrame()):
    """Calculate popularity scores for search keywords.

    Args:
        search_df: (pandas.DataFrame) DataFrame of search data. Should have
            column `search` with str items. To include multiple values in an
            item of the `search` column, the item can be a string of values
            combined by ` AND | OR`.
    Returns:
        pandas.DataFrame: DataFrame with columns:
            - "keywords": (str) Keywords from searches
            - "amt": (int) The number of searches a keyword appeared in
            - "pop_score": (float) The calculated popularity score for keywords

            The DataFrame will be sorted in descending order by `pop_score`.
    """
    kw_counts = Counter()
    ignore_words = ["artificial intelligence president"]

    for log in tqdm(search_df.itertuples()):
        terms = split(" AND | OR", log.search, flags=IGNORECASE)
        terms = [" ".join(preprocess(term)) for term in terms]
        terms = [term for term in terms if term not in ignore_words]
        for term in terms:
            kw_counts[term] += 1

    df = DataFrame(columns=["keywords", "amt"], data=kw_counts.items())
    # Add popularity scores and sort the DataFrame by them.
    df["pop_score"] = (df.amt - df.amt.min()) / df.amt.max()
    df.sort_values(["pop_score"], ascending=False, inplace=True)

    return df


def generate_popular_documents(
    prod_df: DataFrame, corpus_df: DataFrame
) -> DataFrame:
    """Create a DataFrame of corpus documents that are popular based on prod
    searches.

    Args:
        prod_df (pandas.DataFrame): DataFrame of prod search data. Expected to
            have the column `search` with str items. To include multiple values
            in an item of the `search` column, the item can be a string of
            values combined by ` AND | OR`.
        corpus_df (pandas.DataFrame): DataFrame of corpus documents. Expected
            to contain the columns `id` and `keywords`.

    Returns:
        DataFrame: DataFrame with columns `id`, `keywords`, and
            `kw_in_doc_score`.
    """
    kw_df = score_search_keywords(prod_df)
    docs = []

    for row_kw in tqdm(kw_df.itertuples()):
        for row_corp in corpus_df.itertuples():
            if len(row_corp.keywords):
                if row_kw.keywords in row_corp.keywords[0]:
                    docs.append(
                        {"id": row_corp.id, "keywords": row_kw.keywords}
                    )
    docs_df = DataFrame(docs, columns=["id", "keywords"])
    counts_df = (
        docs_df.groupby("id").count().sort_values("keywords", ascending=False)
    )

    max_ = counts_df["keywords"].max()
    min_ = counts_df["keywords"].min()
    scores = [
        (score - min_) / (max_ - min_) if score != 0 else 0.00001
        for score in list(counts_df["keywords"])
    ]
    counts_df["keywords"] = scores
    counts_df.rename(columns={"keywords": "kw_in_doc_score"}, inplace=True)

    return counts_df
