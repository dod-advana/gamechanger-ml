class SemanticConfig:
    """Configurations for the Title Searcher/ Similarity model."""

    """Base model name."""
    BASE_MODEL = "msmarco-distilbert-base-v2"

    """Arguments used to load the model.

    

        min_token_len

        verbose: for creating LocalCorpus

        return_id: for creating LocalCorpus

    """

    MODEL_ARGS = {

        "min_token_len": 25,

        "verbose": True,

        "return_id": True,

    }