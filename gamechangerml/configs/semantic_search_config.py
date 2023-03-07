class SemanticSearchConfig:
    """Configurations for the SemanticSearch class."""

    """Base model name"""
    BASE_MODEL = "msmarco-distilbert-base-v2"

    """True to load the index from a file, False otherwise. If False, you must
    build the index."""
    LOAD_INDEX_FROM_FILE = True

    USE_GPU = False

    """Default value for `threshold` argument when SemanticSearch.search() is
    called in the ML API."""
    DEFAULT_THRESHOLD_ARG = "auto"

    """If no threshold is recommended in evaluations, or an exception is thrown
    when calculating the auto threshold, this is the default minimum score for
    for a search result to be considered a passing result.
    """
    DEFAULT_THRESHOLD_FLOAT = 0.7

    """Makes the default threshold less strict. To use exact default, set to 1."""
    EVAL_THRESHOLD_MULTIPLIER = 0.8

    """Flag for whether or not items with scores below the threshold should be
    included in results."""
    INCLUDE_RESULTS_BELOW_THRESHOLD = True

    """Arguments for SemanticSearchFinetuner."""
    FINETUNE = {
        "shuffle": True,
        "batch_size": 32,
        "epochs": 3,
        "warmup_steps": 100,
    }
