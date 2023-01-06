from .semantic_search_config import SemanticSearchConfig


class DocumentComparisonConfig(SemanticSearchConfig):
    """Configurations for the DocumentComparison class."""

    # Metadata for the model these scores were derived from
    # {"user": null, "date_started": "2022-04-29 16:06:06", "date_finished": "2022-04-29 19:52:52", "doc_id_count": 1495122, "corpus_name": "/opt/app-root/src/gamechangerml/corpus", "encoder_model": "multi-qa-MiniLM-L6-cos-v1"}
    SCORE_DISPLAY_MAP = {
        0.8: "High",
        0.5: "Medium",
        0.4: "Low",
        0.0: "Very Low",
    }

    DEFAULT_THRESHOLD_FLOAT = 0.25

    DEFAULT_THRESHOLD_ARG = DEFAULT_THRESHOLD_FLOAT

    INCLUDE_RESULTS_BELOW_THRESHOLD = False
