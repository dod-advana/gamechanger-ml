"""Class definition for DocumentComparison, which is used in the ML API 
/documentCompare endpoint."""

from gamechangerml.configs import DocumentComparisonConfig
from ..semantic_search import SemanticSearch


class DocumentComparison(SemanticSearch):
    def __init__(
        self,
        model_path,
        index_directory_path,
        load_index_from_file,
        logger,
        use_gpu,
    ):
        super().__init__(
            model_path,
            index_directory_path,
            load_index_from_file,
            logger,
            use_gpu,
        )

    def get_default_threshold(self) -> float:
        return DocumentComparisonConfig.DEFAULT_THRESHOLD_FLOAT

    def get_score_display_map(self) -> dict:
        return DocumentComparisonConfig.SCORE_DISPLAY_MAP

    def should_include_results_below_threshold(self) -> bool:
        return DocumentComparisonConfig.INCLUDE_RESULTS_BELOW_THRESHOLD
