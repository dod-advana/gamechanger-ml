"""Class definition for DocumentComparison."""

from gamechangerml.configs import DocumentComparisonConfig
from ..semantic_search import SemanticSearch


class DocumentComparison(SemanticSearch):
    """Utilizes a sentence transformer model to determine which paragraphs in
    the corpus most closely match paragraph(s) entered by a user.

    This class is used in the ML API /documentCompare endpoint which drives the
    Document Comparison tool in the GAMECHANGER web application.

    Child of SemanticSearch.
    """

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
