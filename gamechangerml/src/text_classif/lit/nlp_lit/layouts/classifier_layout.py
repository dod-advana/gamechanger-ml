from lit_nlp.api import dtypes as lit_dtypes


# Custom frontend layout; see client/lib/types.ts
def classifier_layout():
    LM_LAYOUT = lit_dtypes.LitComponentLayout(
        components={
            "Main": [
                "embeddings-module",
                "data-table-module",
                "datapoint-editor-module",
                "lit-slice-module",
                "color-module",
            ],
            "Predictions": [
                "lm-prediction-module",
                "confusion-matrix-module",
            ],
            "Counterfactuals": ["generator-module"],
        },
        description="Custom layout for language models.",
    )
    return {"lm": LM_LAYOUT}


def classifier_layout_metric():
    LM_LAYOUT = lit_dtypes.LitComponentLayout(
        components={
            "Main": [
                "embeddings-module",
                "data-table-module",
                "datapoint-editor-module",
                "lit-slice-module",
                "color-module",
            ],
            "Predictions": [
                "lm-prediction-module",
                # "salience_map_module",
                "attention-module",
                "confusion-matrix-module",
            ],
            "Counterfactuals": ["generator-module"],
        },
        description="Custom layout for language models.",
    )
    return {"lm": LM_LAYOUT}
