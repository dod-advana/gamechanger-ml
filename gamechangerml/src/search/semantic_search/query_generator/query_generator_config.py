class QueryGeneratorConfig:
    """Configurations for the QueryGenerator class.

    Attributes:
    -------

    `BASE_MODEL_NAME` (str): Name of the base model to use.

    `MAX_LENGTH` (int): The maximum length the generated query tokens can have.
        Reference: https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/text_generation#transformers.GenerationConfig.max_length

    `TOP_P` (float): If set to float < 1, only the smallest set of most probable
        tokens with probabilities that add up to top_p or higher are kept for
        generation.
        Reference: https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/text_generation#transformers.GenerationConfig.top_p

    `DO_SAMPLE` (bool): Whether or not to use sampling; use greedy decoding
        otherwise.
        Reference: https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/text_generation#transformers.GenerationConfig.do_sample
    """

    BASE_MODEL = "BeIR/query-gen-msmarco-t5-base-v1"
    MAX_LENGTH = 64
    TOP_P = 0.95
    DO_SAMPLE = True
