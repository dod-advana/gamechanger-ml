from sentence_transformers.losses import TripletDistanceMetric


class BiEncoderConfig:
    # Models tuned for cosine-similarity will prefer the retrieval of shorter
    # passages, while models for dot-product will prefer the retrieval of longer
    # passages.
    BASE_MODEL = "sentence-transformers/msmarco-distilbert-cos-v5"  # TODO here
    DISTANCE_METRIC = TripletDistanceMetric.COSINE
    MINIMUM_PASSAGE_LENGTH = 60
    MINIMUM_LINE_LENGTH = 30
    RANDOM_STATE = 123
    SPACY_MODEL = "en_core_web_md"
    NOUN_REPLACEMENT_MINIMUM = 0.1
    NOUN_REPLACEMENT_LIMIT = None
