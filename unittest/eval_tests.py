from gamechangerml.src.model_testing.evaluation import SQuADQAEvaluator, IndomainQAEvaluator, IndomainRetrieverEvaluator, MSMarcoRetrieverEvaluator, NLIEvaluator, QexpEvaluator
from gamechangerml.configs.config import QAConfig, EmbedderConfig, SimilarityConfig, QexpConfig

limit = 10

def _squad(limit):
    print("\nEvaluating QA with SQuAD Data...")
    QAEval = SQuADQAEvaluator(model=None, sample_limit=limit, model_name=QAConfig.BASE_MODEL, **QAConfig.MODEL_ARGS)
    print(QAEval.results)
    assert QAEval.results != {}
    return

def _gc_qa(limit):
    print("\nEvaluating QA with in-domain data...")
    GCEval = IndomainQAEvaluator(model=None, model_name=QAConfig.BASE_MODEL, **QAConfig.MODEL_ARGS)
    print(GCEval.results)
    return

def _gc_retriever(limit):
    print("\nEvaluating Retriever with in-domain data...")
    GoldStandardRetrieverEval = IndomainRetrieverEvaluator(encoder=None, retriever=None, index='sent_index_20211020', **EmbedderConfig.MODEL_ARGS, encoder_model_name= EmbedderConfig.BASE_MODEL, sim_model_name=SimilarityConfig.BASE_MODEL)
    print(GoldStandardRetrieverEval.results)
    return

def _msmarco(limit):
    print("\nEvaluating Retriever with MSMarco Data...")
    MSMarcoEval = MSMarcoRetrieverEvaluator(encoder=None, retriever=None, **EmbedderConfig.MODEL_ARGS, encoder_model_name=EmbedderConfig.BASE_MODEL, sim_model_name=SimilarityConfig.BASE_MODEL)
    print(MSMarcoEval.results)
    return

def _nli(limit):
    print("\nEvaluating Similarity Model with NLI Data...")
    SimilarityEval = NLIEvaluator(model=None, sample_limit=limit, sim_model_name=SimilarityConfig.BASE_MODEL)
    print(SimilarityEval.results)
    return

def _qexp(limit):
    print("\nEvaluating Query Expansion with GC data...")
    QEEval = QexpEvaluator(qe_model_dir = 'gamechangerml/models/qexp_20201217', **QexpConfig.MODEL_ARGS['init'], **QexpConfig.MODEL_ARGS['expansion'])
    print(QEEval.results)