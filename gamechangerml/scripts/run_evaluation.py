from gamechangerml.src.model_testing.evaluation import SQuADQAEvaluator, IndomainQAEvaluator, IndomainRetrieverEvaluator, MSMarcoRetrieverEvaluator, NLIEvaluator, QexpEvaluator
from gamechangerml.configs.config import QAConfig, EmbedderConfig, SimilarityConfig, QexpConfig
from gamechangerml.api.utils.logger import logger

if __name__ == '__main__':

    logger.info("\nStarting QA Evaluation...")
    logger.info("\nEvaluating QA with SQuAD Data...")
    QAEval = SQuADQAEvaluator(model=None, sample_limit=20, **QAConfig.MODEL_ARGS)
    logger.info(QAEval.results)
    logger.info("\nEvaluating QA with in-domain data...")
    GCEval = IndomainQAEvaluator(model=None, **QAConfig.MODEL_ARGS)
    logger.info(GCEval.results)

    logger.info("\nStarting Retriever Evaluation...")
    logger.info("\nEvaluating Retriever with MSMarco Data...")
    MSMarcoEval = MSMarcoRetrieverEvaluator(encoder=None, retriever=None, **EmbedderConfig.MODEL_ARGS, **SimilarityConfig.MODEL_ARGS)
    logger.info(MSMarcoEval.results)
    logger.info("\nEvaluating Retriever with in-domain data...")
    GoldStandardRetrieverEval = IndomainRetrieverEvaluator(encoder=None, retriever=None, **EmbedderConfig.MODEL_ARGS, **SimilarityConfig.MODEL_ARGS)
    logger.info(GoldStandardRetrieverEval.results)

    logger.info("\nLoading Similarity Evaluation...")
    SimilarityEval = NLIEvaluator(model=None, sample_limit=10, **SimilarityConfig.MODEL_ARGS)
    logger.info("\nEvaluating Similarity Model with NLI Data...")
    logger.info(SimilarityEval.results)
    #print("Evaluating Similarity Model with in-domain data...")

    logger.info("\nLoading Query Expansion Evaluation...")
    QEEval = QexpEvaluator(qe_model_dir = 'gamechangerml/models/qexp_20201217', **QexpConfig.MODEL_ARGS['init'], **QexpConfig.MODEL_ARGS['expansion'])
    logger.info("\nEvaluating Query Expansion with GC data...")
    logger.info(QEEval.results)