from gamechangerml.src.model_testing.evaluation import SQuADQAEvaluator, IndomainQAEvaluator, IndomainRetrieverEvaluator, MSMarcoRetrieverEvaluator, SimilarityEvaluator
from gamechangerml.api.utils.logger import logger

if __name__ == '__main__':

    logger.info("\nStarting QA Evaluation...")
    logger.info("\nEvaluating QA with SQuAD Data...")
    QAEval = SQuADQAEvaluator(model=None, sample_limit=100)
    logger.info(QAEval.results)
    logger.info("\nEvaluating QA with in-domain data...")
    GCEval = IndomainQAEvaluator(model=None)
    logger.info(GCEval.results)

    logger.info("\nStarting Retriever Evaluation...")
    logger.info("\nEvaluating Retriever with MSMarco Data...")
    MSMarcoEval = MSMarcoRetrieverEvaluator(encoder=None, retriever=None)
    logger.info(MSMarcoEval.results)
    logger.info("\nEvaluating Retriever with in-domain data...")
    GoldStandardRetrieverEval = IndomainRetrieverEvaluator(encoder=None, retriever=None)
    logger.info(GoldStandardRetrieverEval.results)

    logger.info("\nLoading Similarity Evaluation...")
    SimilarityEval = SimilarityEvaluator(model=None, sample_limit=10)
    logger.info("\nEvaluating Similarity Model with NLI Data...")
    logger.info(SimilarityEval.results)
    #print("Evaluating Similarity Model with in-domain data...")