from gamechangerml.src.model_testing.evaluation import SQuADQAEvaluator, IndomainQAEvaluator, IndomainRetrieverEvaluator, MSMarcoRetrieverEvaluator, SimilarityEvaluator
from gamechangerml.configs.config import QAConfig
#from gamechangerml.api.utils.logger import logger

if __name__ == '__main__':

    print("\nStarting QA Evaluation...")
    #QAEval = QAEvaluator(**QAConfig.MODEL_ARGS, new_model=True)
    print("\nEvaluating QA with SQuAD Data...")
    QAEval = SQuADQAEvaluator(**QAConfig.MODEL_ARGS, new_model=True)
    print(QAEval.results)
    print("\nEvaluating QA with in-domain data...")
    GCEval = IndomainQAEvaluator(**QAConfig.MODEL_ARGS, new_model=True)
    print(GCEval.results)

    print("\nStarting Retriever Evaluation...")
    print("\nEvaluating Retriever with MSMarco Data...")
    MSMarcoEval = MSMarcoRetrieverEvaluator(new_model=True)
    print(MSMarcoEval.results)
    print("\nEvaluating Retriever with in-domain data...")
    GoldStandardRetrieverEval = IndomainRetrieverEvaluator(new_model=True)
    print(GoldStandardRetrieverEval.results)

    print("\nLoading Similarity Evaluation...")
    SimilarityEval = SimilarityEvaluator(new_model=True, sample_limit=10)
    print("\nEvaluating Similarity Model with NLI Data...")
    print(SimilarityEval.results)
    #print("Evaluating Similarity Model with in-domain data...")