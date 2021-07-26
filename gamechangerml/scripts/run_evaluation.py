from gamechangerml.src.model_testing.evaluation import QAEvaluator, MSMarcoEvaluator, SimilarityEvaluator
from gamechangerml.configs.config import QAConfig
#from gamechangerml.api.utils.logger import logger

if __name__ == '__main__':

    print("\nLoading QA Evaluation...")
    QAEval = QAEvaluator(**QAConfig.MODEL_ARGS, new_model=True, new_data=True)
    print("Evaluating QA with SQuAD Data...")
    print(QAEval.squad_results)
    print("Evaluating QA with in-domain data...")
    print(QAEval.domain_results)
    

    print("\nLoading Retriever Evaluation...")
    MSMarcoEval = MSMarcoEvaluator(new_model=True)
    print("Evaluating Retriever with MSMarco Data...")
    print(MSMarcoEval.results)
    #print("Evaluating Retriever with in-domain data...")
    #print(GoldStandardRetrieverEval.results)

    print("\nLoading Similarity Evaluation...")
    SimilarityEval = SimilarityEvaluator(new_model=True, new_data=True, sample_limit=10)
    print("Evaluating Similarity Model with NLI Data...")
    print(SimilarityEval.agg_results)
    #print("Evaluating Similarity Model with in-domain data...")