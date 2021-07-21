from gamechangerml.src.model_testing.evaluation import QAEvaluator, RetrieverEvaluator, SimilarityEvaluator 
from gamechangerml.configs.config import QAConfig
#from gamechangerml.api.utils.logger import logger

if __name__ == '__main__':

    print("Loading QA Evaluation...")
    QAEval = QAEvaluator(**QAConfig.MODEL_ARGS, new_model=True)
    print("Evaluating QA with SQuAD Data...")
    print(QAEval.agg_results)
    #print("Evaluating QA with in-domain data...")

    print("Loading Retriever Evaluation...")
    RetrieverEval = RetrieverEvaluator(new_model=True)
    print("Evaluating Retriever with MSMarco Data...")
    print(RetrieverEval.agg_results)
    #print("Evaluating Retriever with in-domain data...")

    print("Loading Similarity Evaluation...")
    SimilarityEval = SimilarityEvaluator(new_model=True)
    print("Evaluating Similarity Model with NLI Data...")
    print(SimilarityEval.agg_results)
    #print("Evaluating Retriever with in-domain data...")