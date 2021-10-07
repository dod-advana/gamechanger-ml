from gamechangerml.src.model_testing.evaluation import SQuADQAEvaluator, IndomainQAEvaluator, IndomainRetrieverEvaluator, MSMarcoRetrieverEvaluator, NLIEvaluator, QexpEvaluator
from gamechangerml.configs.config import QAConfig, EmbedderConfig, SimilarityConfig, QexpConfig
from gamechangerml.api.utils.logger import logger
import argparse

def _squad(limit):
    logger.info("\nEvaluating QA with SQuAD Data...")
    QAEval = SQuADQAEvaluator(model=None, sample_limit=limit, **QAConfig.MODEL_ARGS)
    logger.info(QAEval.results)
    return

def _gc_qa(limit):
    logger.info("\nEvaluating QA with in-domain data...")
    GCEval = IndomainQAEvaluator(model=None, **QAConfig.MODEL_ARGS)
    logger.info(GCEval.results)
    return

def _gc_retriever(limit):
    logger.info("\nEvaluating Retriever with in-domain data...")
    GoldStandardRetrieverEval = IndomainRetrieverEvaluator(encoder=None, retriever=None, index='sent_index_20210715', **EmbedderConfig.MODEL_ARGS, **SimilarityConfig.MODEL_ARGS)
    logger.info(GoldStandardRetrieverEval.results)
    return

def _msmarco(limit):
    logger.info("\nEvaluating Retriever with MSMarco Data...")
    MSMarcoEval = MSMarcoRetrieverEvaluator(encoder=None, retriever=None, **EmbedderConfig.MODEL_ARGS, **SimilarityConfig.MODEL_ARGS)
    logger.info(MSMarcoEval.results)
    return

def _nli(limit):
    logger.info("\nEvaluating Similarity Model with NLI Data...")
    SimilarityEval = NLIEvaluator(model=None, sample_limit=limit, **SimilarityConfig.MODEL_ARGS)
    logger.info(SimilarityEval.results)
    return

def _qexp(limit):
    logger.info("\nEvaluating Query Expansion with GC data...")
    QEEval = QexpEvaluator(qe_model_dir = 'gamechangerml/models/qexp_20201217', **QexpConfig.MODEL_ARGS['init'], **QexpConfig.MODEL_ARGS['expansion'])
    logger.info(QEEval.results)

FUNCTION_MAP = {
    "squad": _squad,
    "msmarco": _msmarco,
    "nli": _nli,
    "gc_qa": _gc_qa,
    "gc_retriever": _gc_retriever,
    "qexp": _qexp
}

def run(limit, callback):
    callback(limit)

def main(limit, all_gc, all_og, evals):

    if all_gc:
        run(limit, _gc_qa)
        run(limit, _gc_retriever)
    elif all_og:
        run(limit, _squad)
        run(limit, _msmarco)
        run(limit, _nli)
    elif evals:
        for eval_func in evals:
            run(limit, FUNCTION_MAP[eval_func])
    else:
        print("No arguments passed")
        return

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Evaluate models")

    parser.add_argument(
        "--evals", 
        "-e", 
        dest="evals", 
        nargs="+", 
        required=False, 
        help="list of evals to run. Options: '_msmarco', '_squad', '_nli', '_gc_retriever', '_gc_qa'")

    parser.add_argument(
        "--all-gc", 
        "-gc", 
        dest="all_gc", 
        type=bool,
        required=False, 
        help="If this flag is used, will run all transformer model evaluations on just GC data.")
    
    parser.add_argument(
        "--all-OG", 
        "-og", 
        dest="all_og", 
        type=bool,
        required=False, 
        help="If this flag is used, will run all transformer model evaluations on their original datasets (msmarco, squad, nli)")
    
    parser.add_argument(
        "--sample-limit", 
        "-s", 
        dest="limit", 
        type=int,
        required=False, 
        help="Sample limit")

    args = parser.parse_args()
    evals = args.evals if args.evals else None
    all_og = True if args.all_og else False
    all_gc = True if args.all_gc else False
    limit = args.limit if args.limit else 15000

    main(limit, all_gc, all_og, evals)