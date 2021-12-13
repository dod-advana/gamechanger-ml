from gamechangerml.src.model_testing.evaluation import SQuADQAEvaluator, IndomainQAEvaluator, IndomainRetrieverEvaluator, MSMarcoRetrieverEvaluator, NLIEvaluator, QexpEvaluator
from gamechangerml.configs.config import QAConfig, EmbedderConfig, SimilarityConfig, QexpConfig
from gamechangerml.src.utilities.test_utils import *
from gamechangerml.api.utils.logger import logger
import argparse
import os
from gamechangerml import DATA_PATH, MODEL_PATH

def eval_qa(model_name, sample_limit, eval_type="original"):
    if eval_type=="original":
        logger.info(f"Evaluating QA model on SQuAD dataset with sample limit of {str(sample_limit)}.")
        originalEval = SQuADQAEvaluator(model_name=model_name, sample_limit=sample_limit, **QAConfig.MODEL_ARGS)
        return originalEval.results
    elif eval_type == "domain":
        logger.info("No in-domain gamechanger evaluation available for the QA model.")
    else:
        logger.info("No eval_type selected. Options: ['original', 'gamechanger'].")

def eval_sent(model_name, validation_data, eval_type="domain"):
    metadata = open_json('metadata.json', os.path.join(MODEL_PATH, model_name))
    encoder = metadata['encoder_model']
    logger.info(f"Evaluating {model_name} created with {encoder}")
    if eval_type == "domain":
        if validation_data != "latest":
            if os.path.exists(os.path.join(DATA_PATH, "validation", "domain", "sent_transformer", validation_data)):
                data_path = os.path.join(DATA_PATH, "validation", "domain", "sent_transformer", validation_data)
            else:
                logger.warning("Could not load validation data, path doesn't exist")
                data_path = None
        else:
            data_path = None
        results = {}
        for level in ['gold', 'silver']:
            domainEval = IndomainRetrieverEvaluator(index=model_name, data_path=data_path, data_level=level, encoder_model_name=encoder, sim_model_name=SimilarityConfig.BASE_MODEL, **EmbedderConfig.MODEL_ARGS)
            results[level] = domainEval.results
    elif eval_type == "original":
        originalEval = MSMarcoRetrieverEvaluator(**EmbedderConfig.MODEL_ARGS, encoder_model_name=EmbedderConfig.BASE_MODEL, sim_model_name=SimilarityConfig.BASE_MODEL)
        results = originalEval.results
    else:
        logger.info("No eval_type selected. Options: ['original', 'domain'].")
        
    return results

def eval_sim(model_name, sample_limit, eval_type="original"):
    if eval_type=="original":
        logger.info(f"Evaluating sim model on NLI dataset with sample limit of {str(sample_limit)}.")
        originalEval = NLIEvaluator(sample_limit=sample_limit, sim_model_name=model_name)
        results = originalEval.results
        logger.info(f"Evals: {str(results)}")
        return results
    elif eval_type == "domain":
        logger.info("No in-domain evaluation available for the sim model.")
    else:
        logger.info("No eval_type selected. Options: ['original', 'domain'].")

def eval_qe(model_name):
    domainEval = QexpEvaluator(qe_model_dir=os.path.join(MODEL_PATH, model_name), **QexpConfig.MODEL_ARGS['init'], **QexpConfig.MODEL_ARGS['expansion'])
    results = domainEval.results
    logger.info(f"Evals: {str(results)}")
    return results

def _squad(limit):
    logger.info("\nEvaluating QA with SQuAD Data...")
    QAEval = SQuADQAEvaluator(model=None, sample_limit=limit, model_name=QAConfig.BASE_MODEL, **QAConfig.MODEL_ARGS)
    logger.info(QAEval.results)
    return

def _gc_qa(limit):
    logger.info("\nEvaluating QA with in-domain data...")
    GCEval = IndomainQAEvaluator(model=None, model_name=QAConfig.BASE_MODEL, **QAConfig.MODEL_ARGS)
    logger.info(GCEval.results)
    return

def _gc_retriever(limit):
    logger.info("\nEvaluating Retriever with in-domain data...")
    GoldStandardRetrieverEval = IndomainRetrieverEvaluator(encoder=None, retriever=None, index='sent_index_20211020', **EmbedderConfig.MODEL_ARGS, encoder_model_name= EmbedderConfig.BASE_MODEL, sim_model_name=SimilarityConfig.BASE_MODEL)
    logger.info(GoldStandardRetrieverEval.results)
    return

def _msmarco(limit):
    logger.info("\nEvaluating Retriever with MSMarco Data...")
    MSMarcoEval = MSMarcoRetrieverEvaluator(encoder=None, retriever=None, **EmbedderConfig.MODEL_ARGS, encoder_model_name=EmbedderConfig.BASE_MODEL, sim_model_name=SimilarityConfig.BASE_MODEL)
    logger.info(MSMarcoEval.results)
    return

def _nli(limit):
    logger.info("\nEvaluating Similarity Model with NLI Data...")
    SimilarityEval = NLIEvaluator(model=None, sample_limit=limit, sim_model_name=SimilarityConfig.BASE_MODEL)
    logger.info(SimilarityEval.results)
    return

def _qexp(limit):
    logger.info("\nEvaluating Query Expansion with GC data...")
    QEEval = QexpEvaluator(qe_model_dir=os.path.join(MODEL_PATH, "qexp_20201217"), **QexpConfig.MODEL_ARGS['init'], **QexpConfig.MODEL_ARGS['expansion'])
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

def main(limit, evals):

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
        help="list of evals to run. Options: 'msmarco', 'squad', 'nli', 'gc_retriever', 'gc_qa'")

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