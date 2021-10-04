import argparse
from gamechangerml.src.model_testing.training_data import SentenceTransformerTD
from gamechangerml.configs.config import TrainingConfig
from gamechangerml.api.utils.logger import logger

def main(start_date, end_date, min_correct_matches, max_results, exclude_searches, base_dir, train_test_split_ratio, trigger_es):

    ## add function to update existing or start from scratch

    training_data = SentenceTransformerTD(
        trigger_es=trigger_es,
        max_results=max_results,
        min_correct_matches=min_correct_matches,
        start_date=start_date,
        end_date=end_date,
        exclude_searches=exclude_searches,
        base_dir=base_dir, 
        train_test_split_ratio=train_test_split_ratio
    )

    return

    ## make gold level eval data

    ## make silver level eval data

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Creating updated training and evaluation data.")
    
    parser.add_argument(
        "--min-correct-matches", "-mcm", 
        dest="min_correct_matches", 
        required=False,
        type=int,
        help="minimum number of clicks on a document to be counted from search history"
        )

    parser.add_argument(
        "--max-results", "-mr", 
        dest="max_results", 
        required=False,
        type=int,
        help="maximum number of unique search results (clicked on) a search can have to be counted from search history"
        )

    parser.add_argument(
        "--start-date", "-sd", 
        dest="start_date", 
        required=False,
        help="start date for data to be included"
        )

    parser.add_argument(
        "--end-date", "-ed", 
        dest="end_date", 
        required=False,
        help="end date for data to be included"
        )

    args = parser.parse_args()

    exclude_searches=TrainingConfig.DATA_ARGS['exclude_searches']
    base_dir=TrainingConfig.DATA_ARGS['training_data_dir']
    train_test_split_ratio=TrainingConfig.DATA_ARGS['train_test_split_ratio']
    trigger_es=False

    min_correct_matches = args.min_correct_matches if args.min_correct_matches else TrainingConfig.DATA_ARGS['gold_level']['min_correct_matches']
    max_results = args.max_results if args.max_results else TrainingConfig.DATA_ARGS['gold_level']['max_results']
    start_date = args.start_date if args.start_date else TrainingConfig.DATA_ARGS['start_date']
    end_date = args.end_date if args.end_date else TrainingConfig.DATA_ARGS['end_date']

    logger.info(f"exclude searches: {str(exclude_searches)}")
    logger.info(f"base_dir: {str(base_dir)}")
    logger.info(f"train_test_split_ratio: {str(train_test_split_ratio)}")
    logger.info(f"trigger_es: {str(trigger_es)}")
    logger.info(f"min_correct_matches: {str(min_correct_matches)}")
    logger.info(f"max_results: {str(max_results)}")
    logger.info(f"start_date: {str(start_date)}")
    logger.info(f"end_date: {str(end_date)}")

    main(start_date, end_date, min_correct_matches, max_results, exclude_searches, base_dir, train_test_split_ratio, trigger_es)