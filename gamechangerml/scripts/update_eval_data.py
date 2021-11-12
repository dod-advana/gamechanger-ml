import argparse
import os
from gamechangerml.src.model_testing.validation_data import IntelSearchData
from gamechangerml.configs.config import ValidationConfig
from gamechangerml.src.utilities.test_utils import *
from gamechangerml.api.utils.logger import logger

exclude_searches=ValidationConfig.DATA_ARGS['exclude_searches']

def make_tiered_eval_data(
    min_correct_matches, 
    max_results, 
    start_date, 
    end_date, 
    exclude_searches
    ):
    
    sub_dir = os.path.join('gamechangerml/data/validation', 'sent_transformer')
    save_dir = make_timestamp_directory(sub_dir)

    def save_data(level, min_correct_matches, max_results, start_date, end_date, exclude_searches, save_dir=save_dir):

        intel = IntelSearchData(
                    start_date=start_date,
                    end_date=end_date,
                    exclude_searches=exclude_searches,
                    min_correct_matches=min_correct_matches,
                    max_results=max_results
                )

        save_intel = {
            "queries": intel.queries, 
            "collection": intel.collection, 
            "meta_relations": intel.all_relations,
            "correct": intel.correct,
            "incorrect": intel.incorrect}

        metadata = {
            "date_created": str(date.today()),
            "level": level,
            "number_queries": len(intel.queries),
            "number_documents": len(intel.collection),
            "number_correct": len(intel.correct),
            "number_incorrect": len(intel.incorrect),
            "start_date": start_date,
            "end_date": end_date,
            "exclude_searches": exclude_searches,
            "min_correct_matches": min_correct_matches,
            "max_results": max_results
        }

        save_intel = json.dumps(save_intel, cls=CustomJSONizer)
        intel_path = check_directory(os.path.join(save_dir, level))
        intel_file = os.path.join(intel_path, 'intelligent_search_data.json')
        metafile =  os.path.join(intel_path, 'intelligent_search_metadata.json')
        with open(intel_file, "w") as outfile:
            json.dump(save_intel, outfile)
        
        with open(metafile, "w") as outfile:
            json.dump(metadata, outfile)
        
        return metadata

    all_data = save_data(
        min_correct_matches=1,
        max_results=100,
        level='any',
        start_date=start_date,
        end_date=end_date,
        exclude_searches=exclude_searches
        )
    
    silver_data = save_data(
        min_correct_matches=ValidationConfig.DATA_ARGS['silver_level']['min_correct_matches'],
        max_results=ValidationConfig.DATA_ARGS['silver_level']['max_results'],
        level='silver',
        start_date=start_date,
        end_date=end_date,
        exclude_searches=exclude_searches
        )
    
    gold_data = save_data(
        min_correct_matches=min_correct_matches,
        max_results=max_results,
        level='gold',
        start_date=start_date,
        end_date=end_date,
        exclude_searches=exclude_searches
        )
    
    return all_data, silver_data, gold_data

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

    min_correct_matches = args.min_correct_matches if args.min_correct_matches else ValidationConfig.DATA_ARGS['gold_level']['min_correct_matches']
    max_results = args.max_results if args.max_results else ValidationConfig.DATA_ARGS['gold_level']['max_results']
    start_date = args.start_date if args.start_date else ValidationConfig.DATA_ARGS['start_date']
    end_date = args.end_date if args.end_date else ValidationConfig.DATA_ARGS['end_date']
    
    logger.info(f"min_correct_matches: {str(min_correct_matches)}")
    logger.info(f"max_results: {str(max_results)}")
    logger.info(f"start_date: {str(start_date)}")
    logger.info(f"end_date: {str(end_date)}")

    make_tiered_eval_data(min_correct_matches, max_results, start_date, end_date, exclude_searches)