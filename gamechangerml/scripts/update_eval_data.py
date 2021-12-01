import os
import json
from datetime import date
from gamechangerml.src.model_testing.validation_data import IntelSearchData
from gamechangerml.configs.config import ValidationConfig
from gamechangerml.src.utilities.test_utils import (
    make_timestamp_directory, check_directory, CustomJSONizer
    )

SUB_DIR = "gamechangerml/data/validation/domain/sent_transformer"

def make_tiered_eval_data():
    
    save_dir = make_timestamp_directory(SUB_DIR)

    def save_data(level, min_correct_matches, max_results, start_date, end_date, exclude_searches, save_dir=save_dir):

        min_matches = min_correct_matches[level]
        max_res = max_results[level]

        intel = IntelSearchData(
                    start_date=start_date,
                    end_date=end_date,
                    exclude_searches=exclude_searches,
                    min_correct_matches=min_matches,
                    max_results=max_res
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
            "min_correct_matches": min_matches,
            "max_results": max_res
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
        level='any',
        **ValidationConfig.TRAINING_ARGS
        )
    
    silver_data = save_data(
        level='silver',
        **ValidationConfig.TRAINING_ARGS
        )
    
    gold_data = save_data(
        level='gold',
        **ValidationConfig.TRAINING_ARGS
        )
    
    return all_data, silver_data, gold_data

if __name__ == '__main__':

    make_tiered_eval_data()