from datetime import date
import pandas as pd
import argparse
from gamechangerml.src.paths import COMBINED_ENTITIES_FILE
from gamechangerml.src.featurization.make_meta import lookup_wiki_summary

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Getting entity descriptions")
    
    parser.add_argument("--filepath", "-p", dest="filepath", help="path to csv with entities")

    args = parser.parse_args()

    if args.filepath:
        entities_filepath = args.filepath
    else:
        entities_filepath = COMBINED_ENTITIES_FILE
    df = pd.read_csv(entities_filepath)
    df['information'] = df['entity_name'].apply(lambda x: lookup_wiki_summary(x))
    df['information_source'] = "Wikipedia"
    df['information_retrieved'] = date.today().strftime("%Y-%m-%d")
    df.to_csv(entities_filepath)

    print(f"Saved csv to {entities_filepath}")
