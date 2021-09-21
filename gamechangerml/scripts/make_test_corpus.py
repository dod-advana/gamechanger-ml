from os import listdir
from os.path import isfile, join
import random
import argparse
from gamechangerml.src.utilities.test_utils import (check_file_size, check_directory, open_json, save_json)
from gamechangerml.api.utils.logger import logger

def main(test_size, corpus_directory, save_directory, include_ids=None, max_file_size=100000):
    '''Makes a small test corpus for checking validation'''
    all_files = [f.split('.json')[0] + '.json' for f in listdir(corpus_directory) if isfile(join(corpus_directory, f))]
    if include_ids:

        print(include_ids)
        include_ids = [f.split('.json')[0] + '.json' for f in include_ids]
        subset = list(set(all_files).intersection(include_ids))
        other = [i for i in all_files if i not in include_ids]
    else:
        subset = []
        other = all_files
    for i in range(int(test_size) - len(subset)):
        print(i)
        filesize = 1000000
        while filesize > max_file_size: # filter out large files
            random_index = random.randint(0,len(other)-1)
            file = other[random_index]
            filesize = check_file_size(file, corpus_directory)
        subset.append(file)
        subset = list(set(subset)) # remove duplicates

    save_directory = check_directory(save_directory)
    for x in subset:
        f = open_json(x, corpus_directory)
        save_json(x, save_directory, f)
    
    return

if __name__ == "__main__":
   
    parser = argparse.ArgumentParser(description="Profile Corpus")

    parser.add_argument("--test-size", "-ts", dest="test_size", required=True, help="size of test corpus to generate")
    parser.add_argument("--corpus-directory", "-c", dest="corpus_directory", required=True, help="path to full corpus")
    parser.add_argument("--save-directory", "-s", dest="save_directory", required=True, help="path for saving test corpus")
    parser.add_argument("--include-ids", "-i", dest="include_ids", nargs="+", required=False, help="list of docids/filenames to include in the corpus")
    parser.add_argument("--max-file-size", "-m", dest="max_file_size", required=False, help="max size (in bytes) of file to save to test_corpus")

    args = parser.parse_args()
    if not args.include_ids:
        args.include_ids = None
    if not args.max_file_size:
        args.max_file_size = 100000
    
    logger.info("Creating a new test corpus of {} docs; saving to {}.".format(args.test_size, args.save_directory))

    main(test_size=args.test_size, corpus_directory=args.corpus_directory, save_directory=args.save_directory, include_ids=args.include_ids, max_file_size=args.max_file_size)

    logger.info("Finished making test corpus.")