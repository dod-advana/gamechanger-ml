import logging
import os
import random
from fnmatch import fnmatch
import shutil

logger = logging.getLogger(__name__)


def mv_rand_sent_csv(src_dir, output_dir, nfiles):
    all_files = [f for f in os.listdir(src_dir) if fnmatch(f, "*.csv")]
    random.shuffle(all_files)
    for src_file in all_files[:nfiles]:
        shutil.copy(os.path.join(src_dir, src_file), output_dir)


if __name__ == "__main__":
    from argparse import ArgumentParser

    import gamechangerml.src.text_classif.utils.log_init as li

    li.initialize_logger(to_file=False, log_name="none")

    fp_ = "python " + os.path.split(__file__)[-1]
    parser = ArgumentParser(
        prog=fp_,
        description="Select nfiles at random and move to the output_dir",
    )
    parser.add_argument(
        "-s",
        "--source-dir",
        dest="src_dir",
        type=str,
        help="source dir of csv's of sentences and labels",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        type=str,
        help="where to write the selected files",
        required=True,
    )
    parser.add_argument(
        "-n",
        "--num-files",
        dest="num_files",
        type=int,
        help="how many randomly chosen files",
        required=True,
    )
    args = parser.parse_args()
    mv_rand_sent_csv(args.src_dir, args.output_dir, args.num_files)
