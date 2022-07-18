from os import listdir
from os.path import join, isfile
from pandas import read_csv
from json import load


class SentenceTransformerFiles:
    """Manage files used by the SentenceEncoder and SentenceSearcher."""

    # Name of the csv file created in SentenceEncoder.build_index(). Contains input
    # data. Has columns 'text' and 'paragraph_id'."""
    DATA_FILE_NAME = "data.csv"

    # Name of directory that holds eval data.
    EVAL_DIR_NAME = "evals_gc/silver"

    @staticmethod
    def load_data(dir_path):
        """Load the data csv file that contains text and paragraph IDs.

        Args:
            dir_path (str): Path to the directory where the file is stored.

        Returns:
            pandas.DataFrame: DataFrame with columns 'text' (str) and
                'paragraph_id' (str).
        """
        return read_csv(
            join(dir_path, SentenceTransformerFiles.DATA_FILE_NAME),
            dtype={"paragraph_id": str},
        )

    @staticmethod
    def load_most_recent_eval(dir_path):
        """Load the most recent eval file.

        Args:
            dir_path (str): Path to the directory that contains eval files.

        Raises:
            FileNotFoundError

        Returns:
            dict
        """
        error_msg = f"Eval file does not exist in {path}"

        filename = SentenceTransformerFiles.most_recent_eval_name(dir_path)
        if filename is None:
            raise FileNotFoundError(error_msg)

        path = join(dir_path, SentenceTransformerFiles.EVAL_DIR_NAME, filename)
        if not isfile(path):
            raise FileNotFoundError(f"{error_msg} with name {filename}.")

        with open(path) as f:
            file = load(f)

        return file

    @staticmethod
    def most_recent_eval_name(dir_path):
        """Get the most recent eval file name that exists in the given directory.

        Args:
            dir_path (str): Path to directory that contains eval files.

        Returns:
            str or None: If None, no JSON files existed in the directory. If
                str, it is the file name of the most recent eval file in the
                directory.
        """
        json_filenames = [
            f
            for f in listdir(dir_path)
            if f.endswith(".json") and isfile(join(dir_path, f))
        ]
        if json_filenames:
            json_filenames.sort(
                # Sort by:
                # - Getting text after the last "_" in the file name
                # - Then, removing text after (& including) the first "."
                # - Then, removing  "-" characters.
                # - Converting the result to int
                # Example: "test_eval-1.json" -> 1
                key=lambda x: int(
                    x.split("_")[-1].split(".")[0].replace("-", "")
                )
            )
            return json_filenames[-1]
        else:
            return None
