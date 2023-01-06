from sentence_transformers import InputExample
from pandas import DataFrame
from os.path import join
from gamechangerml.src.utilities import open_json
from gamechangerml.api.utils import processmanager
from gamechangerml.src.utilities.test_utils import timestamp_filename


class SemanticSearchTrainingData:
    """Create training data for SemanticSearchFinetuner.

    Args:
        data_directory (str): Path to directory containing a
            training_data.json file.
        logger (logging.Logger)
        test_mode (bool, optional): True to keep only 30 train and 10 test
            items. Defaults to False.

    Attributes:
        data_directory (str): data_directory (str): Path to directory containing
            a training_data.json file.
        logger (logging.Logger):
        total (int): The total number of train and test items.
        df (DataFrame): DataFrame of all train and test data. Contains the
            following columns: "key", "doc", "score", "label"
        csv_path (str): Path to csv file where `df` is saved.
        samples (InputExample[]): Train samples for finetuning.
    """

    def __init__(self, data_directory, logger, test_mode=False):
        self.data_directory = data_directory
        self.logger = logger

        self._load_path = join(self.data_directory, "training_data.json")
        self.csv_path = join(
            self.data_directory, timestamp_filename("finetuning_data", ".csv")
        )
        self._test_mode = test_mode

        self._load()
        self._format()
        self._save_csv()

    def _load(self):
        """Load the train and test data. If test mode only, reduce the train
        set to 30 items and test set to 10 items."""
        try:
            data = open_json(self._load_path)
            train_data = data["train"]
            test_data = data["test"]
        except:
            self.logger.exception(
                f"Failed to load semantic search training data: {self._load_path}"
            )

        if self._test_mode:
            count = 0
            train_tmp = {}
            for key in train_data.keys():
                if count < 30:
                    train_tmp[key] = train_data[key]
                else:
                    break
            train_data = train_tmp

            count = 0
            test_tmp = {}
            for key in test_data.keys():
                if count < 10:
                    test_tmp[key] = test_data[key]
                else:
                    break
            test_data = test_tmp

        self._train_data_file = train_data
        self._test_data_file = test_data
        self.total = len(self._train_data_file) + len(self._test_data_file)

    def _format(self):
        """Convert train data into InputExample items. Also create a list of
        all train and test items as tuples.
        """
        completed = 0
        input_examples = []
        all_data = []

        for data in self._train_data_file:
            query, paragraph, doc, score = self._get_train_fields(data)
            input_examples.append(
                InputExample(str(completed), (query, paragraph), score)
            )
            all_data.append(query, doc, score, "train")
            completed += 1
            self._update_process_manager_status(completed)

        for data in self._test_data_file:
            all_data.append(query, doc, score, "test")
            completed += 1
            self._update_process_manager_status(completed)

        self._all_data = all_data
        self.samples = input_examples

    def _save_csv(self):
        """Create a DataFrame with all train and test items, then save it as a
        csv.
        """
        self.df = DataFrame(
            self._all_data, columns=["key", "doc", "score", "label"]
        )
        self.df.drop_duplicates(subset=["doc", "score", "label"], inplace=True)
        self.logger.info(
            f"Generated training data CSV of {str(self.df.shape[0])} rows."
        )
        self.df.to_csv(self.csv_path)

    def _get_train_fields(self, data: dict):
        return (
            data["query"],
            data["paragraph"],
            data["doc"],
            float(data["label"]),
        )

    def _update_process_manager_status(self, completed):
        processmanager.update_status(
            processmanager.loading_data, completed, self.total
        )
