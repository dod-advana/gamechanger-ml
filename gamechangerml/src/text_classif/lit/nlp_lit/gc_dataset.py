import logging
import os

import pandas as pd
import pandas.errors as pd_err
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types

logger = logging.getLogger(__name__)


class GCDataset(lit_dataset.Dataset):
    def __init__(self, data_path, num_labels):
        if not os.path.isfile(data_path):
            raise FileNotFoundError(data_path)

        self.LABELS = [str(lbl) for lbl in range(num_labels)]
        self._examples = list()

        try:
            df = pd.read_csv(
                data_path,
                delimiter=",",
                header=None,
                names=["src", "label", "sentence"],
            )
        except (pd_err.ParserError, pd_err.EmptyDataError) as e:
            raise e

        logger.info("rows : {:,d}".format(len(df)))

        self._examples = [
            {
                "sentence": row["sentence"],
                "label": row["label"],
                "src": row["src"],
            }
            for _, row in df.iterrows()
        ]

    def spec(self):
        return {
            "sentence": lit_types.TextSegment(),
            "label": lit_types.CategoryLabel(vocab=self.LABELS),
            "src": lit_types.TextSegment(),
        }
