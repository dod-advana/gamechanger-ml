import logging

import pandas as pd

import gamechangerml.src.text_classif.utils.entity_lookup as el
from gamechangerml.src.text_classif.utils.raw_text2csv import raw2df

logger = logging.getLogger(__name__)


def find_serves_as(corpus_dir, glob, output_csv):
    df = pd.DataFrame(columns=["file_name", "content"])
    contains_entity = el.ContainsEntity()
    for sent_list, fname in raw2df(corpus_dir, glob):
        for item in sent_list:
            sent = item["sentence"]
            if contains_entity(sent):
                if "serves as" in sent or "shall serve as" in sent:
                    df = df.append(
                        {"file_name": fname, "content": sent},
                        ignore_index=True,
                    )
    df.to_csv(output_csv, index=False)
