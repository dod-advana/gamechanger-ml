import logging
import pandas as pd

import gamechangerml.src.text_classif.utils.entity_lookup as el
from gamechangerml.src.text_classif.utils.log_init import initialize_logger
from gamechangerml.src.text_classif.utils.raw_text2csv import raw2df

logger = logging.getLogger(__name__)


def find_serves_as(corpus_dir, glob, output_csv):
    df = pd.DataFrame(columns=["file_name", "content"])
    abrv_re, org_re = el.build_entity_lookup()
    for sent_list, fname in raw2df(corpus_dir, glob):
        for item in sent_list:
            sent = item["sentence"]
            if el.contains_entity(sent, abrv_re, org_re):
                if "serves as" in sent or "shall serve as" in sent:
                    df = df.append(
                        {"file_name": fname, "content": sent},
                        ignore_index=True,
                    )
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    initialize_logger(to_file=False, log_name="none")
    corpus_dir_ = "/Users/chrisskiscim/projects/json_corpus_20210419"
    glob_ = "DoDI*.json"
    output_csv_ = "dodi_serves_as.csv"
    find_serves_as(corpus_dir_, glob_, output_csv_)
