import logging
import gamechangerml.src.text_classif.utils.log_init as cu
import json

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    import pandas as pd

    cu.initialize_logger(to_file=False, log_name="none")
    abbrv_file = "/Users/chrisskiscim/projects/gamechanger-data/dataScience/src/featurization/data/abbreviations.json"  # noqa

    starting_txt = [
        "Secretary of",
        "Deputy Secretary of",
        "Assistant Secretary of",
        "Principal Deputy Assistant",
        "Deputy Assistant Secretary",
        "Under Secretary of",
        "Director of",
        "Directors of",
        "Director, ",
        "Directors, ",
        "Principal Deputy Director",
        "Chairman of",
        "Chairmen of",
        "Heads of the",
        "Component Heads",
        "Secretaries of the Military Departments",
    ]

    with open(abbrv_file) as f:
        abbrv_dict = json.load(f)

    scrubbed = dict()
    two_toks = list()
    two_tokens = list()
    df = pd.DataFrame(columns=["Abbreviation", "Expansion"])
    for abbrv, texts in abbrv_dict.items():
        check = [txt.startswith(st) for st in starting_txt for txt in texts]
        if True in check:
            scrubbed[abbrv] = texts
            row = {"Abbreviation": abbrv, "Expansion": " | ".join(texts)}
            df = df.append(row, ignore_index=True)

    # scrubbed_enc = json.dumps(scrubbed)
    df = pd.read_csv("entity.csv")
    alter = list()
    for _, row in df.iterrows():
        alter.append(row["Abbreviation"][:3])
    regex = "|".join(sorted(list(set(alter))))
    print(regex)
    print("|".join(sorted(list(set(starting_txt)))))
