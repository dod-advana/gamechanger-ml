import logging
import re

import gamechangerml.src.text_classif.utils.log_init as cu

logger = logging.getLogger(__name__)

abbrv_file = "/Users/chrisskiscim/projects/gamechanger-ml/gamechangerml/src/text_classif/utils/updated_dod_orgs.txt"  # noqa

if __name__ == "__main__":
    import pandas as pd

    cu.initialize_logger(to_file=False, log_name="none")

    with open(abbrv_file) as f:
        entity_list = f.readlines()

    scrubbed = dict()
    abbrevs = set()
    two_tokens = set()
    df = pd.DataFrame(columns=["Abbreviation", "Expansion"])
    for line in entity_list:
        if line.startswith("#"):
            continue
        line = line.strip()
        if "(" in line:
            entity, abbrv = line.split("(", maxsplit=1)
        else:
            entity = line
            abbrv = None
        two = entity.split()
        two_tokens.add(" ".join(two[:2]))
        if abbrv and abbrv.endswith(")"):
            abbrevs.add(abbrv[:-1])

    final_ent = sorted(list(two_tokens))
    final_abrv = sorted(list(abbrevs))
    final_abrv = [re.sub("[)(]", "", s) for s in final_abrv]
    final_abrv = [re.sub("-", " ", s) for s in final_abrv]
    final_abrv = [re.sub(r"&", "#", s) for s in final_abrv]

    print("|".join(final_abrv))
    print("|".join(final_ent))
