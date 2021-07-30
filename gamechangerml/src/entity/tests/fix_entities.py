import pandas as pd
import os

from gamechangerml.src.entity.entity_mentions import entity_csv_to_df

LF = "long_form"
SF = "short_form"
ETYPE = "etype"
SENT = "sentence"

p = "/Users/chrisskiscim/projects/gamechanger-ml/gamechangerml/src/entity/aux_data"  # noqa
df = entity_csv_to_df(os.path.join(p, "entities.csv"))
print(df.head())
new_df = pd.DataFrame(columns=["entity", "entity_tag"])

for _, row in df.iterrows():
    new_row = {"entity": row[LF], "entity_tag": row[ETYPE]}
    new_df.append(new_row, ignore_index=True)
    if row[SF]:
        new_row = {"entity": row[SF], "entity_tag": row[ETYPE] + "-ABBRV"}
    new_df = new_df.append(new_row, ignore_index=True)
new_df = new_df.drop_duplicates()
new_df = new_df.sort_values(by="entity")
new_df.to_csv(
    os.path.join(p, "flat_entities.csv"), sep=",", index=False, header=False
)
