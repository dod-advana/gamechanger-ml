import pandas as pd
from gamechangerml.src.paths import (
    COMBINED_ENTITIES_FILE,
    ORGS_FILE,
    TOPICS_FILE,
)

# simple script to combine agencies (orgs) and topics for ingestion
topics = pd.read_csv(TOPICS_FILE)
orgs = pd.read_csv(ORGS_FILE)
orgs.drop(columns=["Unnamed: 0"], inplace=True)
topics.rename(columns={"name": "entity_name",
              "type": "entity_type"}, inplace=True)
orgs.rename(columns={"Agency_Name": "entity_name"}, inplace=True)
orgs["entity_type"] = "org"

combined_ents = orgs.append(topics)
combined_ents.to_csv(COMBINED_ENTITIES_FILE, index=False)
