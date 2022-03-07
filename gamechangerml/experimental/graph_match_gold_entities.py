####
# EDA script for comparing "gold" standard entities with new ones generated from NER model which produces as csv determining
# which new entities match "gold" standard entities
####

import pandas as pd
import re
import string
import nltk
nltk.download('omw-1.4')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

### read in data - UPDATE pathing based on where csvs are located
gold_orgs = pd.read_excel("~/Documents/GraphRelations.xlsx","Orgs")
gold_roles = pd.read_excel("~/Documents/GraphRelations.xlsx","Roles")
ner_entities = pd.read_csv("~/Documents/ents_20220225.csv")


# Create WordNetLemmatizer object
wnl = WordNetLemmatizer()
def clean_ent(ent):
    ent = re.sub(r'[0-9]+', '', ent) # remove numbers
    ent = re.sub(r'[^\w\s]', ' ', ent) # replace punctuation with spaces
    ent = re.sub(r'[^a-zA-Z\s]',"",ent) # totally strip non-alphabetical chars
    ent = ent.lower().strip().split()
    ent = " ".join(ent)
    return ent

def lemmatize(ent):
    ent_tokens = ent.split(" ")
    lemma_ent = []
    for ent_token in ent_tokens:
        lemma_ent.append(wnl.lemmatize(ent_token))
    return " ".join(lemma_ent)

def normalize_ents(ent):
    # normalize U.S. to United
    ent = ent.replace("U.S.","United States")
    # expand `,` to normalized " of" - e.g., Director, <> -> Director of <>
    ent = ent.replace(",", " of")
    # Normalize `of the` to `of` - e.g., Director of the <> -> Director of <>
    ent = ent.replace("of the", "of")
    ent = ent.replace("&"," and ")
    return ent


## drop unneeded rows
ner_entities = ner_entities[ner_entities["To Add"]=="y"].reset_index(drop=True)

# clean and lemmatize the ner entities=
ner_entities['lemma_entity'] = list(map(lambda ent:lemmatize(clean_ent(normalize_ents(ent))),ner_entities['entity']))

### pull in gold standard
gold_entities = pd.DataFrame({"entity":list(gold_orgs['Name'])+list(gold_roles['Name']),
                              "aliases":list(gold_orgs['Aliases'])+list(gold_roles['Aliases'])})
gold_entities['aliases'] = list(map(lambda alias: "" if pd.isna(alias) else alias.split(";"),gold_entities['aliases']))
all_gold_entities = list(gold_entities['entity'])+[alias for alias_list in gold_entities['aliases'] for alias in alias_list]
all_gold_entities = list(map(lambda ent:lemmatize(clean_ent(normalize_ents(ent))),all_gold_entities))


unique_ner_entities = list(ner_entities['lemma_entity'].unique())
unfound_ner_ents = [ent for ent in unique_ner_entities if ent not in all_gold_entities]

ner_entities["occur_counter"] = list(map(lambda ent: list(ner_entities['lemma_entity']).count(ent),ner_entities['lemma_entity']))
ner_entities["not_found_in_gold"] = list(map(lambda ent: 1 if ent not in all_gold_entities else 0,ner_entities['lemma_entity']))


## output results
ner_entities[["entity","lemma_entity","occur_counter","not_found_in_gold"]].sort_values("lemma_entity").to_csv("~/Documents/ents_20220225_results.csv",index=False)