import numpy as np

from gamechangerml.src.featurization.extract_improvement.extract_utils import (
    extract_entities,
    create_list_from_dict,
    remove_articles,
    match_parenthesis,
)


def get_agency(df, spacy_model):
    """
    Extract potential agencies from responsibilities text and cleans output.

    Args:
    df: Input responsibilities dataframe
    spacy_model: Spacy langauge model (typically 'en_core_web_lg')

    Returns:
    [List]
    """
    clean_df = df["agencies"].replace(np.nan, "", regex=True)
    all_docs = []

    for i in range(df.shape[0]):
        sentence = df["sentence"][i]
        entities = extract_entities(sentence, spacy_model)

        prev_agencies = [x.strip() for x in df["agencies"][i].split(",")]
        prev_agencies = [i for i in prev_agencies if i]

        flat_entities = create_list_from_dict(entities)
        for j in prev_agencies:
            flat_entities.append(j)

        flat_entities = remove_articles(flat_entities)
        flat_entities = match_parenthesis(flat_entities)
        flat_entities = "|".join(i for i in set(flat_entities))
        all_docs.append(flat_entities)

    df["agencies"] = all_docs
    return df

def entity_map(df, entity_mapping='entity_mapping.csv'):
    """
    Map known non-standard entities to standarized ones. entity_mapping is extensible and can be added to over time.

    Args:
    df:
    entity_mapping: input CSV of entity mappings to change

    Returns:
    [List]
    """
    entities = pd.read_csv('entity_mapping.csv')
    data = df

    output_list = []

    #TODO: Rewrite with Python mapping and/or ditionaries for better performance
    if 'Organization / Personnel' in data.columns:
        for i in range(data.shape[0]):
            for j in range(entities.shape[0]):
                if entities['old_entity'][j] in data['Organization / Personnel'][i]:
                    output_list.append(entities['new_entity'][j])
                else:
                    output_list.append(data['Organization / Personnel'][i])
        return output_list
    else:    
        return None

