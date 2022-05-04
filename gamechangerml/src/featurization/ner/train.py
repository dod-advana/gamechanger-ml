import os
import json
import pandas as pd
import re
import argparse
from collections import Counter
from transformers import (
    RobertaTokenizer,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
import random

class RemoteNERConfig:
    CORPUS_DIR = "/home/ec2-user/kdowdy/ner_corpus"
    GRAPH_DATA_PATH = "/home/ec2-user/model_training/GraphRelations.xls"
    ENTITIES_PATH = "/home/ec2-user/model_training/entities.json"
    TUNED_MODEL_LOC = "/home/ec2-user/model_training/distilroBERTa_checkpoints/checkpoint-final"
    BASE_MODEL_NAME = "distilroberta-base"
    OUTPUT_DIR = "/home/ec2-user/model_training/results03"

class DataConfig:
    SPLIT_PERC=0.2
    RARE_THRESHOLD=10
    TEST_IDS = []
    COMMON_ENTS = ['DoD', 'DOD', 'DoD Components', 'DoD Component', 'ASD', 'OSD', 'Office', 'Agency', 'United States']


class TrainConfig:
    TRAINING_ARGS = {
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "num_train_epochs": 20,
        "weight_decay": 0.01,
        "save_strategy":"epoch",
        "evaluation_strategy":"epoch"
    }

random.seed(42)

def open_json(filename, path):
    '''Opens a json file'''
    with open(os.path.join(path, filename)) as f:
        return json.load(f)

def collect_ents(df, name_type):
    '''Update a dictionary with names, aliases, parents and types from a df'''
    ents_dict = {}
    for i in df.index:
        name = df.loc[i, 'Name'].strip().lstrip()
        aliases = df.loc[i, 'Aliases']
        parent = df.loc[i, 'Parent'].strip().lstrip()
        if name != '':
            ents_dict[name] = name_type
        if aliases != '':
            aliases = [i.strip().lstrip() for i in aliases.split(';')]
        for x in aliases:
            if x != '':
                ents_dict[x] = name_type
        if parent != '':
            ents_dict[parent] = 'ORG'
    
    ents_dict = {k:ents_dict[k] for k in ents_dict.keys() if k!= ''}
    return ents_dict

def clean_df(df):
    '''Clean the df before getting entities'''
    df.dropna(subset = ["Name"], inplace = True)
    df['Parent'].fillna('', inplace = True)
    df['Aliases'].fillna('', inplace = True)
    
    return df

def make_entities_dict(
    entities_path, 
    must_include={'DoD': 'ORG', 'Department of Defense': 'ORG', 'DOD': 'ORG'}
    ):
    '''Makes dictionary of org/roles/aliases/parents and their types'''

    orgs = pd.read_excel(io=entities_path, sheet_name='Orgs')
    orgs = clean_df(orgs)
    orgs_dict = collect_ents(df=orgs, name_type='ORG')
    print(f"Gathered {len(orgs_dict)} orgs")

    roles = pd.read_excel(io=entities_path, sheet_name='Roles')
    roles = clean_df(roles)
    roles_dict = collect_ents(df=roles, name_type='ROLE')
    print(f"Gathered {len(roles_dict)} orgs")
    
    ents_dict = {**roles_dict, **orgs_dict}
    print(f"Shape of combined entities: {len(ents_dict)}")
    
    for x in must_include.keys():
        if x not in ents_dict.keys():
            print(f"Adding {x}")
            ents_dict[x] = must_include[x]
    
    print(f"Shape of final entities: {len(ents_dict)}")
    
    return ents_dict

def open_labeled_csv(pre_labeled_data_path):
    '''Open a CSV of pre-labeled data and convert strings back to lists'''

    df = pd.read_csv(pre_labeled_data_path)
    print(f"Shape of data: {df.shape}")
    df = df[df['tokens']!='[]']
    df = df[df['labels']!='[]']
    print(f"Shape of data: {df.shape}")
    df['tokens'] = df['tokens'].apply(lambda x: [i.strip("'").lstrip("'") for i in x.lstrip('[').strip(']').split(', ')])
    df['labels'] = df['labels'].apply(lambda x: [int(i) for i in x.lstrip('[').strip(']').split(', ')])   
    
    return df

## from gamechanger-ml/gamechangerml/src/utilities/text_utils
def translate_to_ascii_string(_s):
    """
    Translates utf-8 byte sequence to ASCII string
    The point is to approximately translate foreign characters rather than
    deleting them
    Args:
        _s (str|bytes: string to translate

    Returns:
        str

    Raises:
        UnicodeDecodeError if decoding fails

    """
    _str_bytes = _s if isinstance(_s, bytes) else _s.encode("utf-8", "ignore")
    return _str_bytes.decode("ascii", errors="ignore")

## from gamechanger-ml/gamechangerml/src/utilities/text_utils
def simple_clean(text):
    """
    Performs a simple text cleaning: removes newline characters, square and
    curly braces, insures `utf-8` encoding, and reduces inter-word spacing to
    a single space.

    Args:
        text (str): text to be cleaned

    Returns:
        str

    Raises:
        UnicodeDecodeError if an illegal Unicode code-point is encountered

    """
    try:
        text = re.sub("[\\n\\t\\r]+", " ", text)
        text = re.sub("[" + re.escape("][}{)—\\/") + "]+", " ", text)
        text = re.sub("\\s{2,}", " ", text)
        text = translate_to_ascii_string(text)
        return text.strip()
    except UnicodeDecodeError as e:
        print("{}: {}".format(type(e), str(e)))
        raise

def get_longest(ents, verbose = False):
    '''Get the longest entity spans in text (remove shortest overlapping)'''
    
    ents.sort(key=lambda x: x[0], reverse=True) # sort by last (start of span)
    init_len = len(ents)
    remove = []
    full_range = len(ents) - 1
    for i in range(full_range): # for each entity, compare against remaining entities
        remainder = range(full_range - i)
        n = i + 1
        for x in remainder:
            if ents[i][0] < ents[n][1]:
                if verbose:
                    print(f"\nFound overlapping entity: {ents[i]} / {ents[n]}")
                if len(ents[i][3].split()) > len(ents[n][3].split()): # remove the shortest
                    if verbose:
                        print(f"Removing {ents[n]}")
                    remove.append(n)
                else:
                    if verbose:
                        print(f"Removing {ents[i]}")
                    remove.append(i)
            n += 1
       
    remove = list(set(remove))
    remove.sort()
    for x in remove[::-1]:
        try:
            ents.remove(ents[x])
        except:
            print(f"Could not remove {x} / {full_range}")

    if verbose:
        if len(ents) < init_len:
            print(f"Starting length: {init_len}, new length: {len(ents)}")
            print(f"Remaining ents: {ents}")

    return ents

def get_tokenized_spans(text):
    '''Make list of text, character spans for each token'''
    s = 0
    e = 0
    spans = []
    for i in range(len(text)):
        if text[i] != ' ':
            e += 1
        else:
            span = (s, e)
            spans.append({"text": text[s:e], "span": span})
            e += 1
            s = e
            
    return spans

def get_ents(text, entities, verbose = True):
    '''Lookup entities in single page of text, remove overlapping (shorter) spans'''
    
    ents = []
    for x in entities.keys():
        try:
            pattern = r"\b{}\b".format(x)
            for match in re.finditer(pattern, text):
                tup = (match.start(), match.end(), entities[x], text[match.start():match.end()])
                if tup[3] != '':
                    ents.append(tup)
        except Exception as e:
            print(e)
    if ents != []:
        ents = get_longest(ents, verbose) # remove overlapping spans
    
    ents.sort(key=lambda x: x[0], reverse=False)
    
    return ents

def check_entities(i, entities):
    '''Backup: check if single token is in entities dictionary'''
    
    if i in entities.keys():
        return entities[i]
    else:
        return None
    
def make_fake_ents(text, ents): 
    '''Makes two copies of text: one with beginnings/one with endings removed'''
    new_text_end = new_text_beginning = text
    keep_ents = []
    for i in ents[::-1]:
        e_text = i[3]
        start = i[0]
        end = i[1]
        if " of " in e_text:
            tokens = e_text.split(' of ')
            no_end = tokens[0] + ' of'
            no_beginning = 'of ' + tokens[1]
            new_text_end = new_text_end[:start] + no_end + new_text_end[end:]
            new_text_beginning = new_text_beginning[:start] + no_beginning + new_text_beginning[end:]
        elif " for " in e_text:
            tokens = e_text.split(' for ')
            no_end = tokens[0] + ' for'
            no_beginning = 'for ' + tokens[1]
            new_text_end = new_text_end[:start] + no_end + new_text_end[end:]
            new_text_beginning = new_text_beginning[:start] + no_beginning + new_text_beginning[end:]
        elif ", " in e_text:
            tokens = e_text.split(', ')
            no_end = tokens[0] + ', '
            no_beginning = tokens[1]
            new_text_end = new_text_end[:start] + no_end + new_text_end[end:]
            new_text_beginning = new_text_beginning[:start] + no_beginning + new_text_beginning[end:]
        elif ' ' in e_text:
            tokens = e_text.split(' ')
            no_end = ' '.join(tokens[:-1])
            no_beginning = ' '.join(tokens[1:])
            new_text_end = new_text_end[:start] + no_end + new_text_end[end:]
            new_text_beginning = new_text_beginning[:start] + no_beginning + new_text_beginning[end:]
        else:
            keep_ents.append(i)
    
    return new_text_end, new_text_beginning, keep_ents

def check_common_ents(ents, common_ents=DataConfig.COMMON_ENTS):

    if ents != []:
        return [i for i in ents if i[3] not in common_ents]
    else:
        return []

def expand_samples(p, entities, clean_text=True, verbose=False):

    expanded = []
    text = p['p_raw_text']
    pid = p['id']
    print(f"Page: {pid}")
    if clean_text:
        text = simple_clean(text)
    
    spans = get_tokenized_spans(text)
    text = ' '.join([i['text'] for i in spans])
    ents = get_ents(text, entities, verbose)

    if check_common_ents(ents) != []: # check there are entities/not only common entities in sample
        expanded.append({"pid": pid, "text": text, "entities": ents})

        # make fake ents - if there are any, randomly add either partial beginning/end to the expanded samples
        new_text_end, new_text_beginning, keep_ents = make_fake_ents(text, ents)
        if keep_ents != ents:
            expanded.append({
                "pid": pid + '_FAKE', "text": random.choice([new_text_beginning, new_text_end]) , "entities": keep_ents
                })
    else:
        print("Did not find entities (or uncommon entities) in this sample")

    return expanded

def label_strings(text, ents):
    '''Make labels for whether a token is part of an entity:
        0 = Not part of an entity
        1 = ORG (Beginning)
        2 = ORG (Middle/End)
        3 = ROLE (Beginning)
        4 = ROLE (Middle/End)
    '''
    
    spans = get_tokenized_spans(text)
    text = ' '.join([i['text'] for i in spans])
    
    strings = []
    for i in spans:
        label = 0
        s = i['span'][0]
        if i['text'] == '':
            label = 0
        else:
            for ent in ents:
                if s == ent[0]: # if start position matches
                    if ent[2] == 'ORG':
                        label = 1
                    elif ent[2] == 'ROLE':
                        label = 3
                elif int(ent[0]) < s < int(ent[1]): # if any overlap in span
                    if ent[2] == 'ORG':
                        label = 2
                    elif ent[2] == 'ROLE':
                        label = 4 
        mytup = (i['text'], i['span'], label)
        strings.append(mytup)
    
    return strings

def collect_entities(json_list, corpus_dir, entities, output_filename):
    '''Looks up entities for each JSON and creates a CSV file / dataframe '''
    
    records = []

    with open(output_filename,'a') as f:
        total = len(json_list)
        doc_count = 1
        for i in json_list:
            print(f"\nLabeling {doc_count} / {total}: {i}")
            doc_count += 1
            try:
                doc = open_json(i, corpus_dir)
                pages = doc['pages']
                random.shuffle(pages)
                page_count = 0
                for p in pages:
                    if page_count < 20:
                        samples = expand_samples(p, entities, clean_text=True)
                        if samples != []:
                            page_count += 1
                            for s in samples:
                                s['text']
                                mylist = label_strings(s['text'], s['entities'])
                                tokens = [i[0] for i in mylist]
                                labels = [i[2] for i in mylist]
                                records.append({"pid": s['pid'], "text": s['text'], "tokens": tokens, "labels": labels, "entities": s['entities']})
                                f.write(json.dumps({"pid": s['pid'], "text": s['text'], "tokens": tokens, "labels": labels, "entities": s['entities']}))
                                f.write("\u000a")

                del doc

            except Exception as e:
                print("Error retrieving entities")
                print(e)
        
    return records

def get_entity_counts_df(df):
    '''Get entity/type counts from dataframe of extracted entities by paragraph'''
    
    roles = []
    orgs = []
    for i in df.index:
        try:
            ents = df.loc[i, 'dict_ents']
            roles.extend(ents['roles'])
            orgs.extend(ents['orgs'])
        except Exception as e:
            print(e)

    role_counts = pd.DataFrame([dict(Counter(roles)).keys(), dict(Counter(roles)).values()]).T
    role_counts['type'] = 'ROLE'
    org_counts = pd.DataFrame([dict(Counter(orgs)).keys(), dict(Counter(orgs)).values()]).T
    org_counts['type'] = 'ORG'
    entity_counts = pd.concat([role_counts, org_counts])
    entity_counts = entity_counts.sort_values(by=[1], ascending=False)
    
    entity_counts.rename(columns = {0: 'entity', 1: 'count'}, inplace = True)
    
    return entity_counts

def get_rare_entities(counts_df, threshold):
    
    return counts_df[counts_df['count'] <= threshold].copy()

def get_ent_dict(x):
    
    try:
        orgs = list(set([i[3] for i in x if i[2]=='ORG']))
        roles = list(set([i[3] for i in x if i[2]=='ROLE']))
        orgs = [k for k in orgs if k != '']
        roles = [k for k in roles if k != '']
        return {"orgs": orgs, "roles": roles}
    except:
        return {"orgs": [], "roles": []}

def get_word_ids(tokens):
    '''Ethan's function to get word IDs for re-aligning labels'''
    word_ids = []
    w_id = -1
    for i, token in enumerate(tokens):
        if token == '<s>' or token == '</s>':
            word_ids.append(None)
            w_id += 1
        elif(i == 1):
            word_ids.append(w_id)
        elif('Ġ' in token):
            w_id += 1
            word_ids.append(w_id)
        else:
            word_ids.append(w_id)
    return word_ids

def tokenize_and_align_labels(example, tokenizer):
    '''Ethan's function to tokenize for distilRoberta and re-align labels'''
    try:
        tokenized_inputs = tokenizer(example["tokens"], truncation=True, is_split_into_words=True)
        labels = example["labels"]
        tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"])
        word_ids = get_word_ids(tokens)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(labels[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        tokenized_inputs["labels"] = label_ids
        return {'input_ids': tokenized_inputs['input_ids'], 'labels': label_ids}
    except:
        print(f"Error tokenizing and aligning {example}")
        return {'input_ids': [''], 'labels': [0]}

def tokenize(tokenizer, df):
    '''Ethan's function to tokenize'''
    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(lambda examples: tokenize_and_align_labels(examples, tokenizer))

    return tokenized_dataset

def read_in_tokens_csv(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Shape of data: {df.shape}")
    df = df[df['tokens']!='[]']
    df = df[df['labels']!='[]']
    print(f"Shape of data: {df.shape}")
    df['tokens'] = df['tokens'].apply(lambda x: [i.strip("'").lstrip("'") for i in x.lstrip('[').strip(']').split(', ')])
    df['labels'] = df['labels'].apply(lambda x: [int(i) for i in x.lstrip('[').strip(']').split(', ')])  

    df = df[['tokens', 'labels']]
    
    return df

def get_holdout_ents(df, output_dir, rare_threshold):

    counts_df = get_entity_counts_df(df)
    counts_df.to_csv(os.path.join(output_dir, "entity_ner_counts.csv"))
    holdout_df = get_rare_entities(counts_df, rare_threshold)
    holdout_df.to_csv(os.path.join(output_dir,"holdout_entities_ner.csv"))
    holdout_ents = list(set(holdout_df['entity']))

    return holdout_ents


class NERTrainer():

    def __init__(self, entities_path, graph_data_path):

        self.entities = self.get_entities_dictionary(entities_path, graph_data_path)
    
    def get_entities_dictionary(self, entities_path, graph_data_path):

        if not os.path.isfile(entities_path):
            print("\nMaking entities JSON file ... ")
            entities = make_entities_dict(graph_data_path)
            print(f"Saving file to {entities_path}")
            with open(entities_path, "w") as outfile: 
                json.dump(entities, outfile)
    
        else:
            print("\nOpening entities JSON file ...")
            with open(entities_path) as f:
                entities = json.load(f)
        return entities
    
    def make_data(self, corpus_dir, test_size, output_dir):
        '''Function to either load in pre-labeled data or make labeled data'''
        
        training_jsons = [i for i in os.listdir(corpus_dir) if os.path.isfile(os.path.join(corpus_dir, i))]
        if test_size:
            training_jsons = training_jsons[:test_size]
            print(f"\nSetting test size to {test_size}")
    
        print("\nLabeling entities ... ")
        training_file_path = os.path.join(output_dir, "ner_training_data.json")
        data = collect_entities(training_jsons, corpus_dir, self.entities, training_file_path)
        df = pd.DataFrame(data)
        df['dict_ents'] = df['entities'].apply(lambda x: get_ent_dict(x))
    
        return df

    def train_test_split(self, df, split_perc, rare_threshold, test_ids, output_dir, holdout_ents=[]):
        
        if holdout_ents==[]:
            holdout_ents = get_holdout_ents(df, output_dir, rare_threshold)
        if not test_ids: # if we don't already have a list of ids to split on
            test_ids = []
            test_size = split_perc * df.shape[0]
            no_fake = df[~df['pid'].str.contains('_FAKE')].copy()
            # first, split on rare IDs
            for i in holdout_ents:
                for x in no_fake.index:
                    if len(test_ids) < 0.6 * test_size:
                        if i in [f[0] for f in no_fake.loc[x, 'dict_ents']['orgs']]:
                            test_ids.append(no_fake.loc[x, 'pid'])
                        elif i in [f[0] for f in no_fake.loc[x, 'dict_ents']['roles']]:
                            test_ids.append(no_fake.loc[x, 'pid'])
                    else:
                        break
            remaining_ids = [x for x in no_fake['pid'].tolist() if x not in test_ids]
            remainder = int((test_size - len(test_ids)) / 2)
            to_add = random.sample(remaining_ids, remainder)
            test_ids = test_ids + to_add
            test_fakes = [i for i in df['pid'].tolist() if i.split('_FAKE')[0] in test_ids]
            test_ids = test_ids + test_fakes

        # split 
        test_df = df[df['pid'].isin(test_ids)]
        train_df = df[~df['pid'].isin(test_ids)]

        ## split test into test/val
        val_train_size = round(test_df.shape[0] / 2) #split validation into 50-50
        val_df = test_df.sample(val_train_size)
        test_df = test_df[~test_df.index.isin(val_df.index)]

        print("\nSaving separate dataframes")
        train_df.to_csv(os.path.join(output_dir, "ner_train_data.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "ner_test_data.csv"), index=False)
        val_df.to_csv(os.path.join(output_dir, "ner_val_data.csv"), index=False)

        train_df.drop(columns = ['pid', 'text', 'entities'], inplace = True)
        test_df.drop(columns = ['pid', 'text', 'entities'], inplace = True)
        val_df.drop(columns = ['pid', 'text', 'entities'], inplace = True)

        print(f"Train size: {train_df.shape[0]}")
        print(f"Test size: {test_df.shape[0]}")
        print(f"Validation size: {val_df.shape[0]}")

        return train_df, test_df, val_df

    def train(self, tuned_model_loc, train_df, val_df, output_dir, base_model_name):
        '''Ethan's code to train distilRoberta'''

        tokenizer = RobertaTokenizer.from_pretrained(base_model_name)

        if not os.path.exists(output_dir):
            print(f"Making output directory: {output_dir}")
            os.makedirs(output_dir)
        
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        model = AutoModelForTokenClassification.from_pretrained(tuned_model_loc, num_labels=5)

        train_ds = tokenize(tokenizer, train_df)
        val_ds = tokenize(tokenizer, val_df)

        training_args = TrainingArguments(
            output_dir=output_dir,
            **TrainConfig.TRAINING_ARGS
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        trainer.train()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train NER")

    parser.add_argument("--job", "-j", dest="job", help="which process to run - options = ['make_data', 'train']")
    parser.add_argument("--env", "-e", dest="env", required=False, help="['local' or 'remote']")
    parser.add_argument("--test-size", "-t", dest="test_size", required=False, type=int, help="number of JSONs to test on (if None, run pipeline on all)")
    parser.add_argument("--pre-labeled-data", "-p", dest="pre_labeled_data", required=False, help="path to prelabeled data csv (if None, makes the CSV)")
    
    args = parser.parse_args()
    BaseConfig = RemoteNERConfig
    test_size = args.test_size if args.test_size else 10000
    
    if args.job == 'make_data':
        print("\nStarting NER Pipeline ... ")
        Ner = NERTrainer(BaseConfig.ENTITIES_PATH, BaseConfig.GRAPH_DATA_PATH)

        # make data
        print("\nCollecting training data ... ")
        df = Ner.make_data(BaseConfig.CORPUS_DIR, test_size, BaseConfig.OUTPUT_DIR)

        print("\nTrain/test split ... ")
        train_df, test_df, val_df = Ner.train_test_split(
            df, DataConfig.SPLIT_PERC, DataConfig.RARE_THRESHOLD, DataConfig.TEST_IDS, BaseConfig.OUTPUT_DIR)
        
        print("\nDone making & splitting training data")
    
    if args.job == "split":
        print("\nStarting NER Pipeline ... ")
        Ner = NERTrainer(BaseConfig.ENTITIES_PATH, BaseConfig.GRAPH_DATA_PATH)

        print("Train test split ...")
        training_data_path = os.path.join(BaseConfig.OUTPUT_DIR, "ner_training_data.json")
        data = []
        with open(training_data_path) as f:
            for line in f:
                data.append(json.loads(line))
        df = pd.DataFrame(data)
        df['dict_ents'] = df['entities'].apply(lambda x: get_ent_dict(x))

        try:
            holdout_df = pd.read_csv(os.path.join(BaseConfig.OUTPUT_DIR, "holdout_entities_ner.csv"))
            holdout_ents = list(set(holdout_df['entity']))
            print("loaded holdout df")
        except:
            print("no holdout df to read in")
            holdout_df = []
        Ner.train_test_split(df, DataConfig.SPLIT_PERC, DataConfig.RARE_THRESHOLD, test_ids=[], output_dir=BaseConfig.OUTPUT_DIR, holdout_ents=holdout_ents)

    if args.job == 'train':
        print("\nStarting NER Pipeline ... ")
        Ner = NERTrainer(BaseConfig.ENTITIES_PATH, BaseConfig.GRAPH_DATA_PATH)

        print("\nGetting data ... ")
        train_df = read_in_tokens_csv(os.path.join(BaseConfig.OUTPUT_DIR, "ner_train_data.csv"))
        val_df = read_in_tokens_csv(os.path.join(BaseConfig.OUTPUT_DIR, "ner_val_data.csv"))
        test_df = read_in_tokens_csv(os.path.join(BaseConfig.OUTPUT_DIR, "ner_test_data.csv"))
        print("\nTraining ...")
        Ner.train(BaseConfig.TUNED_MODEL_LOC, train_df, val_df, BaseConfig.OUTPUT_DIR, BaseConfig.BASE_MODEL_NAME)
