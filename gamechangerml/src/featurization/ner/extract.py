import numpy as np
from transformers import (
    RobertaTokenizer,
    AutoModelForTokenClassification,
    pipeline
)

class NERExtractor():

    def __init__(self, tuned_model_loc, base_model_name="gamechangerml/models/distilroberta-tokenizer", num_labels=5):

        self.model = AutoModelForTokenClassification.from_pretrained(tuned_model_loc, num_labels=num_labels)
        self.tokenizer = RobertaTokenizer.from_pretrained(base_model_name)
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple")
    
    def predict(self, string):
    
        results = self.nlp(string)
        hits = []
        char_count = 0
        idx = 0
        remaining = len(results)
        for _ in range(len(results)):
            if remaining > 0:
                idx = len(results) - remaining
                res = results[idx]
                start = char_count
                if res['entity_group'] == 'LABEL_1':
                    label = "ORG"
                    remaining -= 1
                    text = res['word']
                    score = [res['score']]
                    found_last = False
                    while found_last == False:
                        remainder = results[idx+1:]
                        for x in range(len(remainder)):
                            next_i = idx + x + 1
                            next_res = results[next_i]
                            if next_res['entity_group'] == 'LABEL_2':
                                text += next_res['word']
                                score.append(next_res['score'])
                                remaining -= 1
                            else:
                                found_last = True
                                break
                        break
                    char_count += len(text)
                    text = text.strip().lstrip()
                    avg_score = np.mean(score)
                    token_span = (start, char_count)
                    hits.append({"label": label, "start": token_span[0], "end": token_span[1], "text": text, "score": float(avg_score)})

                elif res['entity_group'] == 'LABEL_3':
                    label = "ROLE"
                    remaining -= 1
                    text = res['word']
                    score = [res['score']]
                    found_last = False
                    while found_last == False:
                        remainder = results[idx+1:]
                        for x in range(len(remainder)):
                            next_i = idx + x + 1
                            next_res = results[next_i]
                            if next_res['entity_group'] == 'LABEL_4':
                                text += next_res['word']
                                score.append(next_res['score'])
                                remaining -= 1
                            else:
                                found_last = True
                                break
                        break
                    char_count += len(text)
                    text = text.strip().lstrip()
                    avg_score = np.mean(score)
                    token_span = (start, char_count)
                    hits.append({"label": label, "start": token_span[0], "end": token_span[1], "text": text, "score": float(avg_score)})
                else:
                    remaining -= 1
                    text = res['word']
                    char_count += len(text)
                    text = text.strip().lstrip()
                    label = 'None'
        
        return hits
