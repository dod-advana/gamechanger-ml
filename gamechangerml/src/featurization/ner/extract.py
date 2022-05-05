import numpy as np
from transformers import RobertaTokenizer, AutoModelForTokenClassification, pipeline


class NERExtractor:
    def __init__(self, model_name: str, tokenizer: str, num_labels: int = 5):

        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer)
        self.nlp = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
        )

    def predict(self, string: str, simple_results: bool, min_score: float):
        """Returns extracted entities from a paragraph/string
        Args:
            string [str]: paragraph/text
            simple_results [bool]: whether to return just unique results by type
            min_score [float]: cutoff confidence score for returning entities
        Returns:
            entities [dict]: list of dictionaries (simple_results=False) or dictionary of entities by type
        """
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
                if res["entity_group"] == "LABEL_1":
                    label = "ORG"
                    remaining -= 1
                    text = res["word"]
                    score = [res["score"]]
                    found_last = False
                    while found_last == False:
                        remainder = results[idx + 1 :]
                        for x in range(len(remainder)):
                            next_i = idx + x + 1
                            next_res = results[next_i]
                            if next_res["entity_group"] == "LABEL_2":
                                text += next_res["word"]
                                score.append(next_res["score"])
                                remaining -= 1
                            else:
                                found_last = True
                                break
                        break
                    char_count += len(text)
                    text = text.strip().lstrip()
                    avg_score = np.mean(score)
                    token_span = (start, char_count)
                    hits.append(
                        {
                            "label": label,
                            "start": token_span[0],
                            "end": token_span[1],
                            "text": text,
                            "score": float(avg_score),
                        }
                    )

                elif res["entity_group"] == "LABEL_3":
                    label = "ROLE"
                    remaining -= 1
                    text = res["word"]
                    score = [res["score"]]
                    found_last = False
                    while found_last == False:
                        remainder = results[idx + 1 :]
                        for x in range(len(remainder)):
                            next_i = idx + x + 1
                            next_res = results[next_i]
                            if next_res["entity_group"] == "LABEL_4":
                                text += next_res["word"]
                                score.append(next_res["score"])
                                remaining -= 1
                            else:
                                found_last = True
                                break
                        break
                    char_count += len(text)
                    text = text.strip().lstrip()
                    avg_score = np.mean(score)
                    token_span = (start, char_count)
                    hits.append(
                        {
                            "label": label,
                            "start": token_span[0],
                            "end": token_span[1],
                            "text": text,
                            "score": float(avg_score),
                        }
                    )
                else:
                    remaining -= 1
                    text = res["word"]
                    char_count += len(text)
                    text = text.strip().lstrip()
                    label = "None"

        hits = [i for i in hits if i["score"] >= min_score]

        if simple_results:
            orgs = list(set([i["text"] for i in hits if i["label"] == "ORG"]))
            roles = list(set([i["text"] for i in hits if i["label"] == "ROLE"]))
            hits = {"ORGS": orgs, "ROLES": roles}

        return hits
