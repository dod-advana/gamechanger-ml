from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from collections import OrderedDict
import collections
import numpy as np
import time

## ADD TYPING AND COMMENTS

def to_list(tensor):
    try:
        result = tensor.detach().cpu().tolist()
    except:
        result = list(tensor)

    return result

def get_clean_text(tokens, tokenizer):
    text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(tokens))
    # Clean whitespace
    text = text.strip()
    text = " ".join(text.split())

    return text


def prediction_probabilities(predictions):
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    all_scores = [pred.start_logit + pred.end_logit for pred in predictions]
    return softmax(np.array(all_scores))


def preliminary_predictions(start_logits, end_logits, input_ids, nbest):
    
    # convert tensors to lists
    start_logits = to_list(start_logits)[0]
    end_logits = to_list(end_logits)[0]

    # sort our start and end logits from largest to smallest, keeping track of the index
    start_idx_and_logit = sorted(enumerate(start_logits), key=lambda x: x[1], reverse=True)
    end_idx_and_logit = sorted(enumerate(end_logits), key=lambda x: x[1], reverse=True)

    start_indexes = [idx for idx, logit in start_idx_and_logit[:nbest]]
    end_indexes = [idx for idx, logit in end_idx_and_logit[:nbest]]

    # question tokens are between the CLS token (101, at position 0) and first SEP (102) token
    question_indexes = [i + 1 for i, token in enumerate(input_ids[1 : input_ids.index(102)])]

    # keep track of all preliminary predictions
    PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", ["start_index", "end_index", "start_logit", "end_logit"]
    )
    prelim_preds = []
    for start_index in start_indexes:
        for end_index in end_indexes:
            # throw out invalid predictions
            if start_index in question_indexes:
                continue
            if end_index in question_indexes:
                continue
            if end_index < start_index:
                continue
            prelim_preds.append(
                PrelimPrediction(
                    start_index=start_index,
                    end_index=end_index,
                    start_logit=start_logits[start_index],
                    end_logit=end_logits[end_index],
                )
            )
    # sort prelim_preds in descending score order
    prelim_preds = sorted(prelim_preds, key=lambda x: (x.start_logit + x.end_logit), reverse=True)

    return prelim_preds

def best_predictions(prelim_preds, start_logits, end_logits, input_ids, nbest, tokenizer):
    # keep track of all best predictions
    start_logits = to_list(start_logits)[0]
    end_logits = to_list(end_logits)[0]

    # This will be the pool from which answer probabilities are computed
    BestPrediction = collections.namedtuple(
        "BestPrediction", ["text", "start_logit", "end_logit"]
    )
    nbest_predictions = []
    seen_predictions = []
    
    for pred in prelim_preds:
        if len(nbest_predictions) >= nbest:
            break
        if pred.start_index > 0:  # non-null answers have start_index > 0

            begin = pred.start_index
            end = pred.end_index + 1

            toks = input_ids[begin:end]
            text = get_clean_text(toks, tokenizer)

            # if this text has been seen already - skip it
            if text in seen_predictions:
                continue

            # flag text as being seen
            seen_predictions.append(text)

            # add this text to a pruned list of the top nbest predictions
            nbest_predictions.append(
                BestPrediction(
                    text=text, start_logit=pred.start_logit, end_logit=pred.end_logit
                )
            )

    # Add the null prediction
    nbest_predictions.append(
        BestPrediction(text="", start_logit=start_logits[0], end_logit=end_logits[0])
    )

    print(nbest_predictions)

    return nbest_predictions

def compute_score_difference(predictions):
    """ Assumes that the null answer is always the last prediction """
    score_null = predictions[-1].start_logit + predictions[-1].end_logit
    score_non_null = predictions[0].start_logit + predictions[0].end_logit
    return score_null - score_non_null

def one_passage_answers(start_logits, end_logits, input_ids, tokenizer, nbest=1, null_threshold=1.0):

    prelim_preds = preliminary_predictions(start_logits, end_logits, input_ids, nbest)
    nbest_preds = best_predictions(prelim_preds, start_logits, end_logits, input_ids, nbest, tokenizer)
    #probabilities = prediction_probabilities(nbest_preds)
    score_difference = compute_score_difference(nbest_preds)
    # if score difference > threshold, return the null answer
    if score_difference > null_threshold:
        print("NO ANSWER", score_difference)
        return "", score_difference
    else:
        print("nbest", nbest_preds[0].text, score_difference)
        return nbest_preds[0].text, score_difference

def sort_answers(answers):

    sorted_answers = sorted(answers, key=lambda x: x[1], reverse=False)
    print("SORTED ANSWERS", sorted_answers)
    app_answers = []
    for ans in sorted_answers:
        if ans[0].strip().lstrip()[:4] != "[CLS]":
            mydict = {}
            mydict['text'] = ans[0]
            #mydict['probability'] = ans[1]
            mydict['null_score_diff'] = ans[1]
            mydict['context'] = ans[2]
            app_answers.append(mydict)
    #app_answers = ""
    #for ans in sorted_answers:
    #    if ans[0].strip().lstrip()[:4] != "[CLS]":
    #        app_answers += ans[0] + " / "

    return app_answers

class DocumentReader:
    def __init__(
        self, pretrained_model_name_or_path="bert-large-uncased", use_gpu=False
    ):
        self.READER_PATH = pretrained_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.READER_PATH)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.READER_PATH)
        self.max_len = self.model.config.max_position_embeddings
        self.chunked = False
        self.use_gpu = use_gpu

        if use_gpu:
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                self.use_gpu = use_gpu
            else:
                self.use_gpu = False

    def tokenize(self, question, text, separate = True):
        """
        Takes the inputs of the QAReader class and creates a tokenized inputs attribute.

        Args:
            - question (str): The question to ask the QAReader
            - text (List[str]): A list of context paragraphs to feed the QAReader

        Returns:
            - 
        """
        if separate == True:
            text = text.split("\n\n")
            all_inputs = []
            context_flag = 0
            context_tracker = []
            for i in text:
                inputs = self.tokenizer.encode_plus(question, i, add_special_tokens=True, return_tensors="pt")
                input_ids = inputs["input_ids"].tolist()[0]
                print("LENGTH INPUT IDS: ", len(input_ids))
                if len(input_ids) > self.max_len:
                    inputs = self.chunkify(inputs)
                    self.chunked = True
                    all_inputs.extend(inputs)
                    context_tracker.extend([context_flag] * len(inputs))
                else:
                    all_inputs.append(inputs)
                    context_tracker.append(context_flag)
                context_flag += 1

        return all_inputs, context_tracker

    def chunkify(self, inputs):
        """
        Break up a long article into chunks that fit within the max token
        requirement for that Transformer model.

        Calls to BERT / RoBERTa / ALBERT require the following format:
        [CLS] question tokens [SEP] context tokens [SEP].
        """

        # create question mask based on token_type_ids
        # value is 0 for question tokens, 1 for context tokens
        qmask = inputs["token_type_ids"].lt(1)
        qt = torch.masked_select(inputs["input_ids"], qmask)
        chunk_size = self.max_len - qt.size()[0] - 1  # the "-1" accounts for
        # having to add an ending [SEP] token to the end

        # create a dict of dicts; each sub-dict mimics the structure of pre-chunked model input
        chunked_input = OrderedDict()
        for k, v in inputs.items():
            q = torch.masked_select(v, qmask)
            c = torch.masked_select(v, ~qmask)
            chunks = torch.split(c, chunk_size)

            for i, chunk in enumerate(chunks):
                if i not in chunked_input:
                    chunked_input[i] = {}

                thing = torch.cat((q, chunk))
                if i != len(chunks) - 1:
                    if k == "input_ids":
                        thing = torch.cat((thing, torch.tensor([102])))
                    else:
                        thing = torch.cat((thing, torch.tensor([1])))

                chunked_input[i][k] = torch.unsqueeze(thing, dim=0)
        return chunked_input

    def get_answer(self, inputs):

        if self.chunked:
            answer = ""
            for k, chunk in inputs.items():
                if self.use_gpu:
                    chunk = {key: value.cuda() for key, value in chunk.items()}
                answer_start_scores, answer_end_scores = self.model(**chunk)

                if self.use_gpu:
                    answer_start_scores = answer_start_scores.detach().cpu()
                    answer_end_scores = answer_end_scores.detach().cpu()
                    chunk = {key: value.detach().cpu() for key, value in chunk.items()}

                answer_start = torch.argmax(self.model(**chunk)["start_logits"])
                answer_end = torch.argmax(self.model(**chunk)["end_logits"]) + 1
         
                ans = self.convert_ids_to_string(
                    chunk["input_ids"][0][answer_start:answer_end]
                )
                if ans != "[CLS]":
                    answer += ans + " / "

            return answer
        else:
            if self.use_gpu:
                inputs = {key: value.cuda() for key, value in inputs}
            answer_start_scores, answer_end_scores = self.model(**inputs)

            if self.use_gpu:
                answer_start_scores = answer_start_scores.detach().cpu()
                answer_end_scores = answer_end_scores.detach().cpu()
                inputs = {key: value.detach().cpu() for key, value in inputs}

            # get the most likely beginning of answer with the argmax of the score
            answer_start = torch.argmax(answer_start_scores)
            # get the most likely end of answer with the argmax of the score
            answer_end = torch.argmax(answer_end_scores) + 1

            return self.convert_ids_to_string(
                inputs["input_ids"][0][answer_start:answer_end]
            )

    def convert_ids_to_string(self, input_ids):
        return self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(input_ids)
        )

    def get_robust_prediction(self, inputs, context_num, nbest=1, null_threshold=1.0):

        answers = []

        if self.chunked:
            print("CHUNKING INPUTS INTO {}".format(len(inputs.items())))
            for k, chunk in inputs.items():
                print(f"CHUNK {k}\n")
                if self.use_gpu:
                    chunk = {key: value.cuda() for key, value in chunk.items()}

                start_logits = self.model(**chunk)["start_logits"]
                end_logits = self.model(**chunk)["end_logits"]
                input_ids = chunk["input_ids"].tolist()[0]

                ans, diff = one_passage_answers(start_logits, end_logits, input_ids, self.tokenizer, nbest=5, null_threshold=1.0)
                answers.append((ans, diff, context_num))

        else:
            start_logits = self.model(**inputs)["start_logits"]
            end_logits = self.model(**inputs)["end_logits"]
            input_ids = inputs["input_ids"].tolist()[0]

            ans, diff = one_passage_answers(start_logits, end_logits, input_ids, self.tokenizer, nbest=5, null_threshold=1.0)
            answers.append((ans, diff, context_num))

        return answers

    def answer(self, question: str, context):

        print(f"Question: {question}")
        start = time.perf_counter()
        all_answers = []
        count = 0
        context_start = 0
        context_spans = []
        split = context.strip().lstrip().strip("\n\n").lstrip("\n\n").split("\n\n")
        for par in split:
            print(count)
            print(par)
            length_tokens = len(par.split(" "))
            print("LENGTH INPUT IDS: ", length_tokens)
            context_spans.append((context_start, context_start + length_tokens))
            context_start += length_tokens
            count += 1

        print("CONTEXT SPANS", context_spans)
        inputs, tracker = self.tokenize(question, context)
        count = 0
        for j in range(len(inputs)):
            print(inputs[j])
            print(tracker[j])
            answers = self.get_robust_prediction(inputs[j], tracker[j])
            all_answers.extend(answers)
        
        app_answers = sort_answers(all_answers)
        end = time.perf_counter()
        print("app_answers", app_answers)
        print(f"time: {end - start:0.4f} seconds")
        
        return app_answers