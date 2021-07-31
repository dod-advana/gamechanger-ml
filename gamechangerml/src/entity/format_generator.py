import logging

import gamechangerml.src.entity.entity_mentions as em
from gamechangerml.src.entity.ner_training_data import wc

logger = logging.getLogger(__name__)

I_PRFX = "I-"
B_PRFX = "B-"
OH = "O"
SENT = "sentence"


def gen_ner_conll_tags(abbrv_re, ent_re, entity2type, sent_list, nlp):
    """
    Generator to label text tokens according to the entity tags.

    Args:
        abbrv_re (SRE_Pattern): compiled regular expression

        ent_re  (SRE_Pattern): compiled regular expression

        entity2type (dict): map of an entity to its type, e.g.,
            GCORG, GCPER, etc; optional.

        sent_list (list): list of text(sentences) to label

        nlp (spacy.lang.en.English): spaCy language model

    Yields:
       zip(tokens, ner_labels), List[str] of unique labels

    """
    for row in sent_list:
        sentence_text = row[SENT]
        if not sentence_text.strip():
            continue

        doc = nlp(sentence_text)
        starts_ends = [(t.idx, t.idx + len(t.orth_) - 1) for t in doc]
        ner_labels = [OH] * len(starts_ends)
        tokens = [t.orth_ for t in doc]
        ent_spans = em.entities_spans(sentence_text, ent_re, abbrv_re)

        # find token indices of an extracted entity using their spans;
        # create CoNLL tags
        for ent, ent_st_end in ent_spans:
            token_idxs = [
                idx
                for idx, tkn_st_end in enumerate(starts_ends)
                if tkn_st_end[0] >= ent_st_end[0]
                and tkn_st_end[1] <= ent_st_end[1] - 1
            ]
            if not token_idxs:
                continue
            if wc(ent) == 1:
                ner_labels[token_idxs[0]] = I_PRFX + entity2type[ent.lower()]
                continue

            ner_labels[token_idxs[0]] = B_PRFX + entity2type[ent.lower()]
            for idx in token_idxs[1:]:
                if ent.lower() in entity2type:
                    ner_labels[idx] = I_PRFX + entity2type[ent.lower()]
                else:
                    logger.error("KeyError (why?): {}".format(ent.lower()))
        unique_labels = set(ner_labels)
        logger.debug([(t, s) for t, s in zip(tokens, ner_labels)])
        yield zip(tokens, ner_labels), unique_labels
