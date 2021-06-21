import spacy
from spacy.matcher import Matcher
from spacy.util import filter_spans

nlp = spacy.load('en_core_web_md')

sentence = 'The USD(I) shall: a. Develop policy and procedures to implement and manage the RD and FRD security program for access, dissemmination, classification, declassification and handling of RD and FRD information within the DoD in accordance with the Atomic Energy Act of 1954.'

# verb phrase patterns
pattern = [
    {'POS': 'VERB', 'OP': '?'},
    {'POS': 'ADV', 'OP': '*'},
    {'OP': '*'},
    {'POS': 'VERB', 'OP': '+'},
]

matcher = Matcher(nlp.vocab)
matcher.add("verb-phrases", None, pattern)
doc = nlp(sentence)
matches = matcher(doc)
spans_ = filter_spans([doc[start:end] for _, start, end in matches])
print("\n".join([s.orth_ for s in spans_]))
print()
for np in doc.noun_chunks:
    print(np.orth_)
