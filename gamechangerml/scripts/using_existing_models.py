from gamechangerml.src.search.semantic.models import D2V
from gamechangerml.src.text_handling.entity import Phrase_Detector
from gamechangerml.src.text_handling.process import preprocess
from gamechangerml import REPO_PATH
import os

model_dir = os.path.join(
    REPO_PATH,
    "gamechangerml/src/modelzoo/semantic/models"
)
model_name = "2020072720_model.d2v"


phrase_detector = Phrase_Detector("id")
phrase_detector.load(model_dir)

model = D2V("id")
model.load(f"{model_dir}/{model_name}")

tokens = preprocess(
    "National Park",
    min_len=1,
    phrase_detector=phrase_detector,
    remove_stopwords=True,
)
print(model.infer(tokens))

tokens = preprocess(
    "National Parks",
    min_len=1,
    phrase_detector=phrase_detector,
    remove_stopwords=True,
)
print(model.infer(tokens))

tokens = preprocess(
    "taxes", min_len=1, phrase_detector=phrase_detector, remove_stopwords=True
)
print(model.infer(tokens))
