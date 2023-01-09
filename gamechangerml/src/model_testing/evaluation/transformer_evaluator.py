import torch
from .utils import LOCAL_TRANSFORMERS_DIR


class TransformerEvaluator:
    def __init__(self, transformer_path=LOCAL_TRANSFORMERS_DIR, use_gpu=False):

        self.transformer_path = transformer_path
        if use_gpu and torch.cuda.is_available():
            self.use_gpu = use_gpu
        else:
            self.use_gpu = False
