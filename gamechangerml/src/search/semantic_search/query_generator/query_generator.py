from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from typing import List
from .query_generator_config import QueryGeneratorConfig


class QueryGenerator:
    """Generate relevant queries for passages using a synthetic query generation
    model.

    Args:
        base_model (str, optional): Model ID or path.
            Reference: [pretrained_model_name_or_path](https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/model#transformers.PreTrainedModel.save_pretrained).
    """

    def __init__(self, base_model=QueryGeneratorConfig.BASE_MODEL):
        self._tokenizer = T5Tokenizer.from_pretrained(base_model)
        self._model = T5ForConditionalGeneration.from_pretrained(base_model)

    def generate(
        self,
        text: str,
        max_queries: int,
        max_length_per_query: int = QueryGeneratorConfig.MAX_LENGTH,
        do_sample: bool = QueryGeneratorConfig.DO_SAMPLE,
        top_p: float = QueryGeneratorConfig.TOP_P,
    ) -> List[str]:
        """Generate relevant queries for a passage using a synthetic query
        generation model.

        Args:
            text (str): The passage to generate relevant queries for.
            max_queries (int): The maximum number of queries to generate.
            max_length_per_query (int, optional): The maximum length of each
                query. Defaults to QueryGeneratorConfig.MAX_LENGTH.
            do_sample (bool, optional): Whether or not the model should use a
                sampling strategy to generate the text; uses greedy decoding
                otherwise.
                If True, the model will use a sampling strategy to generate the
                text. This means that instead of always selecting the most
                likely next word at each step, the model will randomly select
                from the distribution of possible next words based on their
                probabilities. This can lead to more diverse and creative text,
                but may also produce less coherent and less relevant text.
                If False,  the model will use a greedy strategy to generate the
                text. This means that at each step, the model will always select
                the most likely next word based on the probabilities. This can
                lead to more coherent and relevant text, but may also produce
                repetitive and less diverse text.
                Defaults to QueryGeneratorConfig.DO_SAMPLE.
            top_p (float, optional): If set to float < 1, only the most probable
                tokens with probabilities that add up to this value or higher
                are kept for generation. Defaults to QueryGeneratorConfig.TOP_P.

        Returns:
            List[str]: Relevant queries for the passage.
        """
        input_ids = self._tokenizer.encode(text, return_tensors="pt")

        with torch.no_grad():
            encoded_outputs = self._model.generate(
                input_ids=input_ids,
                num_return_sequences=max_queries,
                max_length=max_length_per_query,
                do_sample=do_sample,
                top_p=top_p,
            )

        decoded_outputs = [
            self._tokenizer.decode(encoded_outputs[i])
            for i in range(len(encoded_outputs))
        ]
        decoded_outputs = self._clean_outputs(decoded_outputs)

        return decoded_outputs

    def _clean_outputs(self, output_texts):
        return [
            " ".join(
                query.replace("<pad>", "").replace("</s>", "").split()
            ).strip()
            for query in output_texts
        ]
