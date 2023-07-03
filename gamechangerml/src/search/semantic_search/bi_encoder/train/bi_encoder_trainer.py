from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    models,
    losses,
    models,
    datasets,
    evaluation,
)
from sys import exit
import torch
from typing import List
import json
from datetime import datetime
from os import makedirs, path, environ
from gamechangerml.src.utilities import configure_logger, Timer
from ..bi_encoder_config import BiEncoderConfig


class BiEncoderTrainer:
    """Train a SentenceTransformer bi-encoder model.

    Args:
        use_gpu (bool): True to use GPU, False otherwise.
        base_model (str, optional): Model name (https://huggingface.co/models)
            or path. Defaults to BiEncoderConfig.BASE_MODEL.
        distance_metric (losses.TripletDistanceMetric, optional): The distance
            metric to use for calculating loss. Given a triplet of
            (anchor, positive, negative), the loss minimizes the distance
            between anchor and positive while it maximizes the distance between
            anchor and negative. Defaults to BiEncoderConfig.DISTANCE_METRIC.
        logger (logging.Logger or None): If None, creates a logger using
            configure_logger() from gamechangerml.src.utilities.
    """

    def __init__(
        self,
        use_gpu,
        base_model=BiEncoderConfig.BASE_MODEL,
        distance_metric: losses.TripletDistanceMetric = BiEncoderConfig.DISTANCE_METRIC,
        logger=None,
    ):
        self._base_model = base_model
        self._distance_metric = distance_metric
        self._distance_function = self._transform_distance_metric(
            distance_metric
        )
        self._logger = logger if logger else configure_logger()
        self._model = self._create_model(base_model, use_gpu)

    def _create_model(self, base_model, use_gpu):
        self._logger.info(
            f"BiEncoderTrainer - Creating Sentence Transformer with base model: {base_model}"
        )

        # Contextualized word embeddings for all input tokens.
        word_embedding = models.Transformer(base_model)

        # The embeddings go through a pooling layer to get a single fixed-length
        # embedding for all the text
        pooling = models.Pooling(word_embedding.get_word_embedding_dimension())

        device = "cpu"
        if use_gpu:
            if torch.cuda.is_available():
                device = "cuda"
                # Set the GPU device.
                environ["CUDA_VISIBLE_DEVICES"] = "0"
            else:
                self._logger.warning(
                    "BiEncoderTrainer - CUDA not available. Using CPU."
                )

        # "modules" are the model layers.
        model = SentenceTransformer(
            modules=[word_embedding, pooling], device=device
        )

        return model

    def _create_train_objectives(
        self,
        inputs: List[InputExample],
        batch_size: int,
        num_epochs: int,
    ):
        self._logger.info("BiEncoderTrainer - Creating training objectives.")
        loader = datasets.NoDuplicatesDataLoader(inputs, batch_size=batch_size)

        warmup_steps = int(len(loader) * num_epochs * 0.1)

        loss = losses.TripletLoss(self._model, self._distance_metric)

        return ([(loader, loss)], warmup_steps)

    def _callback(self, score, epoch, steps):
        # Invoked after each evaluation.
        self._logger.info(
            f"BiEncoderTrainer - Evaluation\nepoch: {epoch}\nsteps: {steps}\nscore: {score}"
        )

    def _transform_distance_metric(
        self, loss_metric: losses.TripletDistanceMetric
    ) -> evaluation.SimilarityFunction:
        if loss_metric == losses.TripletDistanceMetric.COSINE:
            return evaluation.SimilarityFunction.COSINE
        elif loss_metric == losses.TripletDistanceMetric.EUCLIDEAN:
            return evaluation.SimilarityFunction.EUCLIDEAN
        elif loss_metric == losses.TripletDistanceMetric.MANHATTAN:
            return evaluation.SimilarityFunction.MANHATTAN
        else:
            self._logger.error(
                f"'{loss_metric}' is not a valid distance metric. Exiting."
            )
            exit(1)

    def train(
        self,
        train_inputs: List[InputExample],
        eval_inputs: List[InputExample],
        num_epochs: int,
        batch_size: int,
        save_dir: str,
        checkpoint_limit: int,
        use_amp: bool,
        show_progress_bar: bool = True,
    ):
        """Train the bi-encoder model.

        Training and evaluation inputs should be InputExample objects such that
        each object's `texts` attribute contains
        `[query_text, positive_text, negative_text]`.

        Uses sentence_transformers.losses.TripletLoss.
        Reference: https://www.sbert.net/docs/package_reference/losses.html#tripletloss

        Uses sentence_transformers.TripletEvaluator.
        Reference: https://www.sbert.net/docs/package_reference/evaluation.html#sentence_transformers.evaluation.TripletEvaluator

        Args:
            train_inputs (List[InputExample]): Train samples.
            eval_inputs (List[InputExample]): Evaluation samples. Passed to an
                EmbeddingSimilarityEvaluator.
            num_epochs (int): Number of epochs for training.
            batch_size (int): Batch size for training.
            save_dir (str): Path to a directory to save model and evaluation
                files to.
            checkpoint_limit (int): Total number of checkpoints to save.
            use_amp (bool): Use Automatic Mixed Precision (AMP). Only for
                Pytorch >= 1.6.0
            show_progress_bar (bool, optional): True to show a progress bar,
                False otherwise. Defaults to True.
        """
        self._logger.info("BiEncoderTrainer - Starting training process.")

        makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M")
        output_dir = path.join(
            save_dir, f"{self._base_model.split('/')[-1]}_{timestamp}"
        )
        checkpoint_dir = path.join(output_dir, "checkpoints")

        train_objectives, warmup_steps = self._create_train_objectives(
            train_inputs, batch_size, num_epochs
        )

        evaluator = evaluation.TripletEvaluator.from_input_examples(
            train_inputs,
            main_distance_function=self._distance_function,
            show_progress_bar=show_progress_bar,
        )

        info = {
            "base_model": self._base_model,
            "distance_metric": str(self._distance_metric),
            "num_train": len(train_inputs),
            "num_eval": len(eval_inputs),
            "epochs": num_epochs,
            "batch_size": batch_size,
            "warmup_steps": warmup_steps,
            "output_path": output_dir,
        }

        self._logger.info(f"BiEncoderTrainer - Starting training.\n{info}")

        timer = Timer()
        with timer:
            # Tune
            self._model.fit(
                train_objectives=train_objectives,
                epochs=num_epochs,
                warmup_steps=warmup_steps,
                evaluator=evaluator,
                show_progress_bar=show_progress_bar,
                output_path=output_dir,
                save_best_model=True,
                checkpoint_path=checkpoint_dir,
                checkpoint_save_total_limit=checkpoint_limit,
                use_amp=use_amp,
                callback=self._callback,
            )

        time_elapsed = timer.elapsed
        info["time_elapsed"] = time_elapsed

        with open(path.join(output_dir, "metadata.json"), "w") as outfile:
            json.dump(info, outfile)

        self._logger.info(
            f"BiEncoderTrainer - Finished training. Time elapsed: {time_elapsed}."
        )
