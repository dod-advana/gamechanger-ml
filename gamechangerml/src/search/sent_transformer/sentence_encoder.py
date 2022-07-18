from txtai.embeddings import Embeddings
from txtai.ann import ANN
from pickle import load
from os import remove
from os.path import join
from pandas import DataFrame
from threading import current_thread
import numpy as np
import torch
from gamechangerml.api.utils import processmanager
from .utils import SentenceTransformerFiles


class SentenceEncoder:
    def __init__(self, model_path, use_gpu=False):
        """Handles text encoding and creation of ANNOY index for initial search.

        Args:
            model_path (str): Path to the sentence encoder model to load. Model
                should be supported by huggingface and txtai to generate
                embeddings.
            use_gpu (bool, optional): True to use GPU, False otherwise. Default
                is False.
        """
        self.model_path = model_path
        if use_gpu and torch.cuda.is_available():
            use_gpu = True
        self.embedder = Embeddings(
            {"method": "transformers", "path": self.model_path, "gpu": use_gpu}
        )

    def build_index(
        self,
        corpus,
        save_dir,
        logger,
        update_process_manager=True,
        save_embeddings=False,
    ):
        """Encodes text and creates the ANNOY index for initial search.

        Creates the following files in save_dir:
            data.csv: The input data as a DataFrame with columns "text" and
                "paragraph id".
            The embedder is also saved in save_dir.
        If save_embeddings is True, creates the following files in save_dir:
            embeddings.npy: Document embeddings.
            doc_ids.txt: Document IDs.

        Args:
            corpus (list of tuples): The corpus to process. Each tuple in the
                list should contain paragraph id at index 0 and text at index 1.
            save_dir (str): Path to directory to save files to.
            logger (logging.Logger)
            update_process_manager (bool, optional): True to update
                processmanager while running this process, False otherwise.
                Default is True.
            save_embeddings (bool, optional): True to save document embeddings,
                False otherwise. Defaults to False.

        Returns:
            None
        """
        logger.info("Starting to build sent index.")

        # Initialize progress.
        if update_process_manager:
            process = processmanager.training
            msg = "building sent index"
            processmanager.update_status(
                process, 0, 1, msg, thread_id=current_thread().ident
            )

        # Convert the corpus to a temporary file with embeddings arrays.
        ids, dimensions, stream = self.embedder.model.index(corpus)

        # Load streamed embeddings.
        embeddings = np.empty((len(ids), dimensions), dtype=np.float32)
        with open(stream, "rb") as queue:
            for x in range(embeddings.shape[0]):
                embeddings[x] = load(queue)

        # Remove temporary file.
        remove(stream)

        if save_embeddings:
            np.save(join(save_dir, "embeddings.npy"))
            with open(join(save_dir, "doc_ids.txt"), "w") as f:
                f.writelines([i + "\n" for i in ids])

        # TODO: do we need to save/ reload file or can we just directly pass outputs?
        # Save input data.
        df = DataFrame(
            [(doc[0], doc[1]) for doc in corpus],
            columns=["text", "paragraph_id"],
        )
        df.to_csv(
            join(save_dir, SentenceTransformerFiles.DATA_FILE_NAME), 
            index=False
        )

        # Normalize embeddings.
        self.embedder.normalize(embeddings)

        # Save embeddings metadata.
        self.embedder.config["ids"] = ids
        self.embedder.config["dimensions"] = dimensions

        # Create embeddings index.
        self.embedder.embeddings = ANN.create(self.embedder.config)

        # Build the index.
        self.embedder.embeddings.index(embeddings)

        # Update progress.
        if update_process_manager:
            processmanager.update_status(
                process,
                1,
                1,
                "finished " + msg,
                thread_id=current_thread().ident,
            )

        # Save the embedder.
        self.embedder.save(save_dir)

        logger.info(
            f"Finished building sent index. Saved embedder to {save_dir}."
        )
