from gamechangerml.src.text_handling.corpus import LocalCorpus


def prepare_corpus_for_encoder(
    corpus,
    min_token_len,
    return_id,
    verbose,
    logger,
    files_to_use=None,
    bert_based_tokenizer=False,
):
    """Prepare a corpus for SentenceEncoder.

    Args:
        corpus (str or MSMarcoData().corpus): If str, path to the corpus. 
            Otherwise, MSMarcoData().corpus can be used for testing purposes.
        min_token_len: min_token_len argument for LocalCorpus
        return_id: return_id argument for LocalCorpus
        verbose: verbose argument for LocalCorpus
        logger (logging.Logger): _description_
        files_to_use (list of str or None, optional): files_to_use argument 
            for LocalCorpus. Defaults to None.
        bert_based_tokenizer (bool, optional): bert_based_tokenizer argument 
            for LocalCorpus. Defaults to False.

    Returns:
        list of tuples: Each tuple will contain:
            - paragraph id at index 0
            - text at index 1
            - None at index 2
    """
    if type(corpus) == str:
        corpus = LocalCorpus(
            corpus,
            return_id=return_id,
            min_token_len=min_token_len,
            verbose=verbose,
            bert_based_tokenizer=bert_based_tokenizer,
            files_to_use=files_to_use,
        )
        corpus = [
            (par_id, " ".join(tokens), None) for tokens, par_id in corpus
        ]
    else:
        logger.info("Preparing test corpus for encoder.")

    return corpus
