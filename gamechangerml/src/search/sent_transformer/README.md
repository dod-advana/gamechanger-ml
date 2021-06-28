<h1>Sentence Transformers</h1>

<h2>Overview:</h2>
<p>Sentence Transformers are GAMECHANGER’s intelligent (semantic) search solution. Semantic search enables us to return results that don’t contain exact keyword matches but are similar in intent. For example, a semantic query for “artificial intelligence” should also return results for documents that describe “machine learning” (which is more similar than results that may contain “intelligence” or the separated terms “artificial” and “intelligence”.) In the case of longer queries (ex. “who is in charge of the DoD budget?”) semantic search should be able to retrieve documents that address this topic even if phrased differently.</p>

The current method employs two models for returning intelligent search results: a retriever model for returning top n documents relevant to the query, and a similarity model for re-ranking the most relevant result to the top.

<h2>How Sentence Transformers Work in the App</h2>

1. In the web application, the user's query is sent to the `/transSentenceSearch` endpoint of the mlApi. 
2. In the mlApi:
   -  the SentenceEncoder class encodes the query as an embedding by the <b>retriever model</b> so it can be compared against paragraph-level embeddings of the corpus
   -  the SentenceSearcher class retrieves the top n paragraphs in our <b>embeddings index</b> with the highest cosine similiarity to the query embedding (using a Faiss indexing search from txtai)
   -  the SentenceSearcher class feeds the top results (as strings) and query string into the <b>similarity model</b> and re-ranks the results based on their semantic similarity to the original query, with the most similar ranked at the top.
3. In the webapp, (currently) only the top intelligent result is used: if there are results from the sentence transformer search, the top result is re-ordered to the top of all search results (if it is not in the existing results, it is appended to the beginning of the first page of results).

<i>Relevant Code:</i>
- Webapp: 
  -  `gamechanger-web/backend/node_app/utils/searchUtility.js` (see combinedSearchHandler)
- Mlapp: 
  -  `gamechangerml/src/search/sent_transformer/model.py` (see SentenceEncoder and SentenceSearcher classes)
  -  `gamechangerml/api/fastapi/mlapp.py`

<h2>How Sentence Transformers are Updated</h2>

<h3>Retriever Model & Embeddings Index</h3>

The base model used for the encoding and retrieving the embeddings is `msmarco-distilbert-base-v2` from [huggingface](https://huggingface.co/sentence-transformers/msmarco-distilbert-base-v2)

<i>Creating the Embeddings Index</i>:

The `SentenceEncoder` class in `./model.py` includes two methods for creating the embeddings index:
-  `index_documents` takes in the directory of the corpus and path for storing the index, and generates embeddings with the encoder model
-  `new_index_documents` \[experimental\] takes in the directory of the original corpus and newly parsed corpus, concatenates them, and generates an embedding index (see data parsing) 

<i>Data Parsing</i>:

Certain types of heavily referenced policy documents are highly structured - unlike journal or Wikipedia articles, their paragraph-level content is often bulleted/listed within sections and subsections (and chapters, etc.). Because our default parsing methods break up documents on linebreaks into paragraph-level text chunks, bulleted lists and sections can lose important contextual meaning without their headers. 

To create better embeddings, some work has been done to re-parse certain types of highly structured DoD documents into more meaningful paragraphs. Ex: "PURPOSE: \n\n - To create a Center of Excellence for Artificial Intelligence..." would originally have been parsed into two paragraphs: with the new methods, the header and bullet would be combined into a single paragraph. This means that "purpose" level sentences should be indexed closer in the embedding space and retrieved more easily for queries looking for statements of purpose.

  -  the new document parsing is done in `gamechangerml/src/featurization/doc_parser/parser.py`. Current implementation includes regex patterns for parsing DoDI/DoDM/DoDM documents. 

<i>Relevant Code:</i>
  -  `./create_embeddings.py` [under review] creates the embeddings index (using the old corpus and newly parsed corpus files) 
  -  `./corpus.py` [under review] creates a SentenceCorpus class that generates pairs of sentences from the newly parsed corpus and scores their similarity based on document structure (not using a similarity model) -> this method is intended to create training data for both models but is currently under review.
  -  `./dataset.py` [under review] calls the SentenceCorpus class to generate a csv of sentence pairs and similarity scores (training data for models)
  -  `./train_embed.py` finetunes the `msmarco-distilbert-base-v2` model using the dataset.csv and testset.csv generated in `./dataset.py` -> training the model to adjust embedding similarity based on the document structure (documentation on finetuning: https://www.sbert.net/docs/training/overview.html)

<h3>Reader (Semantic Similarity) Model</h3>

The base model used for the Reader (Similarity) Model is `distilbart-mnli-12-3` from [huggingface](https://huggingface.co/valhalla/distilbart-mnli-12-3)

  -  `./trainer.py` [under review] uses the SentenceCorpus generated data to finetune `distilbart-mnli-12-3` 

