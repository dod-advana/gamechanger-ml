from pandas import DataFrame
from networkx import DiGraph, pagerank



class Rank:
    
    @staticmethod
    def pagerank_docs(docs, alpha=0.85):
        """Get pagerank scores for documents.

        The pagerank scores are generated using each document's `doc_id` as
        nodes and items in each document's `ref_list` as edges.

        Args:
            documents (list of dict): Document dictionaries with fields
                `doc_id` (str) and `ref_list` (list of str).
            alpha (float, optional): Damping parameter for pagerank.
                Default is 0.85.

        Returns:
            pandas.DataFrame: DataFrame, with columns `doc_id` and `pr`
                (`pr` is the pagerank score), sorted in descending order by
                `pr`.
        """
        nodes = [doc["doc_id"] for doc in docs]
        edges = [(doc["doc_id"], r) for doc in docs for r in doc["ref_list"]]
        # Create graph.
        graph = DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        # Get pagerank scores & sort by them.
        pr = pagerank(graph, alpha)
        pr = {
            k: v
            for k, v in sorted(
                pr.items(), key=lambda item: item[1], reverse=True
            )
        }
        df = DataFrame.from_dict(pr, orient="index")
        df.reset_index(inplace=True)
        df.rename(columns={"index": "doc_id", 0: "pr"}, inplace=True)

        return df
