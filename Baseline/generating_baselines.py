# Import necessary libraries
from elasticsearch import Elasticsearch
import pandas as pd

# Initialize an Elasticsearch client
es_before_snorkel = Elasticsearch(HOST="http://localhost", PORT=9200)
# es_before_snorkel = Elasticsearch()

def create_wiki_sent_index(df_data):
    """
    Create an Elasticsearch index for Wikipedia sentences using a DataFrame.

    Args:
        df_data (pd.DataFrame): DataFrame containing the sentences and associated indices.

    Returns:
        None
    """
    sentences_indices = df_data["Unnamed: 0"].tolist()
    sentences = df_data["text"].tolist()

    for i in sentences_indices:
        print(i)
        sent = {'sentence': sentences[i]}
        es_before_snorkel.index(index="biomimicry_before_snorkel", doc_type="sentences_before_snorkel", id=i, body=sent)

def print_hits(hits):
    """
    Print information about search hits retrieved from Elasticsearch.

    Args:
        hits (list of dict): List of Elasticsearch search hits.

    Returns:
        None
    """
    print("--Elastic Search (baseline)--\n")
    for hit in hits:
        print("Doc number: ", hit['_id'], " ,Score: ", hit['_score'])
        print("Sentence: ", hit['_source']['sentence'])
        print("---------------------------------------------------------------------------")

def query_top_n_matches(query, n=10):
    """
    Search for the top 'n' matches in the Elasticsearch index based on a query.

    Args:
        query (str): The search query.
        n (int): Number of matches to retrieve.

    Returns:
        dict, list of dict: Elasticsearch search results and list of hits.
    """
    print(str(query))
    query_body = {
        "from": 0,
        "size": n,
        "query": {
            "match": {
                "sentence": query
            }
        }
    }
    top10_matches = es_before_snorkel.search(index="biomimicry_before_snorkel", body=query_body)
    hits = top10_matches['hits']['hits']
    print_hits(hits)
    return top10_matches, hits


def main():
    # Load filtered and entire data DataFrames
    df_filtered_data = pd.read_pickle('filtered_data.pkl')
    df_entire_data = pd.read_pickle('entire_data.pkl')

    # Choose data type ("filtered" or "entire") for creating the index
    data_type = "filtered"

    if data_type == "filtered":
        create_wiki_sent_index(df_filtered_data)
    elif data_type == "entire":
        create_wiki_sent_index(df_entire_data)

    # Search for the top 10 matches for a sample query
    top10_matches, hits = query_top_n_matches("collect water from air", n=10)
    print()


if __name__ == "__main__":
    main()
