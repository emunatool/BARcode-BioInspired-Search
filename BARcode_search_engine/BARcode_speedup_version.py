import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import time
import joblib

# Start measuring time
start_time = time.time()

# Enable or disable log printing
print_logs = True


def time_logs(operation_name, start_time_operation):
    """
    Log the elapsed time for an operation.

    Args:
        operation_name (str): Name of the operation.
        start_time_operation (float): Start time of the operation.
    """
    if not print_logs:
        return

    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time_operation

    # Convert the elapsed time to seconds and minutes
    elapsed_seconds = elapsed_time % 60
    elapsed_minutes = int(elapsed_time / 60)

    # Print the elapsed time
    print(operation_name)
    print(f"Elapsed Time: {elapsed_minutes} minutes and {elapsed_seconds:.2f} seconds")
    pass


def create_svm_model(data_file):
    """
    Create an SVM classifier and train it on the given data.

    Args:
    data_file (str): The path to the CSV data file.

    Returns:
    SVC: The trained SVM classifier.
    """
    print("Creating the SVM classifier...")
    data = pd.read_csv(data_file)

    # Remove duplicate rows based on 'query' and 'phrase' columns
    data = data.drop_duplicates(subset=['query', 'phrase'], keep='first')

    queries = data['query']
    unique_queries = list(set(queries))

    # Sort data by 'rank cosine' in descending order
    sort_df_dec = data.sort_values('rank cosine', ascending=False)

    chosen_data = data[['cosine', 'entailment score', 'contradiction score', 'neutral score', 'relevance labels']]

    # Split data into training and testing sets
    training_set, test_set = train_test_split(chosen_data, test_size=0.3, random_state=1)

    X_train = training_set.iloc[:, 0:4].values
    Y_train = training_set.iloc[:, 4].values
    X_test = test_set.iloc[:, 0:4].values
    Y_test = test_set.iloc[:, 4].values

    # Create and train the SVM classifier
    classifier = SVC(C=100, degree=2, gamma=0.1, kernel='poly')
    classifier.fit(X_train, Y_train)

    # Make predictions and calculate accuracy
    predictions = classifier.predict(X_test)
    cm = confusion_matrix(Y_test, predictions)
    accuracy = float(cm.diagonal().sum()) / len(Y_test)
    print("\nAccuracy Of SVM For The Given Dataset : ", accuracy)

    # Print classification report
    print(classification_report(Y_test, predictions))

    # Re-fit the classifier with the entire chosen dataset
    classifier.fit(chosen_data.iloc[:, 0:4].values, chosen_data.iloc[:, 4].values)

    # Save the trained classifier to a file
    joblib.dump(classifier, 'svm_classifier.pkl')
    return classifier


def cosine_top_num_per_query(query, data_file, top_num, model):
    """
    Calculate cosine similarity for the top N phrases per query.

    Args:
        query (str): The query for which cosine similarity is calculated.
        data_file (str): The path to the CSV data file.
        top_num (int): The number of top phrases to retrieve.
        model (SentenceTransformer): The sentence embedding model used for encoding the query.

    Returns:
        pd.DataFrame: A DataFrame with the top N phrases sorted by cosine similarity.
    """
    print(f"Calculating cosine similarity for query: {query}")

    # Encode the query using the sentence embedding model
    query_embed = model.encode([query])[0]

    start_time_read_candidate_phrases_csv = time.time()

    # Read the pkl file as a Pandas DataFrame
    pandas_df = pd.read_pickle(data_file)
    time_logs("read_candidate_phrases_csv", start_time_read_candidate_phrases_csv)

    embeddings = pandas_df["multi-qa-mpnet embeddings"].tolist()

    start_time_compute_cosine_sim = time.time()
    # Convert the list of embeddings to a 2D array
    ndarray_embeddings = np.array(embeddings)

    # Reshape the query vector to match the shape of the embeddings
    reshaped_vector = query_embed.reshape(1, -1)

    # Compute the cosine similarity between each embedding and the query
    similarities = cosine_similarity(ndarray_embeddings, reshaped_vector)

    pandas_df['cosine_similarity'] = similarities
    time_logs("compute_cosine_sim", start_time_compute_cosine_sim)

    # Sort the DataFrame by cosine similarity in descending order
    df_sorted = pandas_df.sort_values(by='cosine_similarity', ascending=False)

    # Select the top N results
    df_sorted_top_num = df_sorted.head(top_num)

    # Create a new column with the query-candidate phrases cosine values
    df_sorted_top_num.loc[:, "rank cosine"] = list(range(1, df_sorted_top_num.shape[0] + 1))
    return df_sorted_top_num


def nli_addition_top_num_per_query(query, df_sorted_top_4000, model, label_mapping):
    """
    Perform Natural Language Inference (NLI) addition for the top N phrases per query.

    Args:
        query (str): The query for which NLI scores are calculated.
        df_sorted_top_4000 (pd.DataFrame): A DataFrame with the top N phrases sorted by cosine similarity.
        model (CrossEncoder): The NLI model.
        label_mapping (list of str): Mapping of NLI scores to labels.

    Returns:
        pd.DataFrame: A DataFrame with added NLI scores and labels.
    """
    print("Starting NLI...")

    start_time_compute_nli = time.time()

    phrases_topk = df_sorted_top_4000["phrase"].values
    num_phrases = len(phrases_topk)

    start_time_nli_create_pairs = time.time()

    # Create pairs of phrases and the query
    pairs_list = [(query, phrase) for phrase in phrases_topk]
    pairs = np.column_stack((phrases_topk, np.repeat(query, num_phrases)))
    time_logs("create pairs:", start_time_nli_create_pairs)

    start_time_nli_predict = time.time()

    # Predict NLI scores
    scores = model.predict(pairs)

    time_logs("predict NLI scores of pairs:", start_time_nli_predict)

    start_time_nli_map_labels = time.time()

    # Map scores to NLI labels
    labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
    time_logs("create map labels:", start_time_nli_map_labels)

    contradiction_list = scores[:, 0]
    entailment_list = scores[:, 1]
    neutral_list = scores[:, 2]

    df_sorted_top_4000['NLI labels'] = labels
    df_sorted_top_4000['entailment score'] = entailment_list
    df_sorted_top_4000['contradiction score'] = contradiction_list
    df_sorted_top_4000['neutral score'] = neutral_list
    time_logs("compute NLI", start_time_compute_nli)

    return df_sorted_top_4000


def weighted_per_query(df_top_num, classifier):
    """
    Calculate weighted scores for the top N phrases per query.

    Args:
        df_top_num (pd.DataFrame): A DataFrame with NLI scores and labels.
        classifier (SVC): The SVM classifier.

    Returns:
        pd.DataFrame: A DataFrame with added weighted scores and ranks.
    """
    print("Calculating weighted score...")

    chosen_data = df_top_num[['cosine_similarity', 'entailment score', 'contradiction score', 'neutral score']]

    # Predict relevance scores
    Y_pred = classifier.predict(chosen_data)

    df_top_num["Predictions"] = Y_pred

    # Calculate weighted scores using the decision function
    weighted_score = classifier.decision_function(chosen_data)
    df_top_num["weighted score"] = weighted_score

    # Sort the DataFrame by weighted score in descending order
    df_top_num = df_top_num.sort_values(by='weighted score', ascending=False)

    # Remove duplicates based on certain columns
    df_top_num = df_top_num.drop_duplicates(subset=['original doc number'], keep='first')
    df_top_num = df_top_num.drop_duplicates(subset=['sentence'], keep='first')

    # Drop the unnecessary column
    df_top_num = df_top_num.drop(['multi-qa-mpnet embeddings'], axis=1)

    # Add a rank column based on the weighted score
    df_top_num["rank weighted score"] = list(range(1, df_top_num.shape[0] + 1))

    return df_top_num


def load_or_create_embeddings(model, input_pickle, output_pickle):
    """
    Load or create embeddings for candidate phrases and save them to a DataFrame.

    Parameters:
    - model: The language model used for encoding phrases.
    - input_pickle (str): Path to the input pickle file containing the DataFrame with candidate phrases.
    - output_pickle (str): Path to the output pickle file where the updated DataFrame will be saved (with embeddings).

    If the pickle file does not exist, it creates a DataFrame with embeddings.
    """
    if os.path.exists(input_pickle):
        # File exists, load DataFrame
        print(input_pickle + " already exist")
    else:
        # File does not exist, create DataFrame with embeddings
        print(input_pickle + " does not exist, create embeddings...")
        df_candidate_phrases = pd.read_pickle(output_pickle)
        candidate_phrases_list = df_candidate_phrases["phrase"].tolist()
        embeddings_list_filtered_data = [model.encode([phrase])[0] for phrase in candidate_phrases_list]
        df_candidate_phrases["multi-qa-mpnet embeddings"] = embeddings_list_filtered_data
        df_candidate_phrases.to_pickle(input_pickle)
    pass


def run_query(query_list, num_phrases_first_process, num_phrases_sec_process, top_n_results, useSpeedUp = True):
    """
    Execute a query processing pipeline to retrieve relevant sentences and phrases based on user queries.
    Outputs csv files with the rank list of inspirations per query in the 'BARcode_search_engine' folder.
    Args:
        query_list (list): A list of user queries to be processed.
        num_phrases_first_process (int): Number of phrases to process in the initial step.
        num_phrases_sec_process (int): Number of additional phrases to process in the secondary step.
        top_n_results (int): Number of top results to be displayed.
        useSpeedUp (bool): Flag to enable or disable speed-up mode.

    Returns:
        None
    """
    start_time_load_models = time.time()
    # Load pre-trained models
    model_multi_qa = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
    model_crossencoder = CrossEncoder('cross-encoder/nli-deberta-v3-base')
    label_mapping = ['contradiction', 'entailment', 'neutral']

    time_logs("load models", start_time_load_models)

    # Process each query in the query list
    for query in query_list:
        initial_N = num_phrases_first_process
        rest_N = num_phrases_sec_process

        speedUpName = ""
        if useSpeedUp:
            speedUpName = "_speedUp"
            load_or_create_embeddings(model_multi_qa, 'csv_candidate_phrases_filtered_data.pkl', 'csv_candidate_phrases_filtered_data_without_embeddings.pkl')
            df_top_num = cosine_top_num_per_query(query, 'csv_candidate_phrases_filtered_data.pkl',
                                                  top_num=initial_N + rest_N, model=model_multi_qa)
        else:
            load_or_create_embeddings(model_multi_qa, 'candidate_phrases_entire_data.pkl', 'csv_candidate_phrases_entire_data_without_embeddings.pkl')
            df_top_num = cosine_top_num_per_query(query, 'candidate_phrases_entire_data.pkl', top_num=initial_N + rest_N, model=model_multi_qa)

    # Train the SVM model on the entire dataset and get the classifier
        classifier = joblib.load('svm_classifier.pkl')

        initial_top_num = df_top_num.copy(deep=True)[:initial_N]
        rest_top_num = df_top_num.copy(deep=True)[initial_N:initial_N + rest_N]

        initial_df_top_num_per_query = nli_addition_top_num_per_query(query, initial_top_num, model_crossencoder, label_mapping)
        initial_results = weighted_per_query(initial_df_top_num_per_query, classifier)

        # Display top 5 of initial_results to the user

        rest_df_top_num_per_query = nli_addition_top_num_per_query(query, rest_top_num, model_crossencoder, label_mapping)
        total_df_top_num_per_query = pd.concat([initial_df_top_num_per_query, rest_df_top_num_per_query], axis=0)
        total_results = weighted_per_query(total_df_top_num_per_query, classifier)

        # Add the rest top 5 - 15 of total_results to the user

        # Save the results to a CSV file with a query-specific name
        selected_results = total_results[["rank weighted score", "organism name", "sentence", "phrase"]].copy()

        selected_results.head(top_n_results).to_csv(query + '_top_n_results_BARcode' + speedUpName + '.csv', index=False)

    time_logs("full run", start_time)
    pass


def main():
    run_query(query_list=['collect water from air'], num_phrases_first_process=1000, num_phrases_sec_process=3000, top_n_results=15)
    # run_query(query_list=['collect water from air'], num_phrases_first_process=1000, num_phrases_sec_process=3000, top_n_results=15,  useSpeedUp=False)
    print("Finished!")

if __name__ == "__main__":
    main()
