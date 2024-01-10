import json
import re
import spacy
import pandas as pd
from spacy.matcher import DependencyMatcher
from sentence_transformers import SentenceTransformer
import numpy as np
import time
from joblib import Parallel, delayed


def clean_sent(sentence):
    """
    Clean and preprocess a given sentence.

    Args:
        sentence (str): The input sentence.

    Returns:
        str: The cleaned sentence.
    """
    # Combine multiple substitutions into a single regex operation
    cleanSent = re.sub(r'[, ]+', ' ', sentence)
    cleanSent = re.sub(r'\s*([.,])\s*', r'\1', cleanSent)
    cleanSent = re.sub(r"[^a-zA-Z0-9' .,]", '', cleanSent)
    return cleanSent


def clean_phrases(phrase):
    """
    Clean and preprocess a given phrase.

    Args:
        phrase (str): The input phrase.

    Returns:
        str: The cleaned phrase.
    """
    # Combine multiple substitutions into a single regex operation
    newString = re.sub(r"[^a-zA-Z' ]", '', phrase.lower())
    newString = re.sub(r"[' ]+", ' ', newString)

    stop_words = {'further', 'they', 'an', 'is', 'a', 'the', 'any', 'both', 'be', 'that', 'have', 'i', 'it', 'its',
                  'you', 'them', 'their', 'these', 'nearly', 'again', 'very', 'all', 'don', 'more', 'does', 'too',
                  'only', 'few', 'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'a', 's', 'd', 'f', 'g', 'h', 'j',
                  'k', 'l', 'z', 'x', 'c', 'v', 'b', 'n', 'm'}

    # Remove stop words using a set for efficient membership check
    resultwords = [word for word in newString.split() if word not in stop_words]
    return ' '.join(resultwords)


def get_phrases_from_doc_qasrl(data_json, dict_num, nlp):
    """
        Extract phrases from a document using QASRL data and dependency parsing.

        Args:
            data_json (list of dict): The JSON data containing QASRL information.
            dict_num (int): Index of the document in the JSON data.
            nlp: Spacy NLP model.

        Returns:
            list, str, int: A list of extracted phrases, the original sentence, and the document number.
        """
    phrase_list = []

    dict_data = data_json[dict_num]
    words = ' '.join(dict_data['words'])
    sentence = clean_sent(words)
    sentence_nlp = nlp(sentence)

    verbs = dict_data['verbs']

    for verb in verbs:
        for qa in verb['qa_pairs']:
            question = qa['question'].split()
            wh_question = question[0].lower()
            wh = {'when', 'who'}

            keep_word = ""
            if ' '.join(question[:-1]) == 'What is being':
                keep_word = qa['spans'][0]['text']

            if wh_question not in wh:
                verb_lemma = nlp(verb['verb'])[0].lemma_
                last_token = nlp(question[-1])[0].lemma_
                qa_first_span = qa['spans'][0]['text']

                if verb_lemma == last_token or last_token == 'with':
                    phrase_list.append(verb_lemma + ' ' + qa_first_span)
                    if ' '.join(question[:-1]) == 'Why is something being' and keep_word != "":
                        phrase_list.append(verb_lemma + ' ' + keep_word + ' ' + qa_first_span)

    clean_phrases_list = [clean_phrases(phrase) for phrase in phrase_list]
    final_phrases_list = [phrase for phrase in clean_phrases_list if len(phrase.split()) >= 2]

    return final_phrases_list, sentence, dict_num


def create_dependency_patterns(nlp):
    """
        Create dependency patterns for matching in a document.

        Args:
            nlp: Spacy NLP model.

        Returns:
            DependencyMatcher: A Spacy DependencyMatcher with predefined patterns.
        """
    patterns = {
        "OBJECT_Dobj": [
            {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
            {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "object", "RIGHT_ATTRS": {"DEP": "dobj", "POS": "NOUN"}},
        ],
        "OBJECT_middle_Dobj": [
            {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
            {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "object", "RIGHT_ATTRS": {"DEP": "dobj"}},
            {"LEFT_ID": "object", "REL_OP": ">", "RIGHT_ID": "mod/comp", "RIGHT_ATTRS": {"DEP": {"IN": ["amod", "compound"]}}},
        ],
        "OBJECT_middle_Dobj_opposite": [
            {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
            {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "object", "RIGHT_ATTRS": {"DEP": "dobj"}},
            {"LEFT_ID": "object", "REL_OP": "<", "RIGHT_ID": "mod/comp", "RIGHT_ATTRS": {"DEP": {"IN": ["amod", "compound"]}}},
        ],
        "OBJECT_prep_dobj": [
            {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
            {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "object", "RIGHT_ATTRS": {"DEP": "dobj"}},
            {"LEFT_ID": "object", "REL_OP": ">", "RIGHT_ID": "preposition", "RIGHT_ATTRS": {"DEP": {"IN": ["prep", "xcomp"]}}},
            {"LEFT_ID": "preposition", "REL_OP": ">", "RIGHT_ID": "pobj", "RIGHT_ATTRS": {"DEP": "pobj"}},
        ],
        "aux_OBJECT_dobj": [
            {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
            {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "object", "RIGHT_ATTRS": {"DEP": "dobj"}},
            {"LEFT_ID": "verb", "REL_OP": ".", "RIGHT_ID": "auxiliary verb",
             "RIGHT_ATTRS": {"DEP": {"IN": ["aux", "xcomp"]}, "POS": "VERB"}},
        ],
        "OBJECT_oprd": [
            {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
            {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "object predicate",
             "RIGHT_ATTRS": {"DEP": {"IN": ["oprd", "acomp", "prt"]}}},
        ],
        "OBJECT_pobj": [
            {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
            {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "preposition", "RIGHT_ATTRS": {"DEP": "prep"}},
            {"LEFT_ID": "preposition", "REL_OP": ">", "RIGHT_ID": "object of a preposition",
             "RIGHT_ATTRS": {"DEP": "pobj", "POS": "NOUN"}},
        ],
        "OBJECT_advmod": [
            {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
            {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "noun phrase as adverbial modifier",
             "RIGHT_ATTRS": {"DEP": "npadvmod"}},
        ],
        "OBJECT_ADP": [
            {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
            {"LEFT_ID": "verb", "REL_OP": ".", "RIGHT_ID": "adposition",
             "RIGHT_ATTRS": {"DEP": "prt", "POS": "ADP"}},
        ],
        "OBJECT_nsubjpass": [
            {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}},
            {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "nominal subject (passive)",
             "RIGHT_ATTRS": {"DEP": "nsubjpass", "POS": "NOUN"}},
        ],
    }

    matcher = DependencyMatcher(nlp.vocab)
    for pattern_id, pattern in patterns.items():
        matcher.add(pattern_id, [pattern])

    return matcher


def get_phrases_from_doc_depMatch(doc_nlp, matcher):
    """
        Extract phrases from a document using dependency matching.

        Args:
            doc_nlp (spacy.tokens.doc.Doc): Spacy Doc object of the document.
            matcher (DependencyMatcher): DependencyMatcher with predefined patterns.

        Returns:
            list, list: A list of extracted phrases and a list of spans.
        """
    phrase_list = []
    matches = matcher(doc_nlp)
    match_list = []

    for match in range(len(matches)):
        match_list.append(matches[match][1])

    # Use set to remove duplicate matches
    unique_match_list = [list(x) for x in set(frozenset(i) for i in match_list)]

    span_phrases = []
    for match in unique_match_list:
        token_ids = match
        token_ids.sort()
        phrase = []

        for token_id in token_ids:
            token = doc_nlp[token_id]
            lemma = token.lemma_
            pos = token.pos_
            right_edge = token.right_edge

            if pos == "VERB":
                if token.suffix_ == "ing" and token_id != token_ids[0]:
                    phrase.append(lemma.lower())
                elif right_edge.pos_ == "SYM":
                    break
                else:
                    phrase.append(lemma)
            elif right_edge.pos_ == "SYM":
                break
            else:
                phrase.append(lemma)

        if phrase:
            phrase = ' '.join(phrase)
            phrase_list.append(phrase)
            span_phrases.append(token_ids)

    clean_phrases_list = [clean_phrases(phrase) for phrase in phrase_list]
    final_phrases_list = [phrase for phrase in clean_phrases_list if len(phrase.split()) >= 2]

    return final_phrases_list, span_phrases


def print_progress_bar(percentage_done):
    """
        Print a progress bar with a given percentage.

        Args:
            percentage_done (float): The completion percentage.
        """
    bar_length = 25
    filled_length = int(bar_length * percentage_done / 100)
    bar = '#' * filled_length + '-' * (bar_length - filled_length)
    print(f'[{bar}] {percentage_done:.2f}%', end='\r')


def main():
    # Load the models and data
    nlp = spacy.load("en_core_web_sm")
    model_multi_qa = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')

    # Create the dependency matcher
    matcher = create_dependency_patterns(nlp)

    with open('wiki_data_out.json', 'r') as f:
        data = json.load(f)

    bio_sentences_with_prob = pd.read_csv("predictions_only_pos.csv", sep='\t')

    # Initialize lists to store data
    doc_num_list, sentence_list, organism_name_list, sentence_with_title_list = [], [], [], []
    qa_dep_phrases_list, tags_list, original_doc_num_list = [], [], []
    location_list, article_length_list, neg_labels_list, pos_labels_list = [], [], [], []
    bert_predictions_list, certainty_list, embeddings_list = [], [], []

    print('Start collecting phrases...')
    for doc_n, doc_data in enumerate(data):
        percentage_done = doc_n / len(data) * 100
        print_progress_bar(percentage_done)

        qa_phrase_list, sentence, dict_num = get_phrases_from_doc_qasrl(data, doc_n, nlp)
        doc_nlp = nlp(sentence)
        dep_phrase_list, span_phrases = get_phrases_from_doc_depMatch(doc_nlp, matcher)

        # Combine QA and Dependency phrases
        qa_dep_combined_phrase_list = qa_phrase_list + dep_phrase_list
        dup = {x for x in qa_dep_combined_phrase_list if qa_dep_combined_phrase_list.count(x) > 1}
        qa_phrase_list = list(set(qa_phrase_list) - dup)
        dep_phrase_list = list(set(dep_phrase_list) - dup)

        qa_tags = ['qasrl'] * len(qa_phrase_list)
        dep_tags = ['dep'] * len(dep_phrase_list)
        qa_and_dep = ['qa and dep'] * len(dup)
        qa_dep_phrases = qa_phrase_list + dep_phrase_list + list(dup)
        tags = qa_tags + dep_tags + qa_and_dep

        # Get corresponding data from bio_sentences_with_prob
        real_doc_num = int(bio_sentences_with_prob.at[dict_num, 'Unnamed: 0'])
        location = bio_sentences_with_prob.at[dict_num, 'location']
        article_length = bio_sentences_with_prob.at[dict_num, 'article_length']
        neg_labels = bio_sentences_with_prob.at[dict_num, 'neg_labels']
        pos_labels = bio_sentences_with_prob.at[dict_num, 'pos_labels']
        bert_predictions = bio_sentences_with_prob.at[dict_num, 'bert_predictions']
        certainty = bio_sentences_with_prob.at[dict_num, 'certainty']

        # Populate lists
        original_doc_num_list += [real_doc_num] * len(tags)
        location_list += [location] * len(tags)
        article_length_list += [article_length] * len(tags)
        neg_labels_list += [neg_labels] * len(tags)
        pos_labels_list += [pos_labels] * len(tags)
        bert_predictions_list += [bert_predictions] * len(tags)
        certainty_list += [certainty] * len(tags)

        doc_num_list += [dict_num] * len(tags)
        sentence_list += [sentence] * len(tags)
        organism_name_list += [bio_sentences_with_prob.at[dict_num, 'title']] * len(tags)
        sentence_with_title_list += [f"{sentence} ({bio_sentences_with_prob.at[dict_num, 'title']})"] * len(tags)
        qa_dep_phrases_list += qa_phrase_list + dep_phrase_list + list(dup)
        tags_list += tags

        # Compute embeddings
        for i in range(len(qa_dep_phrases)):
            phrase_embd_mpnet = model_multi_qa.encode([qa_dep_phrases[i]])[0]
            embeddings_list.append(phrase_embd_mpnet)
    # Create a DataFrame
    df_all_phrases = pd.DataFrame({
        'doc number': doc_num_list,
        'original doc number': original_doc_num_list,
        'sentence': sentence_list,
        'organism name': organism_name_list,
        'sentence with title': sentence_with_title_list,
        'phrase': qa_dep_phrases_list,
        'algo tag': tags_list,
        'multi-qa-mpnet embeddings': embeddings_list,
        'location': location_list,
        'article_length': article_length_list,
        'neg_labels': neg_labels_list,
        'pos_labels': pos_labels_list,
        'bert_predictions': bert_predictions_list,
        'certainty': certainty_list
    })

    # Save the data to pickle
    df_all_phrases.to_pickle('phrases_with_embeddings.pkl')


if __name__ == "__main__":
    main()