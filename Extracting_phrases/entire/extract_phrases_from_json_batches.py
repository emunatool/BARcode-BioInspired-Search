import json
import re
import time
import spacy
import os
import pandas as pd
from spacy.matcher import DependencyMatcher
from sentence_transformers import SentenceTransformer
start_time = time.time()
model_multi_qa = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')


nlp = spacy.load("en_core_web_sm")


def clean_sent(sentence):
    cleanSent = re.sub(' ,', ',', sentence)
    cleanSent = re.sub(r' \.', '.', cleanSent)
    cleanSent = re.sub("[^a-zA-Z0-9' '.,]", "", cleanSent)
    cleanSent = re.sub(' +', ' ', cleanSent)

    return cleanSent


def clean_phrases(phrase):
    # lower case text
    newString = phrase.lower()

    # remove punctuations
    newString = re.sub("[^a-zA-Z' ']", "", newString)
    newString = re.sub("['  ']", " ", newString)

    # remove specific words
    stop_words = ['further', 'they', 'an', 'is', 'a', 'the', 'any', 'both', 'be', 'that', 'have', 'i', 'it', 'its',
                  'you', 'them', 'their', 'these', 'nearly', 'again', 'very', 'all', 'don', 'more', 'does', 'too',
                  'only', 'few', 'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'a', 's', 'd', 'f', 'g', 'h', 'j',
                  'k', 'l', 'z', 'x', 'c', 'v', 'b', 'n', 'm']  # maybe to add 'both' 'in'

    newStringwords = newString.split()

    resultwords = [word for word in newStringwords if word not in stop_words]
    return ' '.join(resultwords)


def get_phrases_from_doc_qasrl(data_json, dict_num):
    phrase_list = []

    dict = data_json[dict_num]
    index_original_all_wiki = dict['index']
    # print('doc_num: ', dict_num)
    sentence = ' '.join(dict['words'])
    sentence = clean_sent(sentence)

    verbs = dict['verbs']
    # extract the phrase list:
    for verb in verbs:
        for qa in verb['qa_pairs']:
            question = (qa['question']).split()
            wh_question = question[0].lower()
            wh = ['when', 'who']
            if ' '.join(question[:-1]) == 'What is being':
                keep_word = qa['spans'][0]['text']
            if wh_question not in wh:

                verb_lemma = nlp(verb['verb'])[0].lemma_
                last_token = nlp(question[-1])[0].lemma_
                qa_first_span = qa['spans'][0]['text']

                if verb_lemma == last_token or last_token == 'with':
                    if ' '.join(question[:-1]) == 'Why is something being':
                        try:
                            phrase_list.append(verb_lemma + ' ' + keep_word + ' ' + qa_first_span)  # added can be removed
                        except:
                            print("No 'keep word'")
                        finally:
                            phrase_list.append(verb_lemma + ' ' + qa_first_span)
                    else:
                        phrase_list.append(verb_lemma + ' ' + qa_first_span) # added can be removed

    clean_phrases_list = [clean_phrases(phrase) for phrase in phrase_list]
    final_phrases_list = []
    for phrase in clean_phrases_list:
        split_phrase = phrase.split()
        if len(split_phrase) >= 2:
            final_phrases_list.append(phrase)
    return final_phrases_list, sentence, index_original_all_wiki


def dependency_matcher():
    matcher = DependencyMatcher(nlp.vocab)

    OBJECT_Dobj = [
        {
            "RIGHT_ID": "OBJECT",
            "RIGHT_ATTRS": {"POS": "VERB"}

        },
        {
            "LEFT_ID": "OBJECT",
            "REL_OP": ">",
            "RIGHT_ID": "Dobj",
            "RIGHT_ATTRS": {"DEP": "dobj", "POS": "NOUN"},
        }
    ]


    OBJECT_middle_Dobj = [
        {
            "RIGHT_ID": "OBJECT",
            "RIGHT_ATTRS": {"POS": "VERB"}

        },
        {
            "LEFT_ID": "OBJECT",
            "REL_OP": ">",
            "RIGHT_ID": "Dobj",
            "RIGHT_ATTRS": {"DEP": "dobj"},
        },
        {
            "LEFT_ID": "Dobj",
            "REL_OP": ">",
            "RIGHT_ID": "mod/comp",
            "RIGHT_ATTRS": {"DEP": {"IN": ["amod", "compound"]}}

        }
    ]

    OBJECT_middle_Dobj_opposite = [
        {
            "RIGHT_ID": "OBJECT",
            "RIGHT_ATTRS": {"POS": "VERB"}

        },
        {
            "LEFT_ID": "OBJECT",
            "REL_OP": ">",
            "RIGHT_ID": "Dobj",
            "RIGHT_ATTRS": {"DEP": "dobj"},
        },
        {
            "LEFT_ID": "Dobj",
            "REL_OP": "<",
            "RIGHT_ID": "mod/comp",
            "RIGHT_ATTRS": {"DEP": {"IN": ["amod", "compound"]}}
        }
    ]


    OBJECT_prep_Dobj = [
        {
            "RIGHT_ID": "VERB",
            "RIGHT_ATTRS": {"POS": "VERB"}

        },
        {
            "LEFT_ID": "VERB",
            "REL_OP": ">",
            "RIGHT_ID": "dobj",
            "RIGHT_ATTRS": {"DEP": "dobj"},
        },
        {
            "LEFT_ID": "dobj",
            "REL_OP": ">",
            "RIGHT_ID": "prep",
            "RIGHT_ATTRS": {"DEP": {"IN": ["prep", "xcomp"]}},
        },
        {
            "LEFT_ID": "prep",
            "REL_OP": ">",
            "RIGHT_ID": "pobj",
            "RIGHT_ATTRS": {"DEP": "pobj"}
        }
    ]


    aux_OBJECT_Dobj = [
        {
            "RIGHT_ID": "OBJECT",
            "RIGHT_ATTRS": {"POS": "VERB"}

        },
        {
            "LEFT_ID": "OBJECT",
            "REL_OP": ">",
            "RIGHT_ID": "Dobj",
            "RIGHT_ATTRS": {"DEP": "dobj"},

        },
        {
            "LEFT_ID": "OBJECT",
            "REL_OP": ".",
            "RIGHT_ID": "aux verb",
            "RIGHT_ATTRS": {"DEP": {"IN": ["aux", "xcomp"]}, "POS": "VERB"}
        }
    ]


    OBJECT_oprd = [
        {
            "RIGHT_ID": "OBJECT",
            "RIGHT_ATTRS": {"POS": "VERB"}

        },
        {
            "LEFT_ID": "OBJECT",
            "REL_OP": ">",
            "RIGHT_ID": "oprd",
            "RIGHT_ATTRS": {"DEP": {"IN": ["oprd", "acomp", "prt"]}},
        }
    ]

    OBJECT_pobj = [
        {
            "RIGHT_ID": "OBJECT",
            "RIGHT_ATTRS": {"POS": "VERB"}

        },
        {
            "LEFT_ID": "OBJECT",
            "REL_OP": ">",
            "RIGHT_ID": "prep",
            "RIGHT_ATTRS": {"DEP": "prep"},
        },
        {
            "LEFT_ID": "prep",
            "REL_OP": ">",
            "RIGHT_ID": "pobj",
            "RIGHT_ATTRS": {"DEP": "pobj", "POS": "NOUN"},
        }
    ]

    OBJECT_conj = [
        {
            "RIGHT_ID": "OBJECT",
            "RIGHT_ATTRS": {"POS": "VERB"}

        },
        {
            "LEFT_ID": "OBJECT",
            "REL_OP": ">>",
            "RIGHT_ID": "conj",
            "RIGHT_ATTRS": {"DEP": "conj"},
        },
        {
            "LEFT_ID": "conj",
            "REL_OP": ">",
            "RIGHT_ID": "compound",
            "RIGHT_ATTRS": {"DEP": "compound"}
        }
    ]

    OBJECT_advmod = [
        {
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {"POS": "VERB"}

        },
        {
            "LEFT_ID": "verb",
            "REL_OP": ">",
            "RIGHT_ID": "advmod",
            "RIGHT_ATTRS": {"DEP": "npadvmod"},
        }
    ]

    OBJECT_ADP = [
        {
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {"POS": "VERB"}

        },
        {
            "LEFT_ID": "verb",
            "REL_OP": ".",
            "RIGHT_ID": "ADP",
            "RIGHT_ATTRS": {"DEP": "prt", "POS": "ADP"},
        }
    ]

    OBJECT_nsubjpass = [
        {
            "RIGHT_ID": "OBJECT",
            "RIGHT_ATTRS": {"POS": "VERB"}

        },
        {
            "LEFT_ID": "OBJECT",
            "REL_OP": ">",
            "RIGHT_ID": "nsubjpass",
            "RIGHT_ATTRS": {"DEP": "nsubjpass", "POS": "NOUN"},
        }
    ]

    matcher.add("OBJECT_Dobj", [OBJECT_Dobj])
    matcher.add("OBJECT_middle_Dobj", [OBJECT_middle_Dobj])
    matcher.add("OBJECT_middle_Dobj_opposite", [OBJECT_middle_Dobj_opposite])
    matcher.add("aux_OBJECT_Dobj", [aux_OBJECT_Dobj])
    matcher.add("OBJECT_prep_Dobj", [OBJECT_prep_Dobj])
    matcher.add("OBJECT_oprd", [OBJECT_oprd])
    matcher.add("OBJECT_pobj", [OBJECT_pobj])
    matcher.add("OBJECT_advmod", [OBJECT_advmod])
    matcher.add("OBJECT_ADP", [OBJECT_ADP])
    matcher.add("OBJECT_nsubjpass", [OBJECT_nsubjpass])

    return matcher


def get_phrases_from_doc_depMatch(doc_nlp, matcher):
    phrase_list = []

    matches = matcher(doc_nlp)
    match_list = []
    [match_list.append(matches[match][1]) for match in range(len(matches))]
    match_list = [list(x) for x in set(frozenset(i) for i in [set(i) for i in match_list])]

    span_phrases = []
    for match in match_list:
        token_ids = match
        token_ids.sort()
        phrase = []
        for token_id in token_ids:
            if doc_nlp[token_id].pos_ == "VERB":
                if (doc_nlp[token_id].suffix_ == "ing") and (token_id != token_ids[0]):
                    phrase.append(doc_nlp[token_id].lower_)
                elif doc_nlp[token_id].right_edge.pos_ == "SYM":
                    break
                else:
                    phrase.append(doc_nlp[token_id].lemma_)
            elif doc_nlp[token_id].right_edge.pos_ == "SYM":
                break
            else:
                phrase.append(doc_nlp[token_id].lower_)

        if len(phrase) == 0:
            break
        else:
            phrase = ' '.join(phrase)
            phrase_list.append(phrase)
            span_phrases.append(token_ids)

    clean_phrases_list = [clean_phrases(phrase) for phrase in phrase_list]
    final_phrases_list = []
    for phrase in clean_phrases_list:
        split_phrase = phrase.split()
        if len(split_phrase) >= 2:
            final_phrases_list.append(phrase)
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
    file_names = ["batch_{}".format(i) for i in range(1, 14)]

    # Specify the path to the JSON file
    file_path = './batches_for_extracting_phrases/'
    matcher = dependency_matcher()

    # file_path = './Extracting_phrases/entire/batches_for_extracting_phrases/'
    for file_name in file_names:
        # Open the JSON file
        with open(file_path + file_name + '.json', 'r') as file:
            # Load the contents of the file
            data = json.load(file)

        doc_num_list = []
        sentence_list = []
        organism_name_list = []
        sentence_with_title_list = []
        original_sentence_list = []
        original_sentence_with_title_list = []
        qa_dep_phrases_list = []
        tags_list = []
        embeddings_list = []
        bert_pred_list = []
        certainty_list = []

        print('start collecting phrases...')
        df_for_titles = pd.read_csv("predictions_balanced.csv", sep='\t')
        titles = df_for_titles["title"].tolist()
        for doc_n in range(0, len(data)):
            percentage_done = doc_n / len(data) * 100
            print_progress_bar(percentage_done)
            qa_phrase_list, sentence, index_original_all_wiki = get_phrases_from_doc_qasrl(data, doc_n)
            doc_nlp = nlp(sentence)
            dep_phrase_list, span_phrases = get_phrases_from_doc_depMatch(doc_nlp)
            qa_dep_combined_phrase_list = qa_phrase_list + dep_phrase_list
            dup = {x for x in qa_dep_combined_phrase_list if qa_dep_combined_phrase_list.count(x) > 1}
            qa_phrase_list = list(set(qa_phrase_list) - dup)
            dep_phrase_list = list(set(dep_phrase_list) - dup)
            qa_tags = ['qasrl'] * len(qa_phrase_list)
            dep_tags = ['dep'] * len(dep_phrase_list)
            qa_and_dep = ['qa and dep'] * len(dup)
            qa_dep_phrases = qa_phrase_list + dep_phrase_list + list(dup)
            embeddings = []
            for i in range(len(qa_dep_phrases)):
                phrase_embd_mpnet = model_multi_qa.encode([qa_dep_phrases[i]])[0]
                embeddings.append(phrase_embd_mpnet)
            tags = qa_tags + dep_tags + qa_and_dep

            title = titles[int(index_original_all_wiki)]
            sentence_with_title = [sentence + " (" + title + ")"] * len(tags)
            doc_num = [index_original_all_wiki] * len(tags)
            sentence = [sentence] * len(tags)
            organism_name = [title] * len(tags)
            original_sentence = [df_for_titles.loc[index_original_all_wiki, "text"]] * len(tags)
            original_sentence_with_title = [df_for_titles.loc[index_original_all_wiki, "text"] + " (" + title + ")"] * len(tags)
            bert_pred = [df_for_titles.loc[index_original_all_wiki, "bert_predictions"]] * len(tags)
            certainty = [df_for_titles.loc[index_original_all_wiki, "certainty"]] * len(tags)

            bert_pred_list += bert_pred
            certainty_list += certainty
            original_sentence_list += original_sentence
            original_sentence_with_title_list += original_sentence_with_title
            doc_num_list += doc_num
            sentence_list += sentence
            organism_name_list += organism_name
            sentence_with_title_list += sentence_with_title
            qa_dep_phrases_list += qa_dep_phrases
            embeddings_list += embeddings
            tags_list += tags

            progress = (doc_n + 1) / len(data) * 100
            elapsed_time = (time.time() - start_time) / 3600  # Convert elapsed time to hours
            print(f"Progress: {progress:.2f}%, Elapsed Time: {elapsed_time:.2f} hours", end="\r")
            time.sleep(0.1)

        df_all_phrases = pd.DataFrame(list(zip(doc_num_list, sentence_list, organism_name_list, sentence_with_title_list, qa_dep_phrases_list, tags_list, embeddings_list, original_sentence_list, original_sentence_with_title_list, bert_pred_list, certainty_list)),
                                      columns=['original doc number', 'sentence', 'organism name', 'sentence with title', 'phrase', 'algo tag', 'multi-qa-mpnet embeddings', 'original_sentence', 'original_sentence_with_title', 'bert_predictions', 'certainty'])



        # Directory name
        csv_batches = './Extracting_phrases/entire/batches_for_extracting_phrases/'
        # Extract the file name without extension from the file path
        json_file_name = os.path.splitext(os.path.basename(file_path))[0]
        # Create the full file path for the CSV file
        csv_file_path = os.path.join(csv_batches, f"candidate_phrases_entire_{file_name}.pkl")
        # Save the DataFrame to the pkl file in the specific path
        df_all_phrases.to_pickle(csv_file_path)
    print("Processing complete!")
    print()


if __name__ == "__main__":
    main()