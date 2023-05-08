from allennlp.predictors import predictor
import spacy
from spacy.matcher import Matcher
from nltk import Tree
import matplotlib.pyplot as plt
import numpy as np

import nlpvizbase

# https://stackoverflow.com/questions/36610179/how-to-get-the-dependency-tree-with-spacy
def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_

def get_depend_triplets(sent):
    return _get_depend_triplets(sent.root, None, [])

def _get_depend_triplets(node, parent, ret_list):
    if node.n_lefts + node.n_rights <= 0:
        return ret_list

    for c in node.children:
        if parent != None:
            pair = [parent.i, node.i, c.i]
            pair.sort()

            ret_list.append(pair)

        _get_depend_triplets(c, node, ret_list)

    return ret_list

def get_depend_pairs(sent):
    return _get_depend_pairs(sent.root, [])

def _get_depend_pairs(node, ret_list):
    if node.n_lefts + node.n_rights <= 0:
        return ret_list

    for c in node.children:
        pair = [node.i, c.i]
        pair.sort()

        ret_list.append(pair)
        _get_depend_pairs(c, ret_list)

    return ret_list

def extract_best_group(output_diff, input_groups, spacey_tokens):
    max_pair = np.argmax(output_diff)
    if output_diff[max_pair] <= 0:
        return -1, []

    sel_pair = input_groups[max_pair]

    #print("Most important words:")
    # for i in sel_pair:
    #     print("    " + spacey_tokens[i].text, "(" + str(spacey_tokens[i].sent.root) + ", " + str(spacey_tokens[i].dep_) + ")")

    #print("")

    #print("Word positions: ", end="")

    # for t in spacey_tokens:
    #     if t.i in sel_pair:
    #         print("[" + t.text + "]", end=" ")
    #     else:
    #         print(t.text, end=" ")
    #print("\n")

    return max_pair, sel_pair

class LeaveoneoutViz(nlpvizbase.nlpVizBase):
    def __init__(self, predictor, vocab, dataset_reader) -> None:
        self.predictor = predictor
        self.vocab = vocab
        self.dataset_reader = dataset_reader

        self.nlp = spacy.load('en_core_web_sm')
        self.matcher = Matcher(self.nlp.vocab)

    def classify_sample(self, text):
        output = self.predictor.predict(text)
        output_probs = [(self.vocab.get_token_from_index(label_id, "labels"), prob) for label_id, prob in enumerate(output["probs"])]

        return output_probs

    def _bert_text_tokens_combine(self, text_arr):
        has_sep_tok = False
        for t in text_arr:
            if t.startswith("##"):
                has_sep_tok = True
                break

        if not has_sep_tok:
            return text_arr

        new_text_arr = []
        group_start = -1
        group_end = -1

        for i in range(len(text_arr)):
            if text_arr[i].startswith("##"):
                if group_start == -1:
                    group_start = i - 1
                group_end = i
            else:
                if group_start == -1:
                    new_text_arr.append(text_arr[i])

                else:
                    word_group = text_arr[group_start:group_end + 1]

                    combined_word = ""
                    for w in word_group:
                        if w.startswith("##"):
                            w = w[2:]
                        combined_word += w

                    new_text_arr[len(new_text_arr) - 1] = combined_word

                    group_start = -1
                    group_end = -1

        return new_text_arr

    def get_heatmap_weights(self, text, use_bert_tokens = False):

        if use_bert_tokens:
            bert_text_arr = self._bert_text_tokens_combine(self.dataset_reader.text_to_instance(text)["text"].human_readable_repr())
            core_text_arr = bert_text_arr[1:len(bert_text_arr) - 1]
            text = " ".join(core_text_arr)

        spacey_tokens = self.nlp(text)
        text_tokens = [e.text for e in spacey_tokens]
        
        orig_classification = self.classify_sample(" ".join(text_tokens))

        orig_label = 0
        for i in range(len(orig_classification)):
            if (orig_classification[i][1] > orig_classification[orig_label][1]):
                orig_label = i

        orig_pred = orig_classification[orig_label][1]

        #print("Prediction label:", orig_label)

        input_sentences = []
        output_diff = [0.0] * len(text_tokens)

        for i in range(len(text_tokens)):
            modified_sent = text_tokens.copy()
            del modified_sent[i]

            input_sentences.append(" ".join(modified_sent))


        for i in range(len(input_sentences)):
            classification = self.classify_sample(input_sentences[i])
            output_diff[i] = orig_pred - classification[orig_label][1]
            if output_diff[i] < 0:
                output_diff[i] = 0.0


        word_weights = np.array(output_diff.copy())
        word_weights = word_weights / np.amax(output_diff)
        

        if use_bert_tokens:

            if len(word_weights) != len(bert_text_arr) - 2:
                raise RuntimeError("Word weight and text token list length mismatch: " + str(len(word_weights)) + " != " + str(len(bert_text_arr) - 2))

            # TODO: Replace with numpy insert code
            word_weights = word_weights.tolist()

            # Add 0 weights to start and end of list to account for bert [CLS] and [SEP] tokens
            word_weights.insert(0, 0.0)
            word_weights.insert(len(word_weights), 0.0)
            text_tokens = bert_text_arr

        return text_tokens, np.array(word_weights), orig_classification[orig_label], orig_label
        
