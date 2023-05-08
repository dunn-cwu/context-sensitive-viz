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

def is_target_pos(spacey_token):
    return True
    #return spacey_token.pos_ == 'NOUN' or spacey_token.pos_ == 'VERB' or spacey_token.pos_ == 'ADV' or spacey_token.pos_ == 'ADJ'

def _gen_perm(tokens, n, start_index, base_group):
    if len(base_group) >= n:
        return [base_group]

    word_groups = []

    for i in range(start_index, len(tokens) - (n - len(base_group))):
        grp = base_group.copy()
        grp.append(tokens[i].i)
        word_groups += _gen_perm(tokens, n, i + 1, grp)

    return word_groups

def gen_perm(tokens, n):
    if n < 1:
        return []
    elif n == 1:
        word_groups = []
        for i in range(len(tokens) - n):
            word_groups.append([tokens[i].i])

        return word_groups

    word_groups = []

    for i in range(len(tokens) - n):
        word_groups += _gen_perm(tokens, n, i + 1, [tokens[i].i])

    return word_groups

def generate_word_groups(spacey_tokens, n):
    tokens = []

    for t in spacey_tokens:
        if is_target_pos(t):
            tokens.append(t)

    if len(tokens) < n:
        return None

    return gen_perm(tokens, n)

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

def extract_best_group(output_diff, input_groups, skip_neg = False):
    max_pair = np.argmax(output_diff)
    if skip_neg and output_diff[max_pair] <= 0:
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

class LeavenoutViz(nlpvizbase.nlpVizBase):
    def __init__(self, predictor, vocab, dataset_reader, use_depend_parser = True, top_n_min = 5) -> None:
        self.predictor = predictor
        self.vocab = vocab
        self.dataset_reader = dataset_reader
        self.use_depend_parser = use_depend_parser
        self.top_n_min = top_n_min

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
        orig_text = " ".join([e.text for e in spacey_tokens])
        #print("Original text:", orig_text)

        #print("\nPOS tags:")
        # for token in spacey_tokens:
        #     print(str(token.text) + " (" + str(token.pos_) + ")")

        #print("\n")

        orig_classification = self.classify_sample(text)

        orig_label = 0
        for i in range(len(orig_classification)):
            if (orig_classification[i][1] > orig_classification[orig_label][1]):
                orig_label = i

        orig_pred = orig_classification[orig_label][1]

        #print("Prediction label:", orig_label)

        sentences = list(spacey_tokens.sents)
        input_sentences = []
        input_groups = []

        for sent in sentences:
            if self.use_depend_parser:
                # word_triplets = get_depend_triplets(sent)
                word_groups = get_depend_pairs(sent)
                #input_groups = input_groups + word_groups + word_triplets
                input_groups = input_groups + word_groups
            else:
                word_groups = generate_word_groups(sent, 2)
                if word_groups == None:
                    continue

                input_groups = input_groups + word_groups

            #sent_text = " ".join([e.text for e in sent])
            #print("Processing:", sent_text)
            
            #to_nltk_tree(sent.root).pretty_print()

            #print("Generated", len(word_groups), "word groups ...")

            for group in word_groups:
                modified_sent = []
                for token in spacey_tokens:
                    if token.i in group:
                        continue
                    modified_sent.append(token)

                input_sentences.append(" ".join([e.text for e in modified_sent]))


        if len(input_groups) != len(input_sentences):
            raise Exception("Input groups and input sentences length mismatch.")

        total_num_input_grps = len(input_groups)
        #print("Number of word groups: " + str(total_num_input_grps))
        
        input_groups = np.array(input_groups)
        output_diff = np.zeros(len(input_sentences))

        for i in range(len(input_sentences)):
            classification = self.classify_sample(input_sentences[i])
            output_diff[i] = orig_pred - classification[orig_label][1]

        pair_counter = 0

        top_word_groups = []
        group_weights = []
        max_weight = 0

        while output_diff.shape[0] > 0:
            pair_counter += 1
            #print("\nExtracting pair #", pair_counter, "...")

            grp_index, word_indices = extract_best_group(output_diff, input_groups, len(top_word_groups) >= self.top_n_min)
            if grp_index < 0:
                #print("No more positive diff pairs. Stopping search ...")
                break

            if pair_counter <= 1:
                max_weight = output_diff[grp_index]
                group_weights.append(1.0)
            else:
                group_weights.append(output_diff[grp_index] / max_weight)

            output_diff = np.delete(output_diff, grp_index)
            input_groups = np.delete(input_groups, grp_index, 0)

            top_word_groups.append(word_indices)

        word_weights = [None] * len(spacey_tokens)

        for i in range(len(top_word_groups)):
            for token_index in top_word_groups[i]:
                if word_weights[token_index] != None:
                    continue

                word_weights[token_index] = group_weights[i]


        text_arr = [token.text for token in spacey_tokens]
        word_weights = [0.0 if w == None else w for w in word_weights]

        if use_bert_tokens:

            if len(word_weights) != len(bert_text_arr) - 2:
                raise RuntimeError("Word weight and text token list length mismatch: " + str(len(word_weights)) + " != " + str(len(bert_text_arr) - 2))

            # Add 0 weights to start and end of list to account for bert [CLS] and [SEP] tokens
            word_weights.insert(0, 0.0)
            word_weights.insert(len(word_weights), 0.0)
            text_arr = bert_text_arr

        return text_arr, np.array(word_weights), orig_classification[orig_label], orig_label, total_num_input_grps
        
