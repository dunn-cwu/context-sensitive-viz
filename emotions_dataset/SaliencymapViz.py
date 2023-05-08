from re import I
import numpy as np
from allennlp.interpret.saliency_interpreters import SimpleGradient
from allennlp.interpret.saliency_interpreters.integrated_gradient import IntegratedGradient
from allennlp.interpret.saliency_interpreters.smooth_gradient import SmoothGradient

import nlpvizbase

class SaliencymapViz(nlpvizbase.nlpVizBase):
    def __init__(self, predictor, vocab, dataset_reader, method = 0) -> None:
        self.predictor = predictor
        self.vocab = vocab
        self.dataset_reader = dataset_reader

        if method == 0:
            self.gradient = SimpleGradient(self.predictor)
        elif method == 1:
            self.gradient = IntegratedGradient(self.predictor)
        else:
            self.gradient = SmoothGradient(self.predictor)

    def classify_sample(self, text):
        output = self.predictor.predict(text)
        output_probs = [(self.vocab.get_token_from_index(label_id, "labels"), prob) for label_id, prob in enumerate(output["probs"])]

        return output_probs

    def _bert_text_tokens_combine(self, text_arr, heatmap_weights):
        has_sep_tok = False
        for t in text_arr:
            if t.startswith("##"):
                has_sep_tok = True
                break

        if not has_sep_tok:
            return text_arr, heatmap_weights

        new_text_arr = []
        new_heatmap_weights = []
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
                    new_heatmap_weights.append(heatmap_weights[i])
                else:
                    word_group = text_arr[group_start:group_end + 1]
                    weight_group = heatmap_weights[group_start:group_end + 1]

                    combined_word = ""
                    for w in word_group:
                        if w.startswith("##"):
                            w = w[2:]
                        combined_word += w

                    max_weight = np.amax(weight_group)

                    new_text_arr[len(new_text_arr) - 1] = combined_word
                    new_heatmap_weights[len(new_heatmap_weights) - 1] = max_weight

                    group_start = -1
                    group_end = -1

        return new_text_arr, new_heatmap_weights

    def get_heatmap_weights(self, text, combine_bert_tokens = False):
        output_probs = self.classify_sample(text)

        if output_probs[0][1] > output_probs[1][1]:
            orig_label = 0
            pred_label = output_probs[0][0]
        else:
            orig_label = 1
            pred_label = output_probs[1][0]

        text_arr = self.dataset_reader.text_to_instance(text)["text"].human_readable_repr()

        dict_output = self.gradient.saliency_interpret_from_json({'sentence': text})

        heatmap_weights = np.array(dict_output["instance_1"]["grad_input_1"])
        heatmap_weights = heatmap_weights - np.min(heatmap_weights)
        heatmap_weights = heatmap_weights / np.max(heatmap_weights)

        if combine_bert_tokens:
            text_arr, heatmap_weights = self._bert_text_tokens_combine(text_arr, heatmap_weights)
        

        return text_arr, heatmap_weights, output_probs[orig_label], orig_label
