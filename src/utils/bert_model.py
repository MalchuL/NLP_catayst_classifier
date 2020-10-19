
import re
import numpy as np
from deeppavlov.core.common.file import read_json
from deeppavlov import build_model, configs
import math

BERT_MODEL = None
def get_bert_text_model(path, device=1):
    global BERT_MODEL
    if BERT_MODEL is None:
        print('Load text model')
        bert_config = read_json(configs.embedder.bert_embedder)
        bert_config['metadata']['variables']['BERT_PATH'] = path
        BERT_MODEL = build_model(bert_config)
    else:
        print('Fast text model already loaded')
    return BERT_MODEL

class BertTextEmbs:
    def __init__(self, path_or_model='wiki.en', seps=' '):
        if isinstance(path_or_model, str):
            self.model = get_bert_text_model(path_or_model)
        else:
            self.model = path_or_model
        self.seps = seps

    def get_sent_mean_embs(self, text):
        embs = self.get_embs(text).mean(0)
        additional_fetures = self.get_additional_fetures(text)
        embs = np.concatenate([embs, additional_fetures])
        return embs

    def get_sent_max_embs(self, text):
        embs = self.get_embs(text).max(0)
        additional_fetures = self.get_additional_fetures(text)
        embs = np.concatenate([embs, additional_fetures])
        return embs

    def preprocess_word(self, word):
        return word


    def get_additional_fetures(self, text): # plus 0.008 to F1
        features = []
        words = text.split(self.seps)
        words = [self.preprocess_word(word) for word in words if
                 len(self.preprocess_word(word)) > 0 and self.preprocess_word(word) in self.model]
        features.append(1 - math.sin(1 / max(1, len(words))))
        features.append(1 - math.cos(1 / max(1, len(words))))
        return np.array(features)


    def get_tokens_embs(self, text):
        tokens, token_embs, subtokens, subtoken_embs, sent_max_embs, sent_mean_embs, bert_pooler_outputs = self.model(
            [text])
        return token_embs[0]

    def get_embs(self, text, bad_string_is_posible=True):
        #words = text.split(self.seps)


        if bad_string_is_posible:
            if len(text) == 0:
                embs = np.zeros([1, 768])
            else:
                embs = self.get_tokens_embs(text)
        else:
            assert len(text) > 0, text
            embs = self.get_tokens_embs(text)
        return embs
