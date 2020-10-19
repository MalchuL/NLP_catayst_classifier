
import re
import numpy as np
from gensim.models.wrappers import FastText
import math

FASTTEXT_MODEL = None
def get_fast_text_model(path):
    global FASTTEXT_MODEL
    if FASTTEXT_MODEL is None:
        print('Load text model')
        FASTTEXT_MODEL = FastText.load_fasttext_format(path)
    else:
        print('Fast text model already loaded')
    return FASTTEXT_MODEL

class FastTextEmbs:
    def __init__(self, path_or_model='wiki.en', seps=' '):
        if isinstance(path_or_model, str):
            self.model = FastText.load_fasttext_format(path_or_model)
        else:
            self.model = path_or_model
        self.seps = seps

    def get_sent_mean_embs(self, text):
        #print('mean embs')
        embs = self.get_embs(text).mean(0)
        additional_fetures = self.get_additional_fetures(text)
        embs = np.concatenate([embs, additional_fetures])
        return embs

    def get_sent_max_embs(self, text):
        #print('max embs')
        embs = self.get_embs(text).max(0)
        additional_fetures = self.get_additional_fetures(text)
        embs = np.concatenate([embs, additional_fetures])
        return embs

    def get_sent_full_embs(self, text):
        #print('full embs')
        embs_max = self.get_embs(text).max(0)
        embs_mean = self.get_embs(text).mean(0)
        additional_fetures = self.get_additional_fetures(text)
        embs = np.concatenate([embs_max, embs_mean, additional_fetures])
        return embs

    def preprocess_word(self, word):
        new_word = re.sub(r'\d| ', '', word)  # Remove useless symbols
        return new_word

    def get_additional_fetures(self, text): # plus 0.008 to F1
        features = []
        words = text.split(self.seps)
        words = [self.preprocess_word(word) for word in words if
                 len(self.preprocess_word(word)) > 0 and self.preprocess_word(word) in self.model]
        features.append(1 - math.sin(1 / max(1, len(words))))
        features.append(1 - math.cos(1 / max(1, len(words))))
        return np.array(features)

    def get_embs(self, text, bad_string_is_posible=True):
        words = text.split(self.seps)
        words = [self.preprocess_word(word) for word in words if
                 len(self.preprocess_word(word)) > 0 and self.preprocess_word(word) in self.model]
        if bad_string_is_posible:
            if len(words) == 0:
                embs = np.zeros([1, 300])
            else:
                embs = self.model.wv[words]
        else:
            assert len(words) > 0, text
            embs = self.model.wv[words]
        return embs
