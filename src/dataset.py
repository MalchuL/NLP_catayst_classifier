import collections

import torch
import torch.utils.data as data
import re
import numpy as np
from gensim.models.wrappers import FastText
import pyarrow.parquet as pq
import math
from .utils.fasttext_model import FastTextEmbs, get_fast_text_model
from torch.nn.utils.rnn import pack_sequence, pad_sequence, pack_padded_sequence
from torch.utils.data.dataloader import default_collate

def convert_answers_to_embs(row, column_names, process_funcs, aggregation='concat'):
    embs = []
    for column_name, process_func in zip(column_names, process_funcs):
        if process_func is None:
            process_func = lambda x: x
        embs.append(process_func(row[column_name]))
    if aggregation == 'concat':
        result = np.concatenate(embs, axis=0)
    return result





def split_tags(tags):
    return re.findall(r'<((?:\w|[#\+\-\.\?\!\)\(])+?)>', tags)


def cleanup_text(text):
    new_text = text
    new_text = re.sub(r'\[|\]|\(|\)|\"|\{|\}|\<|\>|\'|\:|\;|\»|\«' + r'|\n|\r|\t', '. ', new_text) # Split by separators


    # new_text = re.sub(r'\[|\(|\{|\<|\«', '. bracked left ', new_text) # Split by separators
    # new_text = re.sub(r'\]|\)|\}|\>|\«', ' bracked right. ', new_text) # Split by separators
    # new_text = re.sub(r'\"|\'', ' quotes ', new_text) # Split by separators
    # new_text = re.sub(r'\n|\r|\t|\:|\;', '. ', new_text) # Split by separators

    new_text = re.sub(r'\@|\$|\*|\[|\]|\(|\)|\"|\^|\~|\{|\}|\<|\>|\,|\'|\:|\;|\»|\«|\-|\_|\+|\%' + r'|\n|\r|\t', ' ', new_text) # Remove useless symbols
    new_text =  re.sub(' +', ' ', new_text) # Remove duplicated spaces
    new_text = new_text.lower().strip()

    return new_text


def process_text_onto_sentences(text, max_len_symbols=1):
    new_text = cleanup_text(text)
    new_text = [sent for sent in re.split('\. |\! |\? ', new_text) if len(sent) > max_len_symbols]

    return new_text
def get_sent_emb(text, aggregation='max', model=None):
    if aggregation == 'max':
        result = model.get_sent_max_embs(text)
    elif aggregation == 'mean':
        result = model.get_sent_mean_embs(text)
    else:
        result = model.get_sent_full_embs(text)
    return result

def get_stack_overflow_embs(row, model):
    tags_func = lambda text: get_sent_emb(' '.join(split_tags(text)), model=model)
    body_func = lambda text: get_sent_emb(cleanup_text(text), model=model)
    emb = convert_answers_to_embs(row, ['Body', 'Tags', 'Title'], [body_func, tags_func, body_func])

    target = row['target']
    return emb, target


class MockFasttext:
    class WV():
        def __getitem__(self, text):
            return np.random.randn(1, 300)

    def __init__(self):
        self.wv = MockFasttext.WV()

    @classmethod
    def load_fasttext_format(*args, **kwargs):
        return MockFasttext()

    def __contains__(self, key):
        return True


class NLPStackOverflowClassificationDataset(data.Dataset):
    def __init__(self, parque_path, fast_text_path, use_sencence_embs=True, embs_aggregation='max', use_mock=False):
        self.parque_path = parque_path
        self.use_sencence_embs = use_sencence_embs
        self.num_classes = 3
        print('load fast text model')
        if use_mock:
            print('use mock')
            self.model = FastTextEmbs(MockFasttext())
        else:
            self.model = FastTextEmbs(get_fast_text_model(fast_text_path))
        self.embs_aggregation = embs_aggregation
        print('load fast text model was finished')
        self.df = pq.read_pandas(parque_path).to_pandas()

    def sencences_to_embs(self, text):
        sentences = process_text_onto_sentences(text)
        assert len(sentences) > 0, (text,sentences)
        sentences_emb = [get_sent_emb(sentence, self.embs_aggregation, model=self.model) for sentence in sentences]
        return np.stack(sentences_emb)

    def __getitem__(self, item):
        elem = self.df.iloc[item]
        # Body part
        body_text = elem['Body']
        body_embs = self.sencences_to_embs(body_text)
        # Title part
        title_text = elem['Title']
        title_embs = self.sencences_to_embs(title_text)
        # Tags part
        tags_text = elem['Tags']
        tags_embs = get_sent_emb(' '.join(split_tags(tags_text)), self.embs_aggregation, model=self.model)



        try:
            target = elem['target']
            target_one_hot = np.zeros(self.num_classes, dtype=np.float32)
            target_one_hot[target] = 1
        except:
            target = 4
            target_one_hot = np.zeros(self.num_classes, dtype=np.float32)

        return {'body': body_embs.astype(np.float32), 'title': title_embs.astype(np.float32), 'tags': tags_embs.astype(np.float32), 'target': target, 'target_one_hot': target_one_hot, 'Id': elem.name}

    def __len__(self):
        return len(self.df)

    keys = ['body', 'title']

    def get_collate_fn(self, batch):
        keys = batch[0].keys()
        if isinstance(batch[0], collections.Mapping):
            result = {}
            for key in batch[0]:
                items = [d[key] for d in batch]
                if key not in self.keys:
                    items = default_collate(items)
                else:
                    items  = [torch.from_numpy(item) for item in items]
                result[key] = items
            return result
        else:
            return default_collate(batch)
