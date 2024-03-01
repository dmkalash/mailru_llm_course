import re
import pickle
from itertools import chain
from datetime import datetime
from collections import defaultdict
from abc import ABC, abstractmethod

import numpy as np

from typing import List, Dict, Optional, Iterable, Tuple

# from tqdm.notebook import tqdm

class Tokenizer:
    def __init__(self,
                 token_pattern: str = '\w+|[\!\?\,\.\-\:]',
                 eos_token: str = '<EOS>',
                 pad_token: str = '<PAD>',
                 unk_token: str = '<UNK>'):
        self.token_pattern = token_pattern
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token

        self.special_tokens = [self.eos_token, self.pad_token, self.unk_token]
        self.vocab = None
        self.inverse_vocab = None

    def text_preprocess(self, input_text: str) -> str:
        """ Предобрабатываем один текст """
        # input_text = ... # приведение к нижнему регистру
        input_text = input_text.lower()
        input_text = re.sub('\s+', ' ', input_text)  # унифицируем пробелы
        input_text = input_text.strip()
        return input_text

    def build_vocab(self, corpus: List[str]) -> None:
        assert len(corpus)
        all_tokens = set()
        for text in corpus:
            all_tokens |= set(self._tokenize(text, append_eos_token=False))
        self.vocab = {elem: ind for ind, elem in enumerate(all_tokens)}
        special_tokens = [self.eos_token, self.unk_token, self.pad_token]
        for token in special_tokens:
            self.vocab[token] = len(self.vocab)
        self.inverse_vocab = {ind: elem for elem, ind in self.vocab.items()}
        return self

    def _tokenize(self, text: str, append_eos_token: bool = True) -> List[str]:
        text = self.text_preprocess(text)
        tokens = re.findall(self.token_pattern, text)
        if append_eos_token:
            tokens.append(self.eos_token)
        return tokens

    def encode(self, text: str, append_eos_token: bool = True) -> List[str]:
        """ Токенизируем текст """
        tokens = self._tokenize(text, append_eos_token)
        ids = [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]
        return ids

    def decode(self, input_ids: Iterable[int], remove_special_tokens: bool = False) -> str:
        assert len(input_ids)
        assert max(input_ids) < len(self.vocab) and min(input_ids) >= 0
        tokens = []
        for ind in input_ids:
            token = self.inverse_vocab[ind]
            if remove_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)
        text = ' '.join(tokens)
        return text

    def save(self, path: str) -> bool:
        data = {
            'token_pattern': self.token_pattern,
            'eos_token': self.eos_token,
            'pad_token': self.pad_token,
            'unk_token': self.unk_token,
            'special_tokens': self.special_tokens,
            'vocab': self.vocab,
            'inverse_vocab': self.inverse_vocab,
        }

        with open(path, 'wb') as fout:
            pickle.dump(data, fout)

        return True

    def load(self, path: str) -> bool:
        with open(path, 'rb') as fin:
            data = pickle.load(fin)

        self.token_pattern = data['token_pattern']
        self.eos_token = data['eos_token']
        self.pad_token = data['pad_token']
        self.unk_token = data['unk_token']
        self.special_tokens = data['special_tokens']
        self.vocab = data['vocab']
        self.inverse_vocab = data['inverse_vocab']


class GenerationConfig:
    def __init__(self, **kwargs):
        """
        Тут можно задать любые параметры и их значения по умолчанию
        Значения для стратегии декодирования decoding_strategy: ['max', 'top-p']
        """
        self.temperature = kwargs.pop("temperature", 1.0)
        self.max_tokens = kwargs.pop("max_tokens", 32)
        self.sample_top_p = kwargs.pop("sample_top_p", 0.9)
        self.decoding_strategy = kwargs.pop("decoding_strategy", 'max')
        self.remove_special_tokens = kwargs.pop("remove_special_tokens", False)
        self.validate()

    def validate(self):
        """ Здесь можно валидировать параметры """
        if not (1.0 > self.sample_top_p > 0):
            raise ValueError('sample_top_p')
        if self.decoding_strategy not in ['max', 'top-p']:
            raise ValueError('decoding_strategy')


# np.random.seed(42)


class StatLM:  # (ModelTemplate):
    def __init__(self,
                 tokenizer: Tokenizer,
                 context_size: int = 2,
                 alpha: float = 0.1
                 ):

        assert context_size >= 2

        self.context_size = context_size
        self.tokenizer = tokenizer
        self.alpha = alpha

        self.n_gramms_stat = defaultdict(int)
        self.nx_gramms_stat = defaultdict(int)

    def get_token_by_ind(ind: int) -> str:
        return self.tokenizer.vocab.get(ind)

    def get_ind_by_token(token: str) -> int:
        return self.tokenizer.inverse_vocab.get(token, self.tokenizer.inverse_vocab[self.unk_token])

    def train(self, train_texts: List[str]):
        for sentence in tqdm(train_texts, desc='train lines'):
            sentence_ind = self.tokenizer.encode(sentence)
            for i in range(len(sentence_ind) - self.context_size):
                seq = tuple(sentence_ind[i: i + self.context_size - 1])
                self.n_gramms_stat[seq] += 1

                seq_x = tuple(sentence_ind[i: i + self.context_size])
                self.nx_gramms_stat[seq_x] += 1

            seq = tuple(sentence_ind[len(sentence_ind) - self.context_size:])
            self.n_gramms_stat[seq] += 1

    def sample_token(self,
                     token_distribution: np.ndarray,
                     generation_config: GenerationConfig) -> int:
        if generation_config.decoding_strategy == 'max':
            return token_distribution.argmax()
        elif generation_config.decoding_strategy == 'top-p':
            token_distribution = sorted(list(zip(token_distribution, np.arange(len(token_distribution)))),
                                        reverse=True)
            total_proba = 0.0
            tokens_to_sample = []
            tokens_probas = []
            for token_proba, ind in token_distribution:
                tokens_to_sample.append(ind)
                tokens_probas.append(token_proba)
                total_proba += token_proba
                if total_proba >= generation_config.sample_top_p:
                    break
            # для простоты отнормируем вероятности, чтобы суммировались в единицу
            tokens_probas = np.array(tokens_probas) / generation_config.temperature
            tokens_probas = tokens_probas / tokens_probas.sum()
            return np.random.choice(tokens_to_sample, p=tokens_probas)
        else:
            raise ValueError(f'Unknown decoding strategy: {generation_config.decoding_strategy}')

    def save_stat(self, path: str) -> bool:
        stat = {
            'n_gramms_stat': self.n_gramms_stat,
            'nx_gramms_stat': self.nx_gramms_stat,
            'context_size': self.context_size,
            'alpha': self.alpha
        }
        with open(path, 'wb') as fout:
            pickle.dump(stat, fout)

        return True

    def load_stat(self, path: str) -> bool:
        with open(path, 'rb') as fin:
            stat = pickle.load(fin)

        self.n_gramms_stat = stat['n_gramms_stat']
        self.nx_gramms_stat = stat['nx_gramms_stat']
        self.context_size = stat['context_size']
        self.alpha = stat['alpha']

        return True

    def get_stat(self) -> Dict[str, Dict]:

        n_token_stat, nx_token_stat = {}, {}
        for token_inds, count in self.n_gramms_stat.items():
            n_token_stat[self.tokenizer.decode(token_inds)] = count

        for token_inds, count in self.nx_gramms_stat.items():
            nx_token_stat[self.tokenizer.decode(token_inds)] = count

        return {
            'n gramms stat': self.n_gramms_stat,
            'n+1 gramms stat': self.nx_gramms_stat,
            'n tokens stat': n_token_stat,
            'n+1 tokens stat': nx_token_stat,
        }

    def _get_next_token(self,
                        tokens: List[int],
                        generation_config: GenerationConfig) -> (int, str):
        denominator = self.n_gramms_stat.get(tuple(tokens), 0) + self.alpha * len(self.tokenizer.vocab)
        numerators = []
        for ind in self.tokenizer.inverse_vocab:
            numerators.append(self.nx_gramms_stat.get(tuple(tokens + [ind]), 0) + self.alpha)

        token_distribution = np.array(numerators) / denominator
        max_proba_ind = self.sample_token(token_distribution, generation_config)

        next_token = self.tokenizer.inverse_vocab[max_proba_ind]

        return max_proba_ind, next_token

    def generate_token(self,
                       text: str,
                       generation_config: GenerationConfig
                       ) -> Dict:
        tokens = self.tokenizer.encode(text, append_eos_token=False)
        tokens = tokens[-self.context_size + 1:]

        max_proba_ind, next_token = self._get_next_token(tokens, generation_config)

        return {
            'next_token': next_token,
            'next_token_num': max_proba_ind,
        }

    def generate_text(self, text: str,
                      generation_config: GenerationConfig
                      ) -> Dict:

        all_tokens = self.tokenizer.encode(text, append_eos_token=False)
        tokens = all_tokens[-self.context_size + 1:]

        next_token = None
        while next_token != self.tokenizer.eos_token and len(all_tokens) < generation_config.max_tokens:
            max_proba_ind, next_token = self._get_next_token(tokens, generation_config)
            all_tokens.append(max_proba_ind)
            tokens = all_tokens[-self.context_size + 1:]

        new_text = self.tokenizer.decode(all_tokens, generation_config.remove_special_tokens)

        finish_reason = 'max tokens'
        if all_tokens[-1] == self.tokenizer.vocab[self.tokenizer.eos_token]:
            finish_reason = 'end of text'

        return {
            'all_tokens': all_tokens,
            'total_text': new_text,
            'finish_reason': finish_reason
        }

    def generate(self, text: str, generation_config: Dict) -> str:
        return self.generate_text(text, generation_config)['total_text']


def construct_model():
    """
    задаем все параметры генерации, инициализируем модель, подгружаем в нее все статистики и токенизатор.
    код StatLM и Tokenizer должен совпадать с кодом из ноутбука, в которой происходило сохранение параметров модели и токенизатора
    """
    config = {
        'temperature': 1.0,
        'max_tokens': 32,
        'sample_top_p': 0.9,
        'decoding_strategy': 'top-p',
    }

    stat_lm_path = 'models/stat_lm/stat_lm.pkl'
    tokenizer_path = 'models/stat_lm/tokenizer.pkl'

    tokenizer = Tokenizer()
    tokenizer.load(tokenizer_path)

    stat_lm = StatLM(tokenizer)
    stat_lm.load_stat(stat_lm_path)

    generation_config = GenerationConfig(temperature=config['temperature'],
                                         max_tokens=config['max_tokens'],
                                         sample_top_p=config['sample_top_p'],
                                         decoding_strategy=config['decoding_strategy'],
                                         remove_special_tokens=True)

    kwargs = {'generation_config': generation_config}
    return stat_lm, kwargs
