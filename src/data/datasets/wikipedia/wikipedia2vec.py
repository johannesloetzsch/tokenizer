# -*- coding: utf-8 -*-

import numpy as np
import joblib
from marisa_trie import RecordTrie, Trie

class Dictionary:
    def __init__(
        self,
        word_dict: Trie,
        entity_dict: Trie,
        redirect_dict: RecordTrie,
        word_stats: np.ndarray,
        entity_stats: np.ndarray,
        vectors: np.ndarray
    ):
        self._word_dict = word_dict
        self._word_dict_rev = dict(zip([i[1] for i in word_dict.items()],
                                       word_dict.keys()))
        self._entity_dict = entity_dict
        self._redirect_dict = redirect_dict
        self._word_stats = word_stats[: len(self._word_dict)]
        self._entity_stats = entity_stats[: len(self._entity_dict)]

        self._vectors = vectors
    
    def load(file: str):
        dataset = joblib.load(file)
        vectors = dataset["syn0"]
        dictionary = dataset["dictionary"]

        word_dict = Trie()
        entity_dict = Trie()
        redirect_dict = RecordTrie("<I")

        word_dict.frombytes(dictionary["word_dict"])
        entity_dict.frombytes(dictionary["entity_dict"])
        redirect_dict.frombytes(dictionary["redirect_dict"])

        word_stats = dictionary["word_stats"]
        entity_stats = dictionary["entity_stats"]

        return Dictionary(word_dict, entity_dict, redirect_dict, word_stats, entity_stats, vectors)

    def get_word(self, idx: int):
        return self._word_dict_rev[idx]

    def get_words_by_prefix(self, word_prefix: str):
        return self._word_dict.iterkeys(word_prefix)

    def get_vector(self, word: str):
        index = self._word_dict.get(word)
        if index:
            return self._vectors[index]

    def __getitem__(self, word: str):
        return self.get_vector(word)

    def __keys__(self):
        return self._word_dict.keys()

    def most_similar(self, vec: np.ndarray, k=10):
        """TODO: support Euclidean and Cosine Distance"""
        dst = np.dot(self._vectors, vec) / np.linalg.norm(self._vectors, axis=1) / np.linalg.norm(vec)
        indexes = np.argsort(-dst)
        return [(self._word_dict_rev.get(i), dst[i]) for i in indexes[:k]]


def dict_by_lang(lang="de"):
    file_by_lang = { "de": "../data/wiki2vec/dewiki_20180420_100d.pkl.bz2",
                     "en": "../data/wiki2vec/enwiki_20180420_win10_100d.pkl.bz2" }

    file = file_by_lang[lang]
    print("Load {} ({})…".format(lang, file))
    return Dictionary.load(file)

if __name__ == '__main__':
    de = dict_by_lang()
    print(de.most_similar(de["bruder"] + de["frau"] - de["mann"], 1))