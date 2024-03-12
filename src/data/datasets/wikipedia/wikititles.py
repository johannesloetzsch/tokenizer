#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://huggingface.co/datasets/wmt/wikititles
"""

from datasets import load_dataset

de_en = load_dataset("wmt/wikititles", 'de-en', split=['train'])[0]

def languages(translations):
    return translations[0]['translation'].keys()

def translations_processed(translations):
    lang1, lang2 = languages(translations)
    return [[t['translation'][lang1].lower(), t['translation'][lang2].lower()]
            for t in translations]


def translations_with_vectors(translations, vectors_lang1, vectors_lang2, verbose=True):
    """
    >>> de_en_small, de, en = load_datasets(1000, verbose=False)
    >>> de_en_vectors = translations_with_vectors(de_en_small, de, en, verbose=False)
    >>> len(de_en_vectors)
    457
    >>> de_en_vectors[11]
    ['arzt', 'physician']
    >>> len(de['arzt'])
    100
    >>> len(en['physician'])
    100
    """
    if verbose:
        print('Create de_en_vectors…')

    vectors_lang1_keys = vectors_lang1._word_dict.keys()
    vectors_lang2_keys = vectors_lang2._word_dict.keys()

    return [[t1, t2]
            for [t1, t2] in translations_processed(translations)
            if t1 in vectors_lang1_keys and t2 in vectors_lang2_keys]


def load_datasets(limit=None, verbose=True, g=None):
    g = g or globals()

    if verbose:
        print('Create de_en_small…')
    de_en_small = g.get('de_en_small') or list(de_en)[:limit]

    from data.datasets.wikipedia.wikipedia2vec import dict_by_lang
    de = g.get('de') or dict_by_lang('de')
    en = g.get('en') or dict_by_lang('en')

    return [de_en_small, de, en]

if __name__ == '__main__':
    de_en_small, de, en = load_datasets(1000)
    de_en_vectors = globals().get('de_en_vectors') or translations_with_vectors(de_en_small, de, en)

    import doctest
    print('Run tests…')
    print(doctest.testmod())