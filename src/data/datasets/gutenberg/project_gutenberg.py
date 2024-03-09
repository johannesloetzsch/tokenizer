#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://huggingface.co/datasets/manu/project_gutenberg
"""

from datasets import load_dataset

#de = load_dataset("manu/project_gutenberg", split="de")


def gutenberg_meta(book, lang=""):
    """
    >>> de = load_dataset('manu/project_gutenberg', split='de')
    >>> [gutenberg_meta(book, 'de') for book in list(de)[:2]]
    [['44722-0', 'de', 'Eduard Bernstein', 'Ferdinand Lassalle'], ['24288-8', 'de', 'Rainer Maria Rilke', 'Das Stunden-Buch']]
    """
    try:
        title_and_author = book["text"].split("EBook of ")[1].split("\n\n")[0]
    except:
        title_and_author = ""

    try:
        title, author = title_and_author.split(", by ", maxsplit=1)
    except:
        title, author = [None, None]

    return [book["id"], lang, author, title]


if __name__ == "__main__":
    import doctest
    print(doctest.testmod())