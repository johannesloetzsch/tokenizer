#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://huggingface.co/datasets/BeIR/dbpedia-entity-generated-queries
"""

from datasets import load_dataset


x = load_dataset("BeIR/dbpedia-entity-generated-queries", split='train')