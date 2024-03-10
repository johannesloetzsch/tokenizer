#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://www.kaggle.com/datasets/kenshoresearch/kensho-derived-wikimedia-data
"""


import kaggle

dataset_path = "../data/"
dataset_name = "kenshoresearch/kensho-derived-wikimedia-data"
d = kaggle.api.dataset_download_cli(dataset_name, path=dataset_path, unzip=False)