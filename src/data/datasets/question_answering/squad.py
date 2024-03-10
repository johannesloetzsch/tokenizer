#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://huggingface.co/datasets/rajpurkar/squad
https://huggingface.co/datasets/rajpurkar/squad_v2
"""

from datasets import load_dataset

squad = load_dataset("squad", split="train")
