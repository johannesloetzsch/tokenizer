#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Questions and Answers from Wikipedia
"""


from torch.utils.data import Dataset
import kaggle
import json
import joblib
from itertools import groupby

def unique(l):
    return [list(g)[0] for k,g in groupby(sorted(l))]


class NaturalQuestionsRaw(Dataset):
    
    def __init__(self, limit=None, dataset_path = "../data/"):
        dataset_name = "validmodel/the-natural-questions-dataset"
        kaggle.api.dataset_download_cli(dataset_name, path=dataset_path, unzip=True)
        dataset_file = dataset_path + kaggle.api.dataset_list_files(dataset_name).files[0].name

        with open(dataset_file) as f:
            datasets_all = json.loads('[' + ",".join(f.readlines()[:limit]) + ']')
            
        self._d = list(filter(lambda x: x['annotations'][0]['short_answers'], datasets_all))

    def __len__(self):
        return len(self._d)

    def __getitem__(self, idx):
        return self._d[idx]

# pickle.dump(d, open('../data/NaturalQuestionsRaw.pickle', 'wb'))
#joblib.dump(list(d), '../data/NaturalQuestionsRaw.joblib.gz', compress='gzip')



class NaturalQuestions(Dataset):
    """
    >>> #list(NaturalQuestions(max_long_answers=0))[1]
    ['https://en.wikipedia.org//w/index.php?title=The_Mother_(How_I_Met_Your_Mother)&amp;oldid=802354471',
     'how i.met your mother who is the mother',
     ['Tracy McConnell']]
    """
    
    def __init__(self, natural_questions_raw=None, max_long_answers=1, max_short_answers=None):
        if not natural_questions_raw:
            natural_questions_raw = NaturalQuestionsRaw()
        self._d = natural_questions_raw
        self.max_long_answers = max_long_answers
        self.max_short_answers = max_short_answers

    def __len__(self):
        return len(self._d)

    def __getitem__(self, idx):
        entry = self._d[idx]
        document_url = entry['document_url']
        question_text = entry['question_text']
        tokens = entry['document_text'].split(" ")
        annotations = entry['annotations'][0]
        answer_ranges = [[a['start_token'], a['end_token']] for a in
                         [annotations['long_answer']][:self.max_long_answers] +
                         annotations['short_answers'][:self.max_short_answers]]
        answers = [' '.join(tokens[start_token:end_token]) for [start_token,end_token] in answer_ranges]
        return [document_url, question_text, answers]

# pickle.dump(list(NaturalQuestions(d)), open('../data/NaturalQuestions.pickle', 'wb'))
#joblib.dump(list(NaturalQuestions(d)), '../data/NaturalQuestions.joblib.gz', compress='gzip')


class Wikipages(Dataset):
    """Articles from en.wikipedia.org used by NaturalQuestions"""

    def __init__(self, natural_questions_raw=None):
        if not natural_questions_raw:
            natural_questions_raw = NaturalQuestionsRaw()
        entries_all = [[e['document_url'], e['document_text']] for e in natural_questions_raw]
        self._entries = unique(entries_all)

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, idx):
        return self._entries[idx]


if __name__ == "__main__":
    import doctest
    print(doctest.testmod())