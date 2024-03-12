#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from data.datasets.wikipedia.wikititles import load_datasets, translations_with_vectors
import torch
import torch.nn.functional as F

def inv(v):
    dim = v.shape[-1]
    return torch.tensor(v)  \
                .reshape(dim,1)  \
                .pinverse()  \
                .reshape(dim)  \
                .tolist()

def test_inverse_embedding(de_en_small, de, en, activation=lambda t: t):
    """
    Encode one-hot to vector (oh2v) and vice versa (v2oh).
    Encoding/decoding should yield the correct results in the other representation.
    Since the matrices are not square (different dimensions of representations), only a pseudoinverse can be calculated and a loss should be expected.

    >>> de_en_small, de, en = load_datasets(1000, g=globals(), verbose=False)
    >>> test_inverse_embedding(de_en_small, de, en)
    inverse error: 9.9e-02
    True v_ == v
    True idx: 11, hotness: 1.00
    loss: 5.8e-02

    Using an appropriate activation function, the loss can be reduced:
    >>> test_inverse_embedding(de_en_small, de, en, activation = lambda t: t.pow(2))
    inverse error: 2.1e-02
    True v_ == v
    True idx: 11, hotness: 1.00
    loss: 6.0e-03
    >>> test_inverse_embedding(de_en_small, de, en, activation = lambda t: t.pow(2).round())
    inverse error: 1.2e-02
    True v_ == v
    True idx: 11, hotness: 1.00
    loss: 0.0e+00
    """

    de_en_vectors = globals().get('de_en_vectors') or translations_with_vectors(de_en_small, de, en)
    concepts_count = len(de_en_vectors)

    arzt_idx = 11
    arzt_oh = F.one_hot(torch.tensor([arzt_idx]), concepts_count).float().reshape([concepts_count])
    arzt_v = torch.tensor(de['arzt'])

    oh2v_de = torch.tensor([de[t_de] for [t_de, t_en] in de_en_vectors])
    v2oh_de = torch.tensor([inv(de[t_de]) for [t_de, t_en] in de_en_vectors]).T
    #v2oh_de = oh2v_de.pinverse()  ## alternative for lower loss, but also lower hotness

    print("inverse error: {:.1e}".format(F.mse_loss(activation(oh2v_de @ v2oh_de), torch.eye(oh2v_de.shape[0]))))

    arzt_v_ = arzt_oh @ oh2v_de
    print(bool(torch.all(torch.eq(arzt_v_ , arzt_v))), "v_ == v")

    arzt_oh_ = activation(arzt_v @ v2oh_de)
    print(bool(arzt_oh_.argmax() == arzt_idx), "idx: {}, hotness: {:.2f}".format(arzt_oh_.argmax(), torch.max(arzt_oh_)))
    print("loss: {:.1e}".format(F.mse_loss( arzt_oh_, arzt_oh )))


if __name__ == '__main__':
    de_en_small, de, en = load_datasets(1000, g=globals())
    test_inverse_embedding(de_en_small, de, en)


    import doctest
    print('Run tests…')
    print(doctest.testmod())