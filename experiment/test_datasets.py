# coding: utf-8
# import pdb


def test_datasets():
    from experiment.datasets import (
        Ricci, German, Adult, PropublicaRecidivism,
        PropublicaViolentRecidivism, preprocess,
        adversarial)

    dt = Ricci()
    dt = German()
    dt = Adult()
    dt = PropublicaRecidivism()
    dt = PropublicaViolentRecidivism()

    df = dt.load_raw_dataset()
    ans = preprocess(dt, df)

    ans = adversarial(dt, df, ratio=.95)
    # pdb.set_trace()
    return


def test_preprocessing():
    pass
