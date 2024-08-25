import pandas as pd
import numpy as np
import pytest

from melinda.models import ProbaformsSynthesizer
from probaforms.models import CVAE

def test_with_num_cols():
    # create data
    n = 100
    data = pd.DataFrame()
    data['num_1'] = np.random.rand(n)
    data['num_2'] = np.random.rand(n)

    num_cols = ['num_1', 'num_2']
    cat_cols = None
    lab_cols = None

    # fit synthesizer
    model = CVAE(latent_dim=10, hidden=(10,), lr=0.001, n_epochs=10)
    gen = ProbaformsSynthesizer(model, num_cols, cat_cols, lab_cols, cat_transform='OneHotEncoder')
    gen.fit(data)

    # sample synthetic data
    data_synth = gen.sample(10)


def test_with_num_cat_cols():
    # create data
    n = 100
    data = pd.DataFrame()
    data['num_1'] = np.random.rand(n)
    data['num_2'] = np.random.rand(n)
    data['cat_1'] = [str(i) for i in np.random.randint(0, 10, n)]
    data['cat_2'] = [str(i) for i in np.random.randint(0, 5, n)]

    num_cols = ['num_1', 'num_2']
    cat_cols = ['cat_1', 'cat_2']
    lab_cols = None

    # fit synthesizer
    model = CVAE(latent_dim=10, hidden=(10,), lr=0.001, n_epochs=10)
    gen = ProbaformsSynthesizer(model, num_cols, cat_cols, lab_cols, cat_transform='OneHotEncoder')
    gen.fit(data)

    # sample synthetic data
    data_synth = gen.sample(10)