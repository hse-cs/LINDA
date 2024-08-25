# Welcome to LINDA

``MELINDA`` is a python library for creating tabular synthetic data. 
It uses various generative models in artificial intelligence 
to learn statistical properties from your real data and 
use them to generate synthetic data.

## Installation
```python
git clone https://github.com/hse-cs/LINDA.git
cd LINDA
pip install -e .
```
or
```python
poetry install
```

## Basic usage
The following code snippet creates an example of real data, fits a generative model, and samples synthetic data.
```python
import numpy as np
import pandas as pd
from melinda.models import ProbaformsSynthesizer
from probaforms.models import CVAE

# generate an example of real data
n = 100
data_real = pd.DataFrame()
data_real['col_1'] = np.random.rand(n)
data_real['col_2'] = np.random.rand(n)
data_real['col_3'] = [str(i) for i in np.random.randint(0, 10, n)]
data_real['col_4'] = [str(i) for i in np.random.randint(0, 5, n)]

num_cols = ['col_1', 'col_2']
cat_cols = ['col_3', 'col_4']
lab_cols = None

# fit a generative model
model = CVAE(latent_dim=10, hidden=(10,), lr=0.001, n_epochs=10)
gen = ProbaformsSynthesizer(model, num_cols, cat_cols, lab_cols, cat_transform='OneHotEncoder')
gen.fit(data_real)

# sample synthetic data
data_synthetic = gen.sample(n_samples=10)
data_synthetic.head()
```