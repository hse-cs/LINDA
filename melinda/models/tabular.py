from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
import pandas as pd
import numpy as np
from .utils import LogitScaler

class ProbaformsSynthesizer(object):

    def __init__(self, generator, num_cols, cat_cols, lab_cols, cat_transform='OneHotEncoder'):
        self.generator = generator
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.lab_cols = lab_cols
        self.cat_transform = cat_transform
        self.y_ohe = None


    def fit(self, data):

        self.tot_cols = data.columns

        if self.cat_cols is not None:
            X_cat = data[self.cat_cols].values
            if self.cat_transform == 'OrdinalEncoder':
                self.ohe = make_pipeline(OrdinalEncoder(), LogitScaler(eps=0.1), StandardScaler())
                X_cat_ohe = self.ohe.fit_transform(X_cat)
            else:
                self.ohe = OneHotEncoder(sparse_output=False)
                X_cat_ohe = self.ohe.fit_transform(X_cat)
                X_cat_ohe += np.random.normal(0, 0.05, size=X_cat_ohe.shape)

        if self.num_cols is not None:
            X_num = data[self.num_cols].values
            self.ss = make_pipeline(LogitScaler(eps=0.1), StandardScaler())
            X_num_ss = self.ss.fit_transform(X_num)

        if self.lab_cols is not None:
            y = data[self.lab_cols].values
            self.ohe_lab = OneHotEncoder()
            y_ohe = self.ohe_lab.fit_transform(y).toarray()
        else:
            y_ohe = None

        if (self.cat_cols is not None) and (self.num_cols is not None):
            X = np.concatenate((X_num_ss, X_cat_ohe), axis=1)
        elif self.cat_cols is not None:
            X = X_cat_ohe
        elif self.num_cols is not None:
            X = X_num_ss

        self.y_ohe = y_ohe
        self.generator.fit(X, y_ohe)

        return self


    def sample(self, n_samples=10):

        if self.y_ohe is None:
            X_fake = self.generator.sample(n_samples)
        else:
            y_ohe_sample = resample(self.y_ohe, n_samples=n_samples)
            X_fake = self.generator.sample(y_ohe_sample)

        if self.num_cols is not None:
            X_fake_num = X_fake[:, :len(self.num_cols)]
        if self.cat_cols is not None:
            X_fake_cat = X_fake[:, len(self.num_cols):]

        if self.num_cols is not None:
            X_fake_num = self.ss.inverse_transform(X_fake_num)
        if self.cat_cols is not None:
            X_fake_cat = self.ohe.inverse_transform(X_fake_cat)
        if self.lab_cols is not None:
            y_sample = self.ohe_lab.inverse_transform(y_ohe_sample)

        data_fake = None
        if self.cat_cols is not None:
            data_fake_cat = pd.DataFrame(data=X_fake_cat, columns=self.cat_cols)
            data_fake = data_fake_cat
        if self.num_cols is not None:
            data_fake_num = pd.DataFrame(data=X_fake_num, columns=self.num_cols)
            if data_fake is not None:
                data_fake = pd.concat([data_fake, data_fake_num], axis=1)
            else:
                data_fake = data_fake_num
        if self.lab_cols is not None:
            data_fake_lab = pd.DataFrame(data=y_sample, columns=self.lab_cols)
            data_fake = pd.concat([data_fake, data_fake_lab], axis=1)
        data_fake = data_fake[self.tot_cols]

        return data_fake