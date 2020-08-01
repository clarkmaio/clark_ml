from pygam import GAM, l,f,s,te
import pandas as pd
import numpy as np
import pickle
import logging

import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class clark_GAM(object):
    def __init__(self, **kwargs):
        self.mdl = GAM(**kwargs)

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.mdl.fit(X, y)

    def predict(self, X):

        self.check_data(self.X_train, X)
        y_pred = self.mdl.predict(X)
        return y_pred

    def check_data(self, X_train, X):

        if X_train.shape[1] != X.shape[1]:
            logger.error('Different shape of input data between fit and predict')
            raise RuntimeError

        if isinstance(X_train, pd.DataFrame) and isinstance(X, pd.DataFrame):
            if np.any(X_train.columns != X.columns):
                logger.error('fit and predict data have different columns names.\nX_train: {}\nX: {}'.format(X_train.columns, X.columns))
                raise RuntimeError

    def save(self, pathname):
        # Save mdl
        return 0

    def load(self, pathname):
        # Load model
        return 0

    def plot_components(self, term_ind, term_name, save= False, filename = 'GAM_components.png'):
        """
        term_ind: list of index of corresponding terms to plot
        term_name: list of names of corresponding terms. Will be used as plot titles
        save = True to save .png image
        """
        return 0

    def plot_surface(self, term_ind, term_name = ['x', 'v'], save = False, filename = 'GAM_surface.png'):
        """
        term_ind: index of corresponding tensor term
        term_name: list of names of features fitted by tensor
        save = True to save .png image
        """
        return 0