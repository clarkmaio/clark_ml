import panda as pd
import numpy as np
import pickle
from pprint import pprint

import shap
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from plotly.offline import plot
from plotly import graph_objs as go

class clark_BDT(object):
    """
    Wrap of XGBRegressor
    mdl_config is a dictionary containing te keys:
        - params: it will be passed as kwargs to XGBRegressor constructor
        - grid_search: dict with keys:
            - domain
            - params
    """
    def __init__(self, mdl_config):
        self.mdl_config

    def mdl_constructor(self):
        self.mdl = XGBRegressor(**self.mdl_config['params'])

    def fit(self, X, y):
        self.mdl.fit(X, y)

    def predict(self, X):
        y = self.mdl.predict(X)

        if isinstance(X, pd.DataFrame):
            y = pd.DataFrame(y, index = X.index, columns = ['pred'])

        return y

    def save(self, path):
        """
        Save mdl object.
        path is the full path: /.../mdl.p
        """
        return 0

    def load(self, path):
        """
        Save mdl object.
        path is the full path: /.../mdl.p
        """
        return 0

    def grid_search(self, X, y):

        print '>>>>> Starting grid search...'
        param_grid = self.mdl_config['grid_search']['param_grid']
        params = self.mdl_config['grid_search']['params']

        gs = GridSearchCV(estimator=self.mdl, param_grid = param_grid, verbose = 2, **params)
        gs.fit(X, y)
        print 'Best params: {}'.format(gs.best_params_)

        return gs.best_params_

    def shap(self, X, y):
        return 0

    def force_plot(self, x):
        return 0



