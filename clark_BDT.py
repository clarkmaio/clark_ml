import pandas as pd
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
        self.mdl_config = mdl_config
        self.mdl_constructor()

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

        print('>>>>> Starting grid search...')
        param_grid = self.mdl_config['grid_search']['param_grid']
        params = self.mdl_config['grid_search']['params']

        gs = GridSearchCV(estimator=self.mdl, param_grid = param_grid, **params)
        gs.fit(X, y)
        print('>>>>> Best params: {}'.format(gs.best_params_))

        return gs.best_params_

    def shap(self, X, y):
        self.explainer = shap.TreeExplainer(self.mdl)
        self.shap_values = self.explainer.shap_values(X)

    def force_plot(self, X):
        # TODO: Take as input X and sycronise to sha_value via pandas index
        shap.force_plot(self.explainer.expected_value, self.shap_values[0, :], X, matplotlib=True)

if __name__ == '__main__':
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split

    mdl_config = {
        'params':{
            'n_estimators': 100,
            'max_depth':5
        },

        'grid_search':{
            'param_grid':{
                'n_estimators': [100, 200, 300],
                'max_depth': [2,3,4],
            },
            'params':{
                'verbose' : 0, # 2 for full messages
            }
        }
    }

    # Train/pred
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8)
    mdl = clark_BDT(mdl_config = mdl_config)
    mdl.fit(X_train, y_train)
    y_pred = mdl.predict(X_test)

    # Gridsearch
    # best_params = mdl.grid_search(X_train, y_train)

    # Shap
    mdl.shap(X_test, y_test)
    mdl.force_plot(X_test)