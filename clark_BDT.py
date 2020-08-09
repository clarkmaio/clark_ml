__author__ = 'clarkmaio'

import pandas as pd
import pickle
import os


import shap
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

class clark_BDT(object):
    """
    Wrap of XGBRegressor
    mdl_config is a dictionary containing te keys:
        - params: it will be passed as kwargs to XGBRegressor constructor
        - grid_search: dict with keys:
            - domain
            - params

            - mdl_path: path where model will be saved
    """
    def __init__(self, mdl_config):
        self.mdl_config = mdl_config
        self._initialize_mdl()

    def _initialize_mdl(self):
        """
        Initilize model.
        Create mdl_path folder
        """
        if not os.path.exists(self.mdl_config['mdl_path']):
            os.mkdir(self.mdl_config['mdl_path'])

        # Upload grid_search params
        gs_path = os.path.join(self.mdl_config['mdl_path'], 'BDT_grid_search.p')
        if self.mdl_config['use_gs_params'] and os.path.exists(gs_path):
            self.mdl_config['params'] = pickle.load(open(gs_path, 'rb'))

        self.mdl = XGBRegressor(**self.mdl_config['params'])

    def fit(self, X, y, **kwargs):
        print('>>>>> Fitting BDT mdl...')
        self.mdl.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        y = self.mdl.predict(X, **kwargs)

        if isinstance(X, pd.DataFrame):
            y = pd.DataFrame(y, index = X.index, columns = ['pred'])

        return y

    def save(self):
        """
        Save mdl object.
        path is the full path: /.../mdl.p
        """
        print('>>>>> Saving BDT mdl...')
        mdl_path = os.path.join(self.mdl_config['mdl_path'], 'mdl_BDT.p')
        pickle.dump(self.mdl, open(mdl_path, 'wb'))


    def load(self):
        """
        Save mdl object.
        path is the full path: /.../mdl.p
        """
        print('>>>>> Loading BDT mdl...')
        mdl_path = os.path.join(self.mdl_config['mdl_path'], 'mdl_BDT.p')
        self.mdl = pickle.load(open(mdl_path, 'rb'))

    # ---------------------- GRID SEARCH ---------------------------
    # TODO option for Random grid search
    def grid_search(self, X, y):
        """Perform grid search and save dictionary"""

        print('>>>>> Starting grid search...')
        param_grid = self.mdl_config['grid_search']['param_grid']
        params = self.mdl_config['grid_search']['params']

        gs = GridSearchCV(estimator=self.mdl, param_grid = param_grid, **params)
        gs.fit(X, y)
        print('>>>>> Best params: {}'.format(gs.best_params_))

        gs_path = os.path.join(self.mdl_config['mdl_path'], 'BDT_grid_search.p')
        pickle.dump(gs.best_params_,  open(gs_path, 'wb'))

        return gs.best_params_

    def load_grid_search(self):
        """Load grid search params if pickle exist"""

    # ---------------------- SHAP ---------------------------
    def _initialize_shap(self, *args, **kwargs):
        """
        Inizialize shap object once you have fitted mdl
        """
        print('>>>>> Initialize SHAP mdl...')
        self.shap_explainer = shap.TreeExplainer(self.mdl)
        self.shap_values = self.shap_explainer.shap_values(*args, **kwargs)

    def shap_summary_plot(self, X):
        """
        Plot features importance
        """
        shap.summary_plot(self.shap_values, X)
        plt.savefig(os.path.join(self.mdl_config['mdl_path'], 'BDT_shap_values.png'))
        plt.close()

        shap.summary_plot(self.shap_values, X, plot_type='bar')
        plt.savefig(os.path.join(self.mdl_config['mdl_path'], 'BDT_shap_importance.png'))
        plt.close()

    # ---------------------- PLOT ---------------------------
    def loss_plot(self):
        """
        Plot loss for each boosting round.
        Usefull if eval_set argument passed in .fit phase
        """

        train_curve = pd.DataFrame(self.mdl.evals_result()['validation_0'])
        test_curve =  pd.DataFrame(self.mdl.evals_result()['validation_1'])

        # Plot for each metric
        for m in train_curve.columns:
            fig, ax = plt.subplots()
            ax.plot(train_curve.index, train_curve[m], label = 'Train')
            ax.plot(test_curve.index, test_curve[m], label='Validation')
            ax.legend()
            plt.ylabel('{}'.format(m.upper()))
            plt.xlabel('Boosting rounds')
            plt.title('{} curve'.format(m.upper()))

            plot_path = os.path.join(self.mdl_config['mdl_path'], 'BDT_{}_curve.png'.format(m.upper()))
            plt.savefig(plot_path)

            plt.close()



if __name__ == '__main__':
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error

    mdl_config = {
        'mdl_path': 'C:\\Users\\pc\\workspace\\clark_ml\\BDT_mdl_path',
        'use_gs_params': True,
        'params':{
            'n_estimators': 100,
            'max_depth':5
        },

        'grid_search':{
            'param_grid':{
                'n_estimators': [100,300,500,700,1000],
                'max_depth': [4,6,8,10],
                'subsample': [.5, .75, 1]
            },
            'params':{
                'verbose' : 2, # 2 for full messages
                'cv': 4,
                'n_jobs':2
            }
        }
    }

    # Train/pred
    data = load_boston(return_X_y=False)
    X = pd.DataFrame(data['data'], columns=data['feature_names'])
    y = pd.DataFrame(data['target'], columns = ['target'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8)
    mdl = clark_BDT(mdl_config = mdl_config)
    mdl.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_test, y_test)], eval_metric = ['mae', 'rmse'])
    y_trai_pred = mdl.predict(X_train)
    y_pred = mdl.predict(X_test)

    # Score
    print('MAE train: {}'.format(mean_absolute_error(y_train, y_trai_pred)))
    print('MAE test: {}'.format(mean_absolute_error(y_test, y_pred)))

    # Plot loss
    mdl.loss_plot()

    # # Gridsearch
    # best_params = mdl.grid_search(X_train, y_train)

    # Shap
    mdl._initialize_shap(X_train, y_train)
    mdl.shap_summary_plot(X_train)

    # Save mdl
    #mdl.save()