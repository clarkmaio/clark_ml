import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
import shap
from sklearn.model_selection import GridSearchCV

class clark_BDT2(XGBRegressor):
    '''
    Hereditary xgboost class.

    +GridSearch
    +Shape
    +Loss plot (wrt n_estimators)
    '''

    # ---------------------- GRID SEARCH ---------------------------
    def grid_search(self, X, y, param_grid, gs_params, path):
        """
        Perform grid search and save dictionary.

        Exmaple input:
        param_grid = {'n_estimators': [100, 200, 300],
                      'max_depth': [4,6,8,10]}
        gs_params = {'cv': 4,
                     'n_jobs': 2,
                     'verbose': 2}
        """

        print('>>>>> Starting grid search...')
        print('Grid: {}'.format(param_grid))
        print('GS params: {}'.format(gs_params))

        gs = GridSearchCV(estimator=self, param_grid=param_grid, **gs_params)
        gs.fit(X, y)
        print('>>>>> Best params: {}'.format(gs.best_params_))


        gs_path = os.path.join(path, 'BDT_grid_search.p')
        print('>>>>> Dump opt params: {}'.format(gs_path))
        pickle.dump(gs.best_params_, open(gs_path, 'wb'))

        return gs.best_params_

    def load_grid_search(self, path):
        """Load grid search params if pickle exist"""
        gs_best_params = pickle.load(open(path, 'rb'))
        return gs_best_params


    # ---------------------- SHAP ---------------------------
    def _initialize_shap(self, *args, **kwargs):
        """
        Inizialize shap object once you have fitted mdl
        """
        print('>>>>> Initialize SHAP mdl...')
        self.shap_explainer = shap.TreeExplainer(self)
        self.shap_values = self.shap_explainer.shap_values(*args, **kwargs)

    def shap_summary_plot(self, X, path):
        """
        Plot features importance
        """
        print('>>>>> Generate shap summary plot...')
        shap.summary_plot(self.shap_values, X)
        plot_path = os.path.join(path, 'BDT_shap_values.png')
        plt.savefig(plot_path)
        print(plot_path)
        plt.close()

        shap.summary_plot(self.shap_values, X, plot_type='bar')
        plot_path = os.path.join(path, 'BDT_shap_importance.png')
        plt.savefig(plot_path)
        print(plot_path)
        plt.close()

    # ---------------------- PLOT ---------------------------

    def loss_plot(self, path):
        """
        Plot loss for each boosting round.
        Usefull if eval_set argument passed in .fit phase
        """

        train_curve = pd.DataFrame(self.evals_result()['validation_0'])
        test_curve = pd.DataFrame(self.evals_result()['validation_1'])

        # Plot for each metric
        for m in train_curve.columns:
            fig, ax = plt.subplots()
            ax.plot(train_curve.index, train_curve[m], label='Train')
            ax.plot(test_curve.index, test_curve[m], label='Validation')
            ax.legend()
            plt.ylabel('{}'.format(m.upper()))
            plt.xlabel('Boosting rounds')
            plt.title('{} curve'.format(m.upper()))
            plt.grid()

            plot_path = os.path.join(path, 'BDT_{}_curve.png'.format(m.upper()))
            plt.savefig(plot_path)

            plt.close()





if __name__ == '__main__':
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error

    mdl_config = {
        'mdl_path': 'C:\\Users\\pc\\Desktop',
        'use_gs_params': True,
        'params': {
            'n_estimators': 150,
            'max_depth': 5
        },

        'grid_search': {
            'param_grid': {
                'n_estimators': [100, 300, 500, 700, 1000],
                'max_depth': [4, 6, 8, 10],
                'subsample': [.5, .75, 1]
            },
            'params': {
                'verbose': 2,  # 2 for full messages
                'cv': 4,
                'n_jobs': 2
            }
        }
    }

    # Train/pred
    data = load_boston(return_X_y=False)
    X = pd.DataFrame(data['data'], columns=data['feature_names'])
    y = pd.DataFrame(data['target'], columns=['target'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8)
    mdl = clark_BDT2(**mdl_config['params'])
    mdl.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric=['mae', 'rmse'])
    y_trai_pred = mdl.predict(X_train)
    y_pred = mdl.predict(X_test)

    # Score
    print('MAE train: {}'.format(mean_absolute_error(y_train, y_trai_pred)))
    print('MAE test: {}'.format(mean_absolute_error(y_test, y_pred)))

    # Plot loss
    mdl.loss_plot(path = mdl_config['mdl_path'])

    # Gridsearch
    best_params = mdl.grid_search(X_train, y_train,
                                  param_grid=mdl_config['grid_search']['param_grid'],
                                  gs_params=mdl_config['grid_search']['params'],
                                  path = mdl_config['mdl_path'])

    best_params_load = mdl.load_grid_search(os.path.join(mdl_config['mdl_path'], 'BDT_grid_search.p'))

    # Shap
    mdl._initialize_shap(X_train, y_train)
    mdl.shap_summary_plot(X_train, path= mdl_config['mdl_path'])