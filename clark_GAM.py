__author__ = 'clarkmaio'

from pygam import GAM, l, f, s, te
import pandas as pd
import numpy as np
import pickle
import os

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns

class clark_GAM(object):
    def __init__(self, mdl_config):
        self.mdl_config = mdl_config
        self._initialize_mdl()

    def _initialize_mdl(self):

        """
        Initilize model.
        Create mdl_path
        """
        if not os.path.exists(self.mdl_config['mdl_path']):
            os.mkdir(self.mdl_config['mdl_path'])
        self.mdl = GAM(**self.mdl_config['params'])


    def fit(self, X, y):
        self.mdl.fit(X, y)

    def predict(self, X):
        y_pred = self.mdl.predict(X)
        return y_pred

    def save(self):
        """
        Save mdl object.
        path is the full path: /.../mdl.p
        """
        print('>>>>> Saving GAM mdl...')
        mdl_path = os.path.join(self.mdl_config['mdl_path'], 'mdl_GAM.p')
        pickle.dump(self.mdl, open(mdl_path, 'wb'))

    def load(self):
        """
        Save mdl object.
        path is the full path: /.../mdl.p
        """
        print('>>>>> Loading GAM mdl...')
        mdl_path = os.path.join(self.mdl_config['mdl_path'], 'mdl_GAM.p')
        self.mdl = pickle.load(open(mdl_path, 'rb'))

    def plot_splines(self, s_ind, s_name):
        """
        Plot splines

        s_ind: index of corresponding terms to plot
        s_name:  names of corresponding terms. Will be used as plot titles
        """

        # Deduce feature index
        feature_ind = self.mdl.terms[s_ind].feature

        fig, ax = plt.subplots()

        XX = self.mdl.generate_X_grid(term=s_ind)
        ax.plot(XX[:, feature_ind], self.mdl.partial_dependence(term=s_ind, X=XX))
        ax.plot(XX[:, feature_ind], self.mdl.partial_dependence(term=s_ind, X=XX, width=.95)[1], c='r', ls='--')
        ax.set_title(s_name)

        plt.savefig(os.path.join(self.mdl_config['mdl_path'], 'GAM_spline_{}.png'.format(s_name)))
        plt.close()

    def plot_tensor(self, te_ind, te_components = ['x', 'y']):
        """
        Plot tensor surface

        te_ind: index of corresponding tensor term (index start from 0)
        te_components: list of names of features fitted by tensor
        """
        XX = self.mdl.generate_X_grid(term = te_ind, meshgrid = True)
        Z = self.mdl.partial_dependence(term = te_ind, X = XX, meshgrid = True)

        ax = plt.axes(projection = '3d')
        ax.plot_surface(XX[0], XX[1], Z, cmap = 'viridis')
        ax.set_xlabel(te_components[0])
        ax.set_ylabel(te_components[1])

        plt.savefig(os.path.join(self.mdl_config['mdl_path'], 'GAM_tensor_{}.png'.format(te_ind)))
        plt.close()


if __name__ == '__main__':
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split

    mdl_config = {
        'mdl_path': 'C:\\Users\\pc\\workspace\\clark_ml\\GAM_mdl_path',
        'params': {
            'terms':s(1, n_splines=50) + s(2, n_splines=50) + te(0,1)
        }
    }

    # Train/pred
    data = load_boston(return_X_y=False)
    X = pd.DataFrame(data['data'], columns=data['feature_names'])
    y = pd.DataFrame(data['target'], columns=['target'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8)

    mdl = clark_GAM(mdl_config=mdl_config)
    mdl.fit(X_train, y_train)
    mdl.save()

    # Plot
    mdl.plot_tensor(te_ind = 2, te_components=['feature 0', 'feature 1'])
    mdl.plot_splines(s_ind=0, s_name ='spline 0')
    mdl.plot_splines(s_ind=1, s_name='spline 1')