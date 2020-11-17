import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


class welford_online():
    '''
    Class to peform online outliers detection using Welford formulas.

    Criteria consist in online z score.

    avg_decay: exp decay factor for rolling mean (usually between 0 and 2)
    var_decay: exp decay factor for rolling variance (usually between 0 and 2)
    outlier_cutoff: outliers detection cutoff in multiplies of standard deviations
    '''

    def __init__(self, mean_decay = 1, var_decay = 1, outlier_cutoff = 1.5):
        self.mean_decay = mean_decay
        self.var_decay = var_decay
        self.outlier_cutoff = outlier_cutoff

        self.__inizialize()

    def __inizialize(self):
        self.f_mean = np.exp(- np.log(2) / self.mean_decay)
        self.f_var = np.exp(- np.log(2) / self.var_decay)

        self.mean = 0
        self.M2n = 0
        self.dev_from_mean = 0
        self.n = 0
        self.count = 0

        self.label = pd.DataFrame(columns = ['y', 'mean', 'M2n', 'n','dev_from_mean','isOutlier', 'outlier_score'])

    def __update_label(self, y):
        '''
        update label with current params
        '''

        self.label.loc[self.count, 'y'] = y
        self.label.loc[self.count, 'mean'] = self.mean
        self.label.loc[self.count, 'M2n'] = self.M2n
        self.label.loc[self.count, 'n'] = self.n
        self.label.loc[self.count, 'dev_from_mean'] = self.dev_from_mean
        self.label.loc[self.count, 'isOutlier'] = self.isOutlier
        self.label.loc[self.count, 'outlier_score'] = self.outlier_score

    def update(self, y):
        self.count += self.count
        self.n = self.n*self.f_var + 1


        if self.count == 1:
            self.mean = y
            self.M2n = 1e-5
            self.dev_from_mean = 0

            self.isOutlier = False
            self.outlier_score = 0

        elif self.count == 2:
            self.mean = (self.mean + y)/2.
            self.M2n = ((self.mean - y)*10)**2
            self.dev_from_mean = 0

            self.isOutlier = False
            self.outlier_score = 0

        else:
            self.dev_from_mean = np.abs(self.mean - y) / np.sqrt(self.M2n / self.n)

            if self.dev_from_mean < self.outlier_cutoff:
                # Update since it is not outlier
                from copy import copy
                prevmean = copy(self.mean)
                self.mean = self.f_mean*self.mean + (1-self.f_mean)*y
                self.M2n = self.f_var*self.M2n + (y - prevmean) * (y - self.mean)

                self.isOutlier = False
            else:
                # If outlier DO NOT update values (except for counter and n)
                self.isOutlier = True

            self.dev_from_mean = np.abs(self.mean - y) / np.sqrt(self.M2n / self.n)
            self.outlier_score = self.dev_from_mean / self.outlier_cutoff

        self.__update_label(y)

    def fit(self, Y):
        '''
        Train by performing update on the time series.
        At the end you will get update params
        '''
        for y in Y:
            self.update(y)

    def pred(self, Y):

        for y in Y:
            self.update(y)

    def fit_pred(self, Y_train, Y_test):
        # Update on train
        for y in Y_train:
            self.update(y)

        # and now on test
        for y in Y_test:
            self.update(y)

    def summary(self):
        '''
        Visualize status
        '''

        import matplotlib.pyplot as plt
