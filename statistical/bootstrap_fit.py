"""
This module defines a class used to perform non-linear regression using analytical models on bootstrapped data sets.
"""

import numpy as np
from scipy.optimize import curve_fit
from statistical.util import linear_model_p1, single_exponential_model, sinusoid


models = {'linear': linear_model_p1,
          'single_exponential': single_exponential_model,
          'sinusoid': sinusoid}


class BootStrapCB:
    """
    Use to perform regression on bootstrapped data sets. Includes methods to create bootstrapped data sets, fit
    arbitrary analytical models to these data and find the corresponding fit parameters, compute bootstrapped
    confidence bands.

    Parameters
    ----------
    x: independent variable. Numpy array.
    y: dependent variable. Numpy array.
    p0: optional initial guess for fit parameters. numpy array.
    model: select model to fit to data. Currently available: 'linear', 'single_exponential'. Can add new models to
    the statistical.util module. Default model is 'linear'.
    n_straps: number of bootstrap samples.Default is 1000.
    ci: confidence level for confidence bands. For example ci=0.95 is a 95% or 2*sigma confidence level.
    """
    n_points = 1000  # Number of points to generate for plotting confidence bands.
    error_counter = 0  # Initialize an optimization error counter.

    def __init__(self,
                 x,
                 y,
                 p_init = np.array([]),
                 model=models['linear'],
                 n_straps=1000,
                 ci=0.95):
        self.x = x
        self.y = y
        self.p_init = p_init
        self.model = models[model]
        self.n_straps = n_straps
        self.ci = ci


    def resampled_fit(self):
        """
        Sample with replacement the given data and perform a model fit on each sample. The fit routine here uses
        lower and upper bounds on the model parameters. This is essential to avoid overflow errors in, for example,
        non-linear models. The bounds are = parameter best estimate +/- 15 * sigma.
        Parameters
        ----------
        Returns the fit parameters.
        """
        y_shape = len(self.y)
        random_indices = np.random.randint(0, y_shape, (self.n_straps, y_shape))
        param_container = np.array([])
        params0, covariance = self.single_fit(covariance=True)
        l_bound = params0 - 15 * covariance.diagonal() ** 1 / 2
        u_bound = params0 + 15 * covariance.diagonal() ** 1 / 2
        for r_ind in random_indices:
            x_sample = self.x[r_ind]
            y_sample = self.y[r_ind]
            try:
                params, cov_m = curve_fit(self.model, x_sample, y_sample, p0=params0, bounds=(l_bound, u_bound))
            except RuntimeError:
                BootStrapCB.error_counter += 1
                pass
            param_container = np.append(param_container, params)
        param_container = param_container.reshape(self.n_straps, -1)
        param_container = np.swapaxes(param_container, 0, 1).reshape(-1, self.n_straps, 1)
        return param_container

    def single_fit(self, covariance=False):
        """
        Single fit to the data.
        Parameters
        ----------
        covariance: if True will also return the covariance matrix.
        p0: optional initial guess for fit parameters. numpy array.
        """
        if self.p_init.shape[0] == 0:
            params, cov_m = curve_fit(self.model, self.x, self.y)
        else:
            params, cov_m = curve_fit(self.model, self.x, self.y, p0=self.p_init)
        if covariance:
            return params, cov_m
        else:
            return params

    def make_xs(self):
        """
        Returns an array of independent variables.
        """
        x_min = self.x.min() - 1
        x_max = self.x.max() + 1
        return np.linspace(x_min, x_max, BootStrapCB.n_points)

    def evaluate_bootstrapped_cb(self, verbose=True):
        """
        Returns the bootstrapped confidence bands. The shape is (n_points, n_points, n_points, n_points).
        First column is an array of independent variables, second column is the lower confidence band, third column
        is an array of independent variables, the last column is the upper confidence band.
        Parameters
        ----------
        verbose: default True. Prints number of unsuccessful model parameter estimations.
        """
        x = self.make_xs()
        xx = np.repeat(x.reshape(1, BootStrapCB.n_points), self.n_straps, axis=0)
        btsrp_params = self.resampled_fit()
        if verbose:
            print(BootStrapCB.error_counter, '/',  BootStrapCB.n_points, 'unsuccessful model parameter estimations')
        BootStrapCB.error_counter = 0
        sigma = (1 + self.ci) * self.model(xx, *btsrp_params).std(ddof=1, axis=0)
        cb_minus = self.model(x, *self.single_fit()) - sigma
        cb_plus = self.model(x, *self.single_fit()) + sigma
        return np.vstack((x, cb_minus)), np.vstack((x, cb_plus))

    def evaluate_classical_lm_cb(self):
        """
        Returns the analytical confidence bands. The shape is (n_points, n_points, n_points, n_points).
        First column is an array of independent variables, second column is the lower confidence band, third column
        is an array of independent variables, the last column is the upper confidence band.
        """
        deg_f = len(self.y)
        params = self.single_fit()
        sig = np.sqrt(((self.y - self.model(self.x, *params)) ** 2).sum() / (deg_f - 2))
        std_x = ((self.x - self.x.mean()) ** 2).sum()
        xx = self.make_xs()
        sigma = sig * np.sqrt((1 / deg_f) + (xx - self.x.mean()) ** 2 / std_x)
        cb_minus = self.model(xx, *params) - (1 + self.ci) * sigma
        cb_plus = self.model(xx, *params) + (1 + self.ci) * sigma
        return np.vstack((xx, cb_minus)), np.vstack((xx, cb_plus))
