"""
Utility functions and common models used to model data.
"""


import numpy as np


def linear_model_p1(x, a, b):
    return a * x + b


def single_exponential_model(x, a, b, tau):
    return a * np.exp(-x / tau) + b


def sinusoid(x, a, omega, phi, c):
    return a * np.sin(omega * x + phi) + c
