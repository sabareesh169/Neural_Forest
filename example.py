# Import libraries required.

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import itertools
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize

from build_nn import *
from mis_func import *
from neural_forest import *

data = pd.read_csv("Concrete_Data_missing.csv", delimiter=',', index_col=0)
target_col = ['Concrete compressive strength(MPa, megapascals) ']

nf = Neural_Forest()
layers = [6, 100, 100, 100, 100, 1]
nf.fit(data, layers, target_col)
