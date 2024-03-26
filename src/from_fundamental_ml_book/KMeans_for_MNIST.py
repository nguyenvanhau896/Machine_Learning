from __future__ import print_function
import numpy as np
from sklearn.datasets import fetch_mldata

data_dir = '../../data' # patch to data folder
mnist = fetch_mldata('MNIST original', data_home=data_dir)
print('Shape of MNIST data', mnist.data.shape)
