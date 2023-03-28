#!/usr/bin/env python
# coding: utf-8

# # Network Parameter Optimization

# In[2]:


import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from models import ShortCut11
from numpy.random import seed
import tensorflow
import time
seed(4750)
tensorflow.random.set_seed(4750)
time1 = time.time()
data = loadmat('./dataset/mango/mango_dm_split.mat')
x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=12, shuffle=True)
x_train, x_val, x_test = x_train[:, np.newaxis, :], x_val[:, np.newaxis, :], x_test[:, np.newaxis, :]
print(f"shape of data:\n"
      f"x_train: {x_train.shape}, y_train: {y_train.shape},\n"
      f"x_val: {x_val.shape}, y_val: {y_val.shape}\n"
      f"x_test: {x_test.shape}, y_test: {y_test.shape}")


# In[4]:


model_parameter_optimization = {"neuron num":[], "r2":[], "rmse":[]}
epoch, batch_size = 1024, 64

for i in range(2, 500):
      model = ShortCut11(network_parameter=i, input_shape=(1, 102))
      history_shortcut_11 = model.fit(x_train, y_train, x_val, y_val, epoch=epoch, batch_size=batch_size, save="/tmp/temp.hdf5", is_show=False)
      model = load_model("/tmp/temp.hdf5")
      y_pred = model.predict(x_test).reshape((-1, ))
      model_parameter_optimization['neuron num'].append(i)
      model_parameter_optimization['r2'].append(r2_score(y_test, y_pred))
      model_parameter_optimization['rmse'].append(mean_squared_error(y_test, y_pred))
      print(f"model with parameter {i}: r2: {model_parameter_optimization['r2'][-1]}, rmse: {model_parameter_optimization['rmse'][-1]}")
pd.DataFrame(model_parameter_optimization).to_csv("./dataset/test_result.csv")


# In[ ]:




