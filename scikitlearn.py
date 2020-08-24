from __future__ import print_function
%matplotlib inline

//task 1: importing libraries

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
import pandas as pd
import pandas_profiling
plt.style.use("ggplot")
warnings.simplefilter("ignore")

lt.rcParams['figure.figsize'] = (12,8)

//task 2: Exploratory data analysis

hr =  pd.read_csv('data/employee_data.csv')
hr.head()

hr.profile_report(titile = "Data report")

pd.crosstab(hr.department, hr.quit).plot(kind = 'bar')
plt.title("Turnover Frequency on salary bracket")
plt.xlabel('department')
plt.ylabel('frequency of turnover')
plt.show()

pd.crosstab(hr.salary, hr.quit).plot(kind = 'bar')
plt.title("Turnover Frequency on salary bracket")
plt.xlabel('salary')
plt.ylabel('frequency of turnover')
plt.show()

//task 3: Encode categorical features

cat_vars = ['department', 'salary']
for var in cat_vars:
  cat_list = pd.get_dummies(hr[var], prefix = var)
  hr = hr.join(cat_list)

hr.head()

hr.drop(columns = ['department', 'salary'], axis = 1, inplace = True)

//Task 4: Visualize class imbalance

from yellowbrick.target import ClassBalance
plt.style.use("ggplot")
plt.rcParams['figure.figsize'] = (12,8)

visualizer =  ClassBalance(labels = ['stayed','quit']).fit(hr.quit)
visualizer.show()

//Task 5: Create traning and tests sets

X = hr.loc[:, hr.columns != 'quit']
y = hr.quit

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 0, test_size=0.2,stratify=y)

//Task 6 & 7: Build an interactive decision tree classifer

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz # display the tree within a Jupyter notebook
from IPython.display import SVG
from graphviz import Source
from IPython.display import display
from ipywidgets import interactive, IntSlider, FloatSlider, interact
import ipywidgets
from IPython.display import Image
from subprocess import call
import matplotlib.image as mpimg

