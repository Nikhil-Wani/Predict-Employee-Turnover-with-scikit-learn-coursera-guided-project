from __future__ import print_function
%matplotlib inline

#task 1: importing libraries

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

#task 2: Exploratory data analysis

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

#task 3: Encode categorical features

cat_vars = ['department', 'salary']
for var in cat_vars:
  cat_list = pd.get_dummies(hr[var], prefix = var)
  hr = hr.join(cat_list)

hr.head()

hr.drop(columns = ['department', 'salary'], axis = 1, inplace = True)

#Task 4: Visualize class imbalance

from yellowbrick.target import ClassBalance
plt.style.use("ggplot")
plt.rcParams['figure.figsize'] = (12,8)

visualizer =  ClassBalance(labels = ['stayed','quit']).fit(hr.quit)
visualizer.show()

#Task 5: Create traning and tests sets

X = hr.loc[:, hr.columns != 'quit']
y = hr.quit

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 0, test_size=0.2,stratify=y)

#Task 6 : Build an interactive decision tree classifer

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

@interact
def plot_tree(crit=['gini','entropy'],
                 split=['best','random'],
                 depth=IntSlider(min=1,max=30,value=2,  continuous_update=False),
                 min_split=IntSlider(min=2,max=5,value=2, continous_update=False),
                 min_leaf=IntSlider(min=1,max=5,value=1, continuous_update=False)):
    estimator = DecisionTreeClassifer(random_state=0,
                                      criterion=crit,
                                      splitter=split,
                                      max_depth=depth,
                                      min_samples_split=min_split,
                                      min_samples_leaf=min_leaf)
    estimator.fit(X_train, y_train)
    print('Decision tree tranning accuracy: {:.3f}'.format(accuracy_score(y_train,estimator.predict(X_train))))
    print('Decision tree tranning accuracy: {:.3f}'.format(accuracy_score(y_test,estimator.predict(X_test))))
                                     
    graph = Source(tree.export_graphviz(estimator,out_file=None,
                                        feature_names=X_train.columns,
                                        class_names=['stayed','quit'],
                                        filled=True))
    display(Image(data=graph.pipe(format='png')))
    
    
#Task 7: Build an interactive random forest classifer
    
@interact
def plot_tree_rf(crit=['gini','entropy'],
                 bootstrap=['True','False'],
                 depth=IntSlider(min=1,max=30,value=2,  continuous_update=False),
                 forests=IntSlider(min=1,max=200,value=100, continuous_update=False),
                 min_split=IntSlider(min=2,max=5,value=2, continous_update=False),
                 min_leaf=IntSlider(min=1,max=5,value=1, continuous_update=False)):
    estimator = DecisionTreeClassifer(random_state=0,
                                      criterion = crit,
                                      bootstrap = bootstrap,
                                      n_estimators = forests,
                                      max_depth = depth,
                                      min_samples_split=min_split,
                                      min_samples_split=min_leaf,
                                      n_jobs = -1,
                                      verbose= False).fit(X_train,y_train)
    estimator.fit(X_train, y_train)
                                      
    print('Decision tree tranning accuracy: {:.3f}'.format(accuracy_score(y_train,estimator.predict(X_train))))
    print('Decision tree tranning accuracy: {:.3f}'.format(accuracy_score(y_test,estimator.predict(X_test))))
    num_tree = estimator.estimators_[0]
    print('\Visualizing tree:',0)         
                                     
    graph = Source(tree.export_graphviz(num_tree,
                                        out_file=None,
                                        feature_names=X_train.columns,
                                        class_names=['stayed','quit'],
                                        filled=True))
    
    display(Image(data=graph.pipe(format='png')))
    
#task 8: Feature importance and evolution metrics

from yellowbrick.model_selection import FeatureImportances
plt.rcParams['figure.figsize'] = (12,8)
plt.style.use("ggplot")

rf = RandomForestClassifer(bootstrap = 'True', class_weight = None, criterion='gini',
                           max_depth=5, max_feature='auto', max_leaf_nodes = None,
                           min_impurity_decrease=0.0, min_impurity_split= None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
                           oob_score=False, random_state=1, verbose=False,
                           warn_start=False)
viz= FeatureImportances(rf)
viz.fit(X_train,y_train)
viz.show();


dt = DecisionForestClassifer(class_weight = None, criterion='gini',
                           max_depth=3, max_feature='None', max_leaf_nodes = None,
                           min_impurity_decrease=0.0, min_impurity_split= None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort=False,random_state=0,
                           splitters='best')
                           
viz= FeatureImportances(dt)
viz.fit(X_train,y_train)
viz.show();


from yellowbrick.classifer import ROCAUC

visualizer = ROCAUC(rf, classes=['stayed','quit'])

visualizer.fit(X_train, y_train)
visulizer.score(X_test, y_test)
visualizer.pool();
