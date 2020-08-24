# Predict Employee Turnover with scikit-learn

Now, we you will apply decision trees and random forests using scikit-learn and Python to build an employee churn prediction application with interactive controls. We will accomplish this with the help of following tasks in the project:

1. Introduction and Import Libraries
2. Exploratory Data Analysis
3. Encode Categorical Features
4. Visualize Class Imbalance
5. Create Training and Test Sets
6. Build a Decision Tree Classifier with Interactive Controls
7. Build a Random Forest Classifier with Interactive Controls
8. Feature Importance Plots and Evaluation Metrics

# Project Structure

The hands on project on Predict Employee Churn with Decision Trees and Random Forests is divided into the following tasks:

<b>Task 1: Introduction and Import Libraries</b>

Introduction to the data set and the problem overview.
See a demo of the final product you will build by the end of this project.
Introduction to the Rhyme interface.
Import essential modules and helper functions from NumPy, Matplotlib, and scikit-learn.

<b>Task 2: Exploratory Data Analysis</b>

Load the employee dataset using pandas
Explore the data visually by graphing various features against the target with Matplotlib.

<b>Task 3: Encode Categorical Features</b>

The dataset contains two categorical variables: Department and Salary.
Create dummy encoded variables for both categorical variables.

<b>Task 4: Visualize Class Imbalance</b>

Use Yellowbrick's Class Balance visualizer to create a frequency plot of both classes.
The presence or absence of a class balance problem will inform your sampling strategory while creating training and validation sets.

<b>Task 5: Create Training and Validation Sets</b>

Split the data into a 80/20 training/validation split.
Use a stratified sampling strategy

<b>Tasks 6: Build a Decision Tree Classifier with Interactive Controls</b>

Use the interact function to automatically create UI controls for function arguments.
Build and train a decision tree classifier with scikit-learn.
Calculate the training and validation accuracies.
Display the fitted decision tree graphically.

<b>Task 7: Build a Random Forest Classifier with Interactive Controls</b>

Use the interact function again to automatically create UI controls for function arguments.
To overcome the variance problem associated with decision trees, build and train a random forests classifier with scikit-learn.
Calculate the training and validation accuracies.
Display a fitted tree graphically.

<b>Task 8: Feature Importance Plots and Evaluation Metrics</b>

Many model forms describe the underlying impact of features relative to each other.
Decision Tree models and Random Forest in scikit-learn, feature_importances_ attribute when fitted.
Utilize this attribute to rank and plot the features.

