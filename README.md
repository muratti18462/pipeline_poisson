# pipeline_poisson

This repository contains a Python script for predicting sales using various regression models. The script uses historical sales data provided in an Excel file and implements a pipeline with multiple regressors, including Gradient Boosting, Random Forest, Multi-layer Perceptron (MLP), and Poisson Regression.


- The script utilizes scikit-learn for building a pipeline with different regressors.
- Data preprocessing involves aggregating monthly and seasonal data, and a 4D plot of the data is generated for visualization.
- The main model is a Voting Regressor combining the strengths of MLP, Poisson Regression, and Gradient Boosting.
