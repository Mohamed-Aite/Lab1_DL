# Project Description

This project explores the implementation of **Deep Neural Networks (DNNs)** and **Multi-Layer Perceptrons (MLPs)** using **PyTorch**. It covers two fundamental machine learning tasks: regression and multi-class classification, following a complete workflow from exploratory data analysis and preprocessing to model training, hyperparameter tuning, evaluation, and regularization.

The project is organized into two parts.

## Part 1: Stock Price Regression

This part uses the [NYSE dataset](https://www.kaggle.com/datasets/dgawlik/nyse) to predict stock closing prices based on market features, including opening, high, and low prices, as well as trading volume.

The notebook covers:

- Exploratory data analysis (EDA) and data visualization.
- Data preprocessing and feature standardization.
- Implementation of an MLP regression model using PyTorch.
- Hyperparameter optimization with `GridSearchCV`.
- Visualization of training and testing loss curves.
- Comparison of regularization techniques, including dropout, L1 and L2 regularization, and batch normalization.

## Part 2: Multi-Class Predictive Maintenance Classification

This part uses the [Machine Predictive Maintenance dataset](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification) to classify machine operating conditions and identify different types of equipment failure.

The notebook covers:

- Data cleaning, categorical encoding, normalization, and preprocessing.
- Exploratory analysis of feature distributions and class imbalance.
- Application of SMOTE to address class imbalance.
- Implementation of a multi-class MLP classifier using PyTorch.
- Hyperparameter optimization with `GridSearchCV`.
- Visualization of training and testing loss and accuracy curves.
- Model evaluation using accuracy, precision, recall (sensitivity), specificity, F1-score, and a confusion matrix.
- Comparison of regularization techniques to study their impact on model performance.

## Project Structure

| Notebook | Description |
|---|---|
| `Part1_Regression.ipynb` | Stock closing price prediction using regression |
| `Part2_Classification.ipynb` | Machine condition and failure type classification |

Both notebooks use `kagglehub` to download the datasets and can be executed in Google Colab or Kaggle.

## Key Takeaways

Through this project, I developed practical experience in:

- Exploring, analyzing, and preparing real-world datasets for machine learning.
- Building and training DNN and MLP models with PyTorch.
- Applying neural networks to regression and multi-class classification problems.
- Selecting appropriate hyperparameters using `GridSearchCV`.
- Evaluating model performance through statistical metrics and visualizations.
- Understanding the impact of class imbalance and data leakage on model reliability.
- Applying SMOTE to improve learning from imbalanced datasets.
- Comparing regularization methods to better understand and mitigate overfitting.