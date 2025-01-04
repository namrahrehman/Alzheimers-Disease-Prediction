# Alzheimer’s Disease Prediction Model

This project develops a predictive model to classify whether a patient has Alzheimer's Disease (AD) based on clinical and demographic features. The dataset includes attributes such as age, gender, BMI, Mini-Mental State Examination (MMSE) scores, and other relevant factors. The goal is to apply machine learning algorithms to predict the likelihood of AD diagnosis.

## Dataset

The dataset used in this project is sourced from [Kaggle's Alzheimer’s Disease Prediction Dataset](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset/data). It contains clinical and demographic data of patients, with the following features:

- **Age**: The patient's age (Numerical)
- **Gender**: The patient's gender (Categorical: Male, Female)
- **BMI**: Body Mass Index (Numerical)
- **MMSE**: Mini-Mental State Examination score (Numerical)
- **CholesterolTotal**: Total cholesterol level (Numerical)
- **FamilyHistoryAlzheimers**: Whether the patient has a family history of Alzheimer's (Categorical: Yes, No)
- **Diabetes**: Whether the patient has diabetes (Categorical: Yes, No)
- **CardiovascularDisease**: Whether the patient has cardiovascular disease (Categorical: Yes, No)
- **Depression**: Whether the patient has depression (Categorical: Yes, No)
- **Diagnosis**: The target variable indicating whether the patient has Alzheimer’s Disease (Categorical: Alzheimer's, Non-Alzheimer's)

The dataset is split into training and testing sets, which is used for building and evaluating machine learning models.

## Features

- **Algorithms Used**: 
  - Logistic Regression (without regularization)
  - Random Forest
  - XGBoost

- **Key Steps**:
  1. **Data Preprocessing**: Handling missing values, transforming categorical variables into factors, and splitting the data into training and testing sets.
  2. **Statistical Analysis**: Exploring correlations, performing t-tests, and chi-squared tests to assess feature significance.
  3. **Model Training**: Training models using logistic regression, random forest, and XGBoost algorithms.
  4. **Performance Evaluation**: Evaluating models using metrics like accuracy, precision, recall, F1 score, and AUC-ROC.

## Overview of Key Steps

### 1. Data Preprocessing
Data is loaded, cleaned (missing values imputed), and split into training and testing sets. Categorical variables are converted to factors to ensure proper handling by the machine learning models.

### 2. Statistical Analysis
- A **correlation matrix** is computed for numerical features to understand their relationships.
- **T-tests** and **Chi-squared tests** are applied to check for significant differences between groups (e.g., Age, BMI, Cholesterol, etc.) based on the diagnosis of Alzheimer's.

### 3. Model Training and Prediction
- **Logistic Regression**: Trained using the full dataset without regularization.
- **Random Forest**: A powerful ensemble method used for classification.
- **XGBoost**: A gradient boosting model that performs well on structured datasets.

### 4. Performance Evaluation
Models are evaluated using common classification metrics:
- **Accuracy**, **Precision**, **Recall**, **F1 Score**, and **AUC-ROC** are computed to assess the models' performance.

### 5. Feature Importance
The importance of each feature is evaluated to understand which factors most strongly contribute to the prediction of Alzheimer's.

## Conclusion
This project demonstrates the application of machine learning for Alzheimer's Disease prediction. Statistical analysis provides insight into the significance of various features, while the performance evaluation shows how well different algorithms classify the disease.

## Requirements

- R version 4.0 or later
- Required packages: `randomForest`, `caret`, `xgboost`, `ggplot2`, `pROC`

## Installation

To run the project, you need to install the necessary R packages:

```r
install.packages(c(
  "ggplot2", "dplyr", "tidyr", "caret", "randomForest", "e1071", "xgboost", "survival", "survminer", "glmnet", "DALEX"))
```
