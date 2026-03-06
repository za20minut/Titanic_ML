Titanic Survival Prediction

This project presents a simple machine learning pipeline for predicting passenger survival on the Titanic dataset.
The workflow includes data preprocessing, missing data imputation, model training using Random Forest, and visualization of results.

Project Overview

The goal of this project is to build a classification model that predicts whether a passenger survived the Titanic disaster based on features such as:

age

passenger class

number of siblings/spouses aboard

number of parents/children aboard

fare

sex

port of embarkation

The project also includes exploratory data analysis and visualization to better understand the relationships between variables.

Technologies Used

Python

pandas

numpy

matplotlib

seaborn

scikit-learn

Machine Learning Pipeline

The project follows these main steps:

1. Data Loading

The dataset is loaded from a CSV file provided as a command-line argument.

2. Data Cleaning

The following columns are removed because they contain mostly identifiers or unstructured data:

PassengerId

Name

Ticket

Cabin

3. Missing Data Handling

Missing numerical values are filled using Iterative Imputation from scikit-learn.

Features imputed:

Age

Pclass

SibSp

Parch

Fare

The missing values in Embarked are filled with the most frequent value.

4. Encoding Categorical Variables

Categorical features are converted using:

One-hot encoding for:

Sex

Embarked

Label encoding for the target variable:

Survived

5. Train/Test Split

The dataset is split into:

80% training data

20% testing data

6. Model Training

A Random Forest Classifier is used to train the model.

7. Model Evaluation

The model is evaluated using:

Accuracy score

Classification report

Confusion matrix

Data Visualization

The project includes several visualizations to better understand the dataset:

Survival distribution

Survival by gender

Survival by passenger class

Age distribution vs survival

Feature importance from the trained model

Correlation heatmap

Example Visualizations

The script generates plots such as:

Confusion matrix

Survival statistics

Feature importance

Correlation matrix

How to Run

Clone the repository:

git clone https://github.com/yourusername/titanic-ml-analysis.git
cd titanic-ml-analysis

Install dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn

Run the script:

python script.py path_to_dataset.csv

Example:

python script.py titanic.csv
Project Structure
project/
│
├── script.py
├── README.md
└── dataset.csv
Future Improvements

Possible extensions of the project:

testing other machine learning models

hyperparameter tuning

cross-validation

feature engineering

improving visualizations
