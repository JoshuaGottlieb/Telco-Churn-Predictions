# Telco-Churn-Predictions

## Project Description and Repository Usage
This project analyzes customer churn data from Telco Systems, a telecommunications company. This repository is split into two distinct parts: Exploratory Data Analysis (EDA) and Modeling.

The EDA process consists of cleaning and validating data, identifying outliers, and analyzing feature importance and correlation. The purpose of performing EDA is to obtain a thorough understanding of the data and extract insights about patterns and relationships in the data without placing any assumptions on the underlying factors that shape the data. The EDA process for this project is detailed in the notebook titled [EDA.ipynb](https://github.com/JoshuaGottlieb/Telco-Churn-Predictions/blob/main/src/EDA.ipynb). Within this notebook, the data is cleaned and analyzed, with conclusions drawn about which factors are most important towards predicting customer churn based on summary statistics.

The modeling process consists of fitting multiple models to the data and evaluating the results. The data is split into training and testing sets in order to simulate the models' ability to perform on new, unseen data. A key part of this modeling process is hyperparameter tuning - a process by which the parameters for a model are changed to try to improve performance. Predicting customer churn is a classification task, and there are multiple metrics to evaluate the effectiveness of a model on classification. There are 4 common classification metrics: Accuracy, Recall, Precision, and F1-Score. It is also possible to test the robustness of a model under different levels of certainty using Receiver Operater Characteristic (ROC) curves and Precision-Recall curves (PRC). The modeling process for this project can be found in the notebook titled [Modeling.ipynb](https://github.com/JoshuaGottlieb/Telco-Churn-Predictions/blob/main/src/Modeling.ipynb). This notebook contains the results and evaluation of training four different models on the Telco churn data. In addition, the models were tested using the Synthetic Minority Oversampling Technique (SMOTE) to reduce class imbalances, as far fewer customers churn than those who do not. Subsets of the Telco churn data and Principal Component Analysis (PCA) were used to test how the models performed with reduced dimensionality of the dataset.

All of the models were saved as pickled objects under [models](https://github.com/JoshuaGottlieb/Telco-Churn-Predictions/tree/main/models) and compressed using the LZMA algorithm. In order to keep the notebooks easy to read and to reduce repetition of code, many functions were moved into project [modules](https://github.com/JoshuaGottlieb/Telco-Churn-Predictions/tree/main/src/modules), separated by core function use. For more information, the modules contain full documentation, and module summaries are shown below under the repository structure.

## Respository Structure
```
.
|── data/                            # Raw and cleaned data
|   ├── raw/
|   |	└── telco-customer-churn.csv
|   └──  cleaned/
|   	└── telco-churn-data-cleaned-not-encoded.csv
|
|── models/                          # Pickled and compressed trained scikit-learn models
|   ├── bayes/                       # Trained Bernoulli Naive Bayes models
|   ├── forest/                      # Trained Random Forest Classifier models
|   ├── logreg/                      # Trained Logistic Regression models
|   └──  xgboost/                    # Trained XGBoost Classifier models
|
|── src/                             # Notebooks go here
|   ├── EDA.ipynb                    # Exploratory data analysis and data cleaning
|   ├── Modeling.ipynb               # Training and model evaluation
|   └── modules/                     # Modules used by notebooks go here
|   	├── __init__.py
|   	├── feature_analysis.py      # Functions to assist in EDA of variable correlations
|   	├── model_evaluation.py      # Functions to evaluate model performance
|   	├── preprocessing.py         # Functions to preprocess datasets for use by scikit-learn
|   	├── training.py              # Functions to fit models
|   	└── utils.py                 # Other utility functions
```


## Libraries and Versions
```
# Minimal versions of libraries and Python needed
Python 3.8.10
imbalanced_learn==0.12.4
matplotlib==3.7.5
numpy==1.24.4
pandas==2.0.3
scikit_learn==1.3.2
seaborn==0.13.2
```
