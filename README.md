# Spam-Mail-Prediction-Using-Machine-Learning
A machine learning project to classify emails as spam or ham using a Logistic Regression model. This project involves data cleaning, feature extraction using TF-IDF, model training, evaluation, and building a simple predictive system.

# 🛠️ Technologies Used
Python

Pandas, NumPy

Scikit-learn

TF-IDF Vectorizer

Jupyter Notebook

Matplotlib & Seaborn (for data exploration)

# 📁 Dataset

Dataset used: mail_data.csv

Contains labeled messages with categories: spam and ham

# 🚀 Project Workflow

1)Data Collection & Cleaning

Loaded CSV data into a Pandas DataFrame
Handled missing values and converted categorical labels (spam = 0, ham = 1)

2)Text Preprocessing

Transformed email text into numerical vectors using TF-IDF Vectorizer
Removed stop words and converted text to lowercase

3)Train-Test Split

Split the dataset into training (80%) and testing (20%) sets

4)Model Training

Trained a Logistic Regression model using scikit-learn

5)Evaluation

Evaluated model performance using accuracy score
Verified the model's ability to generalize on unseen data

# 📊 Results

Achieved high accuracy in classifying messages correctly
Demonstrated good generalization on test data

