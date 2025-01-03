# Loan_Default_Prediction
This project implements a machine learning model to predict loan defaults using a Random Forest Classifier. The dataset used for this project is 'Default_Fin.csv', which contains various financial features related to loan applicants. The model aims to predict whether an individual will default on their loan based on these features.
Project Overview

    Dataset: The dataset consists of financial information about loan applicants. It includes columns like income, credit score, loan amount, etc. The target variable, loan_default, indicates whether a loan applicant defaults on their loan (1 for default, 0 for non-default).

    Data Preprocessing:
        Missing values are handled by filling them with the median of each column.
        Categorical variables are converted into dummy variables using one-hot encoding.
        Feature scaling is applied to normalize the data using StandardScaler.

    Model Training:
        A Random Forest Classifier is trained on the preprocessed data.
        The dataset is split into training (80%) and testing (20%) sets using train_test_split.

    Evaluation:
        The model's performance is evaluated using accuracy, confusion matrix, and classification report.
        Feature importance is extracted to understand the contribution of each feature to the model's decision-making.

Libraries Used

    pandas for data manipulation
    numpy for numerical operations
    sklearn for machine learning tools like model training, evaluation, and preprocessing

Results

After training the model, the following metrics are printed:

    Confusion Matrix: To visualize the performance of the classification model.
    Classification Report: To show precision, recall, and F1-score for both classes.
    Accuracy Score: The overall accuracy of the model on the test set.
    Feature Importances: A ranked list of features based on their importance in the prediction.
