# Predictive-Churn-Analysis
# ================================================================
#              PREDICTIVE CHURN ANALYSIS
# ================================================================
#
# Objective:
# Build and evaluate a machine learning classification pipeline
# to predict customer churn probability.
#
# Models:
# 1. Logistic Regression
# 2. Random Forest
#
# Evaluation:
# 1. Accuracy
# 2. Precision
# 3. Recall
# 4. F1-Score
# 5. ROC-AUC
# 6. Confusion Matrix
# 7. ROC Curve
#
# Output:
# 1. Churn predictions
# 2. Churn probability / risk score
# 3. Evaluation report
# 4. Graphs
# 5. CSV prediction file
#
# Required libraries:
# pip install pandas numpy matplotlib seaborn scikit-learn
#
# ================================================================


# ================================================================
# 1. IMPORT LIBRARIES
# ================================================================

import os
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split
)

from sklearn.compose import (
    ColumnTransformer
)

from sklearn.pipeline import (
    Pipeline
)

from sklearn.preprocessing import (
    OneHotEncoder,
    MinMaxScaler
)

from sklearn.impute import (
    SimpleImputer
)

from sklearn.linear_model import (
    LogisticRegression
)

from sklearn.ensemble import (
    RandomForestClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

from sklearn.inspection import (
    permutation_importance
)


warnings.filterwarnings("ignore")


# ================================================================
# 2. PROJECT SETTINGS
# ================================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

TARGET_COLUMN = "churn"


# ================================================================
# 3. CREATE SYNTHETIC DATASET
# ================================================================
#
# This function is used if the user does not have the downloaded
# modeling dataset.
#
# The generated data contains:
#
# customer_id
# age
# tenure_months
# monthly_charges
# total_charges
# support_calls
# usage_hours
# contract_type
# payment_method
# internet_service
# senior_citizen
# partner
# dependents
# paperless_billing
# churn
#
# ================================================================

def create_synthetic_dataset(
    number_of_customers=5000
):

    print("\nCreating synthetic customer churn dataset...")

    np.random.seed(RANDOM_STATE)

    customer_id = np.arange(
        10001,
        10001 + number_of_customers
    )

    age = np.random.randint(
        18,
        75,
        number_of_customers
    )

    tenure_months = np.random.randint(
        1,
        73,
        number_of_customers
    )

    monthly_charges = np.round(
        np.random.uniform(
            20,
            150,
            number_of_customers
        ),
        2
    )

    total_charges = np.round(
        monthly_charges *
        tenure_months *
        np.random.uniform(
            0.85,
            1.15,
            number_of_customers
        ),
        2
    )

    support_calls = np.random.poisson(
        2,