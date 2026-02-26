# Regression-with-insurance-data
# Overview

This project explores the prediction of Insurance Premium Amount using demographic, financial, and policy-related variables.

The objective was not only to build a predictive model, but to:

Establish a strong statistical baseline

Compare multiple model families

Detect and fix data leakage

Evaluate model lift relative to naive prediction

Diagnose dataset signal strength

# Problem Statement

Given customer attributes such as:

Age

Annual Income

Credit Score

Health Score

Previous Claims

Policy Type

Vehicle Age

Predict the Premium Amount charged by the insurer.

# Dataset Summary

Target mean: 1100.57

Target standard deviation: 864.47

Baseline MAE (predicting mean): 668.06

The relatively high variance of the target makes this a challenging regression task.

# Preprocessing Pipeline

A structured ColumnTransformer was used to avoid data leakage.

**Numerical Features**

Median imputation

Scaling

**Categorical Features**

Most frequent imputation

OneHotEncoding (handle_unknown='ignore')

All transformations were encapsulated inside a scikit-learn Pipeline.

# Modeling Experiments

The following models were evaluated:

Model	Target Handling	R²	Test MAE
Linear Regression	log-transformed	~ -0.18	~ 728
Decision Tree	raw	~ 0.03	~ 656
Random Forest	raw	~ 0.03	~ 646
XGBoost	raw	~ 0.04	~ 646

All models were evaluated on the original target scale.

Data Leakage Debugging

During experimentation, an R² ≈ 0.999 was observed.

Investigation revealed:

The target (or a transformation of it) was accidentally included in features.

After removing leakage and retraining:

Performance returned to realistic levels (R² ≈ 0.04).

This debugging step highlights the importance of strict feature isolation in ML workflows.

# Key Findings

Feature correlations with target were very low.

Tree-based models slightly outperformed linear regression.

Model lift over baseline was modest (~3% MAE improvement).

Signal-to-noise ratio in dataset appears limited.

# Feature importance (XGBoost):

Annual Income

Health Score

Credit Score

Age

Vehicle Age

Categorical features had comparatively small impact.

# Interpretation

Despite testing multiple model classes:

The dataset explains only ~4% of target variance.

Premium appears weakly determined by provided variables.

Feature engineering likely offers more value than model switching.

This reflects a realistic modeling scenario where:

Data quality and feature richness dominate algorithm choice.

# Lessons Learned

Always benchmark against a naive baseline.

R² must be interpreted relative to target variance.

Log-transform evaluation must maintain scale consistency.

Data leakage can produce deceptively perfect metrics.

Model performance plateaus when signal is weak.

# Tech Stack

Python

pandas

NumPy

scikit-learn

XGBoost
