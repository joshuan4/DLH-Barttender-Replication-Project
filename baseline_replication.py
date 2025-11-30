# This file replicates the baseline barttender experiments (XGBoost and Logistic Regression)

from barttender.mimic_constants import *
import pandas as pd
from pathlib import Path
import numpy as np
import pandas as pd
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision.models as models
from sklearn.model_selection import train_test_split, KFold
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier  # Using actual XGBoost from CardiomegalyBiomarkers/XGBoost.ipynb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, matthews_corrcoef, average_precision_score
import statsmodels.api as sm


# every path returned from this file MUST be a pathlib.Path object
# Updated paths for user environment
home_out_dir = Path('physionet.org/outputs/cardiomegaly')
scratch_dir  = Path('physionet.org/files/mimic-cxr-jpg/2.1.0')  # MIMIC-CXR-JPG images
nb_group_dir = Path('physionet.org/files/mimic-cxr-jpg/2.1.0/preprocessed')  # Preprocessed images directory

# Import cleaned master dataframe
feature_folder = Path('physionet.org/Mimic_features/')

# Updating functions with new file paths
def get_master_df(idp=False):
    if idp:
        # (2662, 104)
        return pd.read_pickle('barttender/CardiomegalyBiomarkers/Cardiomegaly_Classification/MIMIC_features_OG/MIMIC_features.pkl')
    return pd.read_pickle('physionet.org/Mimic_features/MIMIC_features_v3.pkl')

def get_cardiomegaly_df(idp=False):
    df = None
    if idp:
        df = pd.read_pickle('barttender/CardiomegalyBiomarkers/Cardiomegaly_Classification/MIMIC_features_OG/MIMIC_features.pkl')
    else:
        df = pd.read_pickle('physionet.org/Mimic_features/MIMIC_features_v3.pkl')
    return df[df['Cardiomegaly'].isin([0, 1])]


"""
Functions for baseline comparisons (Logistic Regression, XGBoost) for Barttender
"""

def prepare_data(remove_high_nan=True, nan_threshold=1000, tabular_only=False, idp_only=False):

    # Prepare data for baseline models (shallow ML on tabular/scalar features).

    # Filter df to Cardiomegaly samples only
    # We are ONLY using the idp df, which is what Barttender did for baseline (with and without the two IDP columns), so the tables are comparable
    label = 'Cardiomegaly'
    df = get_cardiomegaly_df(idp=True)

    print(f"Total samples: {len(df)}")

    # Calculate age
    if 'StudyDate' in df.columns and 'anchor_year' in df.columns:
        study_year = np.floor(df['StudyDate'] / 10000)
        delta_years = study_year - df['anchor_year']
        df['age'] = df['anchor_age'] + delta_years
    else:
        df['age'] = df['anchor_age']

    # Normalize age (0-100 to 0-1)
    df['age_label'] = df['anchor_age'].apply(lambda x: min(x / 100, 1))
    demographic_cols = ['age_label']  # idpOnly age, no race/sex (matching limited notebook)

    # Encode demographics - this could be an ablation if we want to hardcode demo info into the features
    # df = standardize_mimic_ethnicity(df)
    #
    # if 'ethnicity' in df.columns:
    #     df['race_label'] = df['ethnicity']
    #     # Handle both standardized and non-standardized ethnicity values
    #     df.loc[df['race_label'] == 'White', 'race_label'] = 0
    #     df.loc[df['race_label'] == 'Asian', 'race_label'] = 1
    #     df.loc[df['race_label'] == 'Black', 'race_label'] = 2
    #     df.loc[df['race_label'] == 'Hispanic/Latino', 'race_label'] = 3
    #     df.loc[df['race_label'] == 'Other', 'race_label'] = 4
    #     # Handle any remaining string values (convert to numeric or drop)
    #     # If race_label is still string after mapping, it means it's an unexpected value
    #     if df['race_label'].dtype == 'object':
    #         # Convert any remaining non-numeric values to 'Other' (4)
    #         mask = pd.to_numeric(df['race_label'], errors='coerce').isna()
    #         df.loc[mask, 'race_label'] = 4
    #     df['race_label'] = pd.to_numeric(df['race_label'], errors='coerce')
    #
    # if 'gender' in df.columns:
    #     df['sex_label'] = df['gender']
    #     df.loc[df['sex_label'] == 'M', 'sex_label'] = 0
    #     df.loc[df['sex_label'] == 'F', 'sex_label'] = 1
    #
    # demographic_cols = ['age_label', 'race_label', 'sex_label']

    # Prepare features based on mode
    # First, only 8 significant tabular features (matching mimic-lr-idp-limited.ipynb)
    feature_cols = demographic_cols + [v for v in significant_variables[1:] if v in df.columns]

    # Add idp features
    idp_features = ['CTR', 'CPAR']
    if not tabular_only:
        for feat in idp_features:
            if feat in df.columns and feat not in feature_cols:
                feature_cols.append(feat)

    # If idp_only, only use these two features
    if idp_only:
        feature_cols = idp_features

    # Prepare splits
    if tabular_only:
        # For limited features (matching paper), use standard 80/20 split (matching mimic-lr-idp-limited.ipynb)
        # Don't use the 'split' column - create our own split
        X = df[feature_cols]
        Y = df[[label]]  # Keep as DataFrame for stratify
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42, stratify=Y[label]
        )
        Y_train = Y_train[label].astype(float)
        Y_test = Y_test[label].astype(float)
    elif 'split' in df.columns:
        # For non-limited mode, use split column if available
        X_train = df[df['split'] == 'train'][feature_cols]
        Y_train = df[df['split'] == 'train'][label].astype(float)
        X_test = df[df['split'] == 'test'][feature_cols]
        Y_test = df[df['split'] == 'test'][label].astype(float)

        # Adjust Train-Test Split to 90/10 (matching notebook Cell 43) - only for no-IDP dataset
        if tabular_only:
            # Calculate the number of samples to move from train to test
            additional_test_samples = 916

            # Sampling additional samples from X_train and Y_train
            X_train_to_test, X_train = train_test_split(
                X_train,
                test_size=(len(X_train) - additional_test_samples) / len(X_train),
                random_state=42,
                stratify=Y_train
            )
            Y_train_to_test, Y_train = train_test_split(
                Y_train,
                test_size=(len(Y_train) - additional_test_samples) / len(Y_train),
                random_state=42,
                stratify=Y_train
            )

            # Concatenate the sampled data to the test sets
            X_test = pd.concat([X_test, X_train_to_test], axis=0)
            Y_test = pd.concat([Y_test, Y_train_to_test], axis=0)
    else:
        # If no split column, create train/test split
        X = df[feature_cols]
        Y = df[label].astype(float)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42, stratify=Y
        )

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    return X_train, X_test, Y_train, Y_test, feature_cols

def compute_metrics(y_true, y_pred, y_prob):
    """
    Compute various classification metrics for binary classification.
    Based on: barttender/notebooks/mimic-lr-idp-limited.ipynb
    """
    auc = roc_auc_score(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)

    # Convert y_pred to binary predictions if it's probabilities
    y_pred_binary = (y_pred > 0.5).astype(int) if len(y_pred.shape) == 1 or y_pred.shape[1] == 1 else (y_pred > 0.5).astype(int)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)

    # Metrics calculations
    tpr = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0  # Sensitivity, Recall
    tnr = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0  # Specificity
    ppv = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0  # Precision
    npv = cm[0, 0] / (cm[0, 0] + cm[1, 0]) if (cm[0, 0] + cm[1, 0]) > 0 else 0  # Negative Predictive Value
    f1 = f1_score(y_true, y_pred_binary)
    mcc = matthews_corrcoef(y_true, y_pred_binary)

    return {
        'AUC': auc,
        'Average Precision': avg_precision,
        'TPR': tpr,
        'TNR': tnr,
        'PPV': ppv,
        'NPV': npv,
        'F1': f1,
        'MCC': mcc
    }

def run_logistic_regression(X_train, X_test, Y_train, Y_test, n_splits=10):
    """
    Run logistic regression baseline.

    Based on: barttender/notebooks/mimic-lr-no-idp.ipynb (Cells 24-29)
    Uses statsmodels.Logit for logistic regression with feature importance (Z-scores).
    """


    # 10-fold cross-validation approach (matching paper methodology)
    # Based on: barttender/notebooks/mimic-lr-idp-limited.ipynb
    # The notebook does: 80/20 split for test, then 10-fold CV on the 80%
    # For each fold, evaluate on the SAME held-out test set
    # So we use X_train/Y_train for CV, and X_test/Y_test as the held-out test set

    # Initialize 10-fold cross-validation on training data
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Initialize lists to store results
    fold_metrics = []
    fold_coefficients = []
    best_fold, best_mcc, best_z_scores = (0, 0, None)

    for fold, (train_index, val_index) in enumerate(kf.split(X_train), 1):
        X_train_fold = X_train.iloc[train_index]
        X_val_fold = X_train.iloc[val_index]
        Y_train_fold = Y_train.iloc[train_index] if hasattr(Y_train, 'iloc') else Y_train[train_index]
        Y_val_fold = Y_train.iloc[val_index] if hasattr(Y_train, 'iloc') else Y_train[val_index]

        # Mean Imputation for NaNs
        imputer = SimpleImputer(strategy='mean')
        X_train_imputed = pd.DataFrame(
            imputer.fit_transform(X_train_fold),
            columns=X_train_fold.columns
        )
        X_val_imputed = pd.DataFrame(
            imputer.transform(X_val_fold),
            columns=X_val_fold.columns
        )
        # Use the SAME test set for all folds (matching notebook)
        X_test_imputed = pd.DataFrame(
            imputer.transform(X_test),
            columns=X_test.columns
        )

        # One-hot encode race_label if present
        if 'race_label' in X_train_imputed.columns:
            X_train_encoded = pd.get_dummies(
                X_train_imputed,
                columns=['race_label'],
                drop_first=True,
                dtype=float
            )
            X_val_encoded = pd.get_dummies(
                X_val_imputed,
                columns=['race_label'],
                drop_first=True,
                dtype=float
            )
            X_test_encoded = pd.get_dummies(
                X_test_imputed,
                columns=['race_label'],
                drop_first=True,
                dtype=float
            )
            # Ensure same columns
            X_val_encoded = X_val_encoded.reindex(
                columns=X_train_encoded.columns,
                fill_value=0
            )
            X_test_encoded = X_test_encoded.reindex(
                columns=X_train_encoded.columns,
                fill_value=0
            )
        else:
            X_train_encoded = X_train_imputed
            X_val_encoded = X_val_imputed
            X_test_encoded = X_test_imputed

        # Add constant term
        X_train_encoded = sm.add_constant(X_train_encoded)
        X_val_encoded = sm.add_constant(X_val_encoded)
        X_test_encoded = sm.add_constant(X_test_encoded)

        # Reset indices
        X_train_encoded = X_train_encoded.reset_index(drop=True)
        if hasattr(Y_train_fold, 'reset_index'):
            Y_train_fold = Y_train_fold.reset_index(drop=True)
        X_val_encoded = X_val_encoded.reset_index(drop=True)
        if hasattr(Y_val_fold, 'reset_index'):
            Y_val_fold = Y_val_fold.reset_index(drop=True)
        X_test_encoded = X_test_encoded.reset_index(drop=True)
        Y_test = Y_test.reset_index(drop=True) if hasattr(Y_test, 'reset_index') else Y_test

        # Fit logistic regression
        logit_model = sm.Logit(Y_train_fold, X_train_encoded)
        result = logit_model.fit(disp=0)

        # Store coefficients
        fold_coefficients.append(result.params)

        # Predictions on test set (same test set for all folds, matching notebook)
        Y_test_pred_prob = result.predict(X_test_encoded)
        Y_test_pred = (Y_test_pred_prob > 0.5).astype(int)

        # Compute and store metrics on the held-out test set
        metrics = compute_metrics(Y_test, Y_test_pred, Y_test_pred_prob)
        fold_metrics.append(metrics)

        # Track best fold based on validation MCC
        Y_val_pred_prob = result.predict(X_val_encoded)
        Y_val_pred = (Y_val_pred_prob > 0.5).astype(int)
        val_metrics = compute_metrics(Y_val_fold, Y_val_pred, Y_val_pred_prob)

        if val_metrics['MCC'] > best_mcc:
            best_mcc = val_metrics['MCC']
            best_z_scores = pd.concat([
                result.summary2().tables[1]['z'],
                result.summary2().tables[1]['P>|z|']
            ], axis=1)
            best_z_scores.columns = ['z', 'p_value']
            best_fold = fold

        print(f"Fold {fold} completed")

    # Average metrics across folds
    metrics_df = pd.DataFrame(fold_metrics)
    avg_metrics = metrics_df.mean()
    std_metrics = metrics_df.std()

    print(f"\nAverage Metrics across {n_splits} folds:")
    print(f"  AUC: {avg_metrics['AUC']:.4f} ± {std_metrics['AUC']:.4f}")
    print(f"  F1:  {avg_metrics['F1']:.4f} ± {std_metrics['F1']:.4f}")
    print(f"  MCC: {avg_metrics['MCC']:.4f} ± {std_metrics['MCC']:.4f}")

    if best_z_scores is not None:
        print(f"\nTop 10 features by Z-score (from best fold {best_fold}):")
        print(best_z_scores.sort_values(by='z', key=abs, ascending=False).head(10))

    return {
        'auc': avg_metrics['AUC'],
        'f1': avg_metrics['F1'],
        'mcc': avg_metrics['MCC'],
        'auc_std': std_metrics['AUC'],
        'f1_std': std_metrics['F1'],
        'mcc_std': std_metrics['MCC'],
        'fold_metrics': fold_metrics,
        'model': None  # No single model in CV
    }

def run_xgboost(X_train, X_test, Y_train, Y_test):
    """
    Run XGBoost baseline.

    Uses XGBClassifier from xgboost library, matching the implementation in:
    - CardiomegalyBiomarkers/Cardiomegaly_Classification/XGBoost.ipynb
    """
    print("\n" + "=" * 80)
    print("Running XGBoost (XGBClassifier)")
    print("=" * 80)

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    # One-hot encode categorical variables
    if 'race_label' in X_train_imputed.columns:
        X_train_encoded = pd.get_dummies(
            X_train_imputed,
            columns=['race_label'],
            drop_first=True,
            dtype=float
        )
        X_test_encoded = pd.get_dummies(
            X_test_imputed,
            columns=['race_label'],
            drop_first=True,
            dtype=float
        )
        X_test_encoded = X_test_encoded.reindex(
            columns=X_train_encoded.columns,
            fill_value=0
        )
    else:
        X_train_encoded = X_train_imputed
        X_test_encoded = X_test_imputed

    # Train model using XGBClassifier (matching CardiomegalyBiomarkers/XGBoost.ipynb)
    # Parameters based on the notebook: eval_metric='logloss', scale_pos_weight=0.3/0.7,
    # colsample_bytree=0.75, gamma=0, learning_rate=0.1, subsample=0.75, max_depth=8
    model = XGBClassifier(
        eval_metric='logloss',
        scale_pos_weight=0.3/0.7,  # Based on CardiomegalyBiomarkers/XGBoost.ipynb
        colsample_bytree=0.75,
        gamma=0,
        learning_rate=0.1,
        subsample=0.75,
        max_depth=8,
        random_state=42,
        verbosity=1
    )
    model.fit(X_train_encoded, Y_train)

    # Predictions
    Y_test_pred_prob = model.predict_proba(X_test_encoded)[:, 1]
    Y_test_pred = model.predict(X_test_encoded)

    # Metrics
    auc = roc_auc_score(Y_test, Y_test_pred_prob)
    f1 = f1_score(Y_test, Y_test_pred)
    mcc = matthews_corrcoef(Y_test, Y_test_pred)
    cm = confusion_matrix(Y_test, Y_test_pred)

    print(f"\nResults:")
    print(f"  AUC: {auc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  MCC: {mcc:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

    # Feature importance (XGBoost has feature_importances_ attribute)
    feature_importance = pd.DataFrame({
        'feature': X_train_encoded.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 10 features by importance:")
    print(feature_importance.head(10))

    return {
        'auc': auc,
        'f1': f1,
        'mcc': mcc,
        'confusion_matrix': cm,
        'model': model,
        'feature_importance': feature_importance
    }



"""
Run baseline comparisons using shallow ML models (logistic regression, XGBoost) on tabular/imaging features.
"""

results = {}

# Linear regression

# Prepare data for the "Tabular Data" runs
X_train, X_test, Y_train, Y_test, feature_cols = prepare_data(
    tabular_only=True
)

# Replicate by using average across 10-folds for logistic
results['Logistic Regression, Tabular Data'] = run_logistic_regression(X_train, X_test, Y_train, Y_test, n_splits=10)
results['XGBoost, Tabular Data'] = run_xgboost(X_train, X_test, Y_train, Y_test)

# Prepare data for the "Image Biomarkers + Tabular Data" runs (same except addition of two biomarker columns, CTR/CPAT)
X_train, X_test, Y_train, Y_test, feature_cols = prepare_data(
    tabular_only=False
)

# Replicate by using average across 10-folds for logistic
results['Logistic Regression, Image Biomarkers + Tabular Data'] = run_logistic_regression(X_train, X_test, Y_train, Y_test, n_splits=10)
results['XGBoost, Image Biomarkers + Tabular Data'] = run_xgboost(X_train, X_test, Y_train, Y_test)

# Prepare data for the "Image Biomarkers" runs (just two columns, CTR/CPAT)
X_train, X_test, Y_train, Y_test, feature_cols = prepare_data(
    tabular_only=False,
    idp_only=True
)

# Replicate by using average across 10-folds for logistic
results['Logistic Regression, Image Biomarkers'] = run_logistic_regression(X_train, X_test, Y_train, Y_test, n_splits=10)
results['XGBoost, Image Biomarkers'] = run_xgboost(X_train, X_test, Y_train, Y_test)

# Summary
print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
for method_name, result in results.items():
    print(f"{method_name.upper()}:")
    print(f"  AUC: {result['auc']:.4f}")
    print(f"  F1:  {result['f1']:.4f}")
    print(f"  MCC: {result['mcc']:.4f}")