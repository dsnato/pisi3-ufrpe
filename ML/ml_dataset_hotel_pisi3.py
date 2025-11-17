# Execute primeiro no Colab (pode levar 1-2 minutos)
!pip uninstall -y scikit-learn scipy numpy
!pip install -q --upgrade scikit-learn xgboost shap imbalanced-learn umap-learn matplotlib seaborn joblib

# Imports principais
import os
import pathlib
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pprint import pprint

# sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# xgboost
import xgboost as xgb

# imbalanced-learn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# shap
import shap

# umap
import umap

# clustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# display
from IPython.display import display

# Links úteis (documentação)
print("Docs úteis:")
print(" scikit-learn: https://scikit-learn.org/stable/")
print(" imbalanced-learn (SMOTE): https://imbalanced-learn.org/stable/")
print(" shap: https://shap.readthedocs.io/")
print(" umap-learn: https://umap-learn.readthedocs.io/")

import pathlib
import pandas as pd # Adicionado import pandas
script_dir = pathlib.Path.cwd()
files = {
    'parquet': script_dir / 'hotel_bookings.parquet',
    'csv': script_dir / 'hotel_bookings.csv'
}

if files['parquet'].exists():
    df = pd.read_parquet(files['parquet'])
    print(f"Loaded parquet: {files['parquet']}")
elif files['csv'].exists():
    df = pd.read_csv(files['csv'])
    print(f"Loaded csv: {files['csv']}")
else:
    raise FileNotFoundError("Coloque hotel_bookings.csv ou hotel_bookings.parquet no diretório do notebook.")

print(f"\nTamanho: {df.shape[0]:,} linhas x {df.shape[1]} colunas")
display(df.head())
print("\nInfo resumida:")
display(df.info())
print("\nTarget distribution (is_canceled):")
display(df['is_canceled'].value_counts(normalize=False))
display(df.isnull().sum().sort_values(ascending=False).head(20))