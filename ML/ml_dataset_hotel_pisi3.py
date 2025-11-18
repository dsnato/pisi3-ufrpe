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
import statistics

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
from sklearn.metrics import silhouette_score, RocCurveDisplay, ConfusionMatrixDisplay

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

df = df.copy()

# Features novas (como no seu script)
df['total_guests'] = df['adults'].fillna(0) + df['children'].fillna(0) + df['babies'].fillna(0)
df['total_nights'] = df['stays_in_weekend_nights'].fillna(0) + df['stays_in_week_nights'].fillna(0)
df['has_special_request'] = (df['total_of_special_requests'].fillna(0) > 0).astype(int)
df['is_family'] = ((df['adults'].fillna(0) > 0) & ((df['children'].fillna(0) > 0) | (df['babies'].fillna(0) > 0))).astype(int)

# Tratar faltantes simples
df['company'].fillna(0, inplace=True)
df['agent'].fillna(0, inplace=True)
df['country'].fillna('Unknown', inplace=True)
df['children'].fillna(0, inplace=True)

# Remover outliers simples de ADR (mantemos <1000)
df = df[df['adr'] < 1000].reset_index(drop=True)

# Features escolhidas (base)
features = [
    'hotel', 'lead_time', 'arrival_date_month', 'arrival_date_week_number',
    'arrival_date_day_of_month', 'stays_in_weekend_nights', 'stays_in_week_nights',
    'adults', 'children', 'babies', 'country', 'market_segment',
    'distribution_channel', 'is_repeated_guest', 'previous_cancellations',
    'previous_bookings_not_canceled', 'reserved_room_type', 'assigned_room_type',
    'booking_changes', 'deposit_type', 'agent', 'company', 'customer_type',
    'adr', 'required_car_parking_spaces', 'total_of_special_requests',
    'total_guests', 'total_nights', 'has_special_request', 'is_family'
]
target = 'is_canceled'

X = df[features].copy()
y = df[target].copy()

# Separar colunas numéricas e categóricas
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"Numéricas: {len(numeric_cols)} - Categóricas: {len(categorical_cols)}")

# Preprocessor: imputers, scaler, onehot
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Salvar artefatos básicos para uso posterior
joblib.dump(preprocessor, 'preprocessor.pkl')
joblib.dump(features, 'features_list.pkl')
df.to_parquet('hotel_bookings_processed.parquet', index=False)
print("Preprocessor salvo: preprocessor.pkl, dataset salvo: hotel_bookings_processed.parquet")

# Split estratificado
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Treino: {len(X_train):,} - Teste: {len(X_test):,}")
print("Balance (treino):")
print(y_train.value_counts(normalize=True))

# Pipeline com SMOTE aplicado APÓS o pré-processamento numérico/categórico:
# Usamos ImbPipeline para garantir que SMOTE opere no espaço numérico transformado.
smote = SMOTE(random_state=42)

rf = RandomForestClassifier(random_state=42, n_jobs=1)
xgb_clf = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=1)
logreg = LogisticRegression(max_iter=200, random_state=42)

# Construir pipelines modelos (ex.: RF)
pipeline_rf = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', smote),
    ('classifier', rf)
])

pipeline_xgb = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', smote),
    ('classifier', xgb_clf)
])

pipeline_log = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', smote),
    ('classifier', logreg)
])

# Salvar pipelines iniciais (sem fit)
joblib.dump({'rf': pipeline_rf, 'xgb': pipeline_xgb, 'log': pipeline_log}, 'pipelines_initial.pkl')
print("Pipelines iniciais salvos: pipelines_initial.pkl")

# Cell 5/10 - Treinamento: cross-val por 10 seeds (agregação de métricas)

models = {
    'RandomForest': pipeline_rf,
    'XGBoost': pipeline_xgb,
    'LogisticRegression': pipeline_log
}

seeds = [0, 7, 13, 21, 42, 99, 123, 2023, 327, 999]
cv = StratifiedKFold(n_splits=5, shuffle=True)

results_by_model = {}

for name, pipeline in models.items():
    print(f"\n=== Modelo: {name} ===")
    metrics_per_seed = []
    for seed in seeds:
        # cross_validate para obter múltiplas métricas
        scores = cross_validate(pipeline, X_train, y_train,
                                scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
                                n_jobs=1, return_train_score=False)
        # agregar médias por seed
        seed_metrics = {m: float(np.mean(scores[f'test_{m}'])) for m in ['accuracy','precision','recall','f1','roc_auc']}
        seed_metrics['seed'] = seed
        metrics_per_seed.append(seed_metrics)
        print(f" seed {seed}: f1={seed_metrics['f1']:.4f} roc_auc={seed_metrics['roc_auc']:.4f}")
    # agregados
    df_metrics = pd.DataFrame(metrics_per_seed)
    summary = df_metrics.mean().to_dict()
    summary['std'] = df_metrics.std().to_dict()
    results_by_model[name] = {'per_seed': df_metrics, 'summary': summary}
    print(f" -> média F1 across seeds: {summary['f1']:.4f} (std {summary['std']['f1']:.4f})")

# Salvar resultados resumidos
joblib.dump(results_by_model, 'cv_results_by_model.pkl')
print("\nResultados de cross-val salvos: cv_results_by_model.pkl")

# Cell 6/10 - Fit final (usar seed 42) e diagnóstico treino vs teste (overfitting)
best_models = {}
for name, info in results_by_model.items():
    # escolher modelo com maior mean f1 across seeds
    mean_f1 = info['summary']['f1']
    print(f"Modelo {name} mean_f1={mean_f1:.4f}")

# Suponha que RandomForest foi o melhor — vamos treinar todos com seed=42 e comparar
final_seed = 42
fitted_models = {}

for name, pipeline in models.items():
    print(f"\nFit final: {name}")
    pipeline.set_params(classifier__random_state=final_seed) if hasattr(pipeline.named_steps['classifier'],'random_state') else None
    pipeline.fit(X_train, y_train)
    fitted_models[name] = pipeline
    # Métricas no treino
    y_train_pred = pipeline.predict(X_train)
    y_train_proba = pipeline.predict_proba(X_train)[:,1] if hasattr(pipeline.named_steps['classifier'],'predict_proba') else None
    train_f1 = f1_score(y_train, y_train_pred)
    train_acc = accuracy_score(y_train, y_train_pred)
    # Métricas no teste
    y_test_pred = pipeline.predict(X_test)
    y_test_proba = pipeline.predict_proba(X_test)[:,1] if hasattr(pipeline.named_steps['classifier'],'predict_proba') else None
    test_f1 = f1_score(y_test, y_test_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f" Train Acc: {train_acc:.4f} F1: {train_f1:.4f} | Test Acc: {test_acc:.4f} F1: {test_f1:.4f}")
    # Diagnóstico simples de overfitting
    gap = train_f1 - test_f1
    if gap > 0.10:
        print("  ⚠️ Possível overfitting (gap F1 train - test > 0.10).")
    else:
        print("  ✓ Gap aceitável.")
    # salvar modelo
    joblib.dump(pipeline, f"final_model_{name.lower()}.pkl")
    print(f"  Modelo salvo: final_model_{name.lower()}.pkl")

# Selecionar melhor modelo por test F1
test_f1s = {}
for name, pipeline in fitted_models.items():
    y_test_pred = pipeline.predict(X_test)
    test_f1s[name] = f1_score(y_test, y_test_pred)
best_model_name = max(test_f1s, key=test_f1s.get)
print(f"\nMelhor modelo (por F1 no teste): {best_model_name} -> {test_f1s[best_model_name]:.4f}")
joblib.dump(fitted_models[best_model_name], 'best_model.pkl')

# Cell 7/10 - ROC, AUC e Confusion Matrix (melhor modelo)
best_pipeline = joblib.load('best_model.pkl')
y_test_pred = best_pipeline.predict(X_test)
if hasattr(best_pipeline.named_steps['classifier'], 'predict_proba'):
    y_test_proba = best_pipeline.predict_proba(X_test)[:,1]
else:
    # fallback para decision_function
    try:
        y_test_proba = best_pipeline.decision_function(X_test)
    except Exception:
        y_test_proba = None

print("Classification Report:")
print(classification_report(y_test, y_test_pred, digits=4))

# Confusion matrix
ConfusionMatrixDisplay.from_estimator(best_pipeline, X_test, y_test, display_labels=['Não Cancelado','Cancelado'])
plt.title('Matriz de Confusão - Melhor Modelo')
plt.show()

# ROC curve + AUC
if y_test_proba is not None:
    RocCurveDisplay.from_predictions(y_test, y_test_proba)
    plt.title('Curva ROC - Melhor Modelo')
    plt.show()
    auc = roc_auc_score(y_test, y_test_proba)
    print(f"AUC: {auc:.4f}")
else:
    print("Probabilidades não disponíveis para ROC/AUC.")