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

# Cell 8/10 - SHAP Explainability (OTIMIZADO COM SALVAMENTO)
print("🔍 Iniciando análise de explicabilidade do modelo...")
print("=" * 60)

from sklearn.inspection import permutation_importance

# Configurações otimizadas
SHAP_CACHE_FILE = 'shap_values_cache.pkl'
SAMPLE_SIZE = 50  # Amostra reduzida para SHAP
BACKGROUND_SIZE = 20  # Tamanho do background para KernelExplainer

def load_or_compute_shap():
    """Carrega resultados do cache ou calcula novos"""

    # Verificar se cache existe
    if os.path.exists(SHAP_CACHE_FILE):
        print("📁 Cache encontrado! Carregando resultados SHAP pré-computados...")
        return joblib.load(SHAP_CACHE_FILE)

    print("🔄 Cache não encontrado. Calculando SHAP (pode demorar alguns minutos)...")

    # 1. Carregar modelo e pré-processador
    print("📥 Carregando modelo treinado...")
    best_pipeline = joblib.load('best_model.pkl')
    clf = best_pipeline.named_steps['classifier']
    preproc = best_pipeline.named_steps['preprocessor']

    # 2. Obter nomes das features transformadas
    print("🔧 Preparando nomes das features...")
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    cat_feature_names = preproc.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)
    feature_names_transformed = numeric_cols + list(cat_feature_names)

    print(f"   • {len(numeric_cols)} features numéricas")
    print(f"   • {len(categorical_cols)} features categóricas → {len(cat_feature_names)} após one-hot")
    print(f"   • Total: {len(feature_names_transformed)} features")

    # 3. Transformar dados de teste
    print("🔄 Transformando dados de teste...")
    X_test_transformed = preproc.transform(X_test)
    print(f"   • Shape transformado: {X_test_transformed.shape}")

    # 4. Configurar explainer baseado no tipo de modelo
    print("🤖 Configurando explainer SHAP...")

    try:
        if isinstance(clf, (RandomForestClassifier, xgb.XGBClassifier)):
            print("   • Usando TreeExplainer (mais rápido para modelos baseados em árvores)")
            explainer = shap.TreeExplainer(clf)
            Xshap = X_test_transformed[:SAMPLE_SIZE]
            shap_values = explainer.shap_values(Xshap)

        else:
            print(f"   • Usando KernelExplainer com {BACKGROUND_SIZE} amostras de background")
            background = shap.sample(X_test_transformed, BACKGROUND_SIZE)
            explainer = shap.KernelExplainer(clf.predict_proba, background)
            Xshap = X_test_transformed[:SAMPLE_SIZE]
            shap_values = explainer.shap_values(Xshap)

        print(f"✅ SHAP calculado com sucesso para {SAMPLE_SIZE} amostras")

    except Exception as e:
        print(f"⚠️  Erro no SHAP: {e}")
        print("🔄 Alternando para Permutation Importance...")
        return compute_permutation_importance(clf, X_test_transformed, y_test, feature_names_transformed)

    # 5. Processar resultados SHAP
    print("📊 Processando resultados SHAP...")
    if isinstance(shap_values, list):
        sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        print("   • Modelo de classificação - usando valores da classe positiva")
    else:
        sv = shap_values.values if hasattr(shap_values, 'values') else shap_values
        print("   • Modelo de regressão ou explainer direto")

    # 6. Salvar no cache
    results = {
        'shap_values': sv,
        'Xshap': Xshap,
        'feature_names': feature_names_transformed,
        'explainer_type': 'shap',
        'sample_size': SAMPLE_SIZE
    }

    print(f"💾 Salvando resultados no cache: {SHAP_CACHE_FILE}")
    joblib.dump(results, SHAP_CACHE_FILE)

    return results

def compute_permutation_importance(clf, X_test_transformed, y_test, feature_names):
    """Calcula importância por permutação como fallback"""
    print("🎯 Calculando Permutation Importance...")

    sample_size = min(300, len(X_test_transformed))
    sample_idx = np.random.choice(len(X_test_transformed), size=sample_size, replace=False)

    print(f"   • Amostra: {sample_size} instâncias")
    print("   • Executando permutações...")

    result = permutation_importance(
        clf, X_test_transformed[sample_idx], y_test.iloc[sample_idx],
        n_repeats=3, random_state=42, n_jobs=1 # Alterado n_jobs=-1 para n_jobs=1
    )

    results = {
        'importances': result.importances_mean,
        'feature_names': feature_names,
        'explainer_type': 'permutation',
        'sample_size': sample_size
    }

    print("💾 Salvando resultados de permutation importance...")
    joblib.dump(results, SHAP_CACHE_FILE)

    return results

# EXECUÇÃO PRINCIPAL
print("\n🚀 Executando análise de explicabilidade...")
results = load_or_compute_shap()

print("\n📈 Gerando visualizações...")

if results['explainer_type'] == 'shap':
    # Plot SHAP summary
    print("   • Criando gráfico summary SHAP...")
    shap.summary_plot(
        results['shap_values'],
        results['Xshap'],
        feature_names=results['feature_names'],
        show=False
    )
    plt.title(f"SHAP Summary Plot (Amostra: {results['sample_size']})")
    plt.tight_layout()
    plt.show()

    # Feature importance a partir de SHAP
    print("   • Calculando importância média das features...")
    shap_importance = np.abs(results['shap_values']).mean(0)
    fi_df = pd.DataFrame({
        'feature': results['feature_names'],
        'importance': shap_importance
    }).sort_values('importance', ascending=False).head(15)

    print("\n🏆 Top 15 Features mais importantes (SHAP):")
    display(fi_df)

else:
    # Results from permutation importance
    fi_df = pd.DataFrame({
        'feature': results['feature_names'],
        'importance': results['importances']
    }).sort_values('importance', ascending=False).head(15)

    print("\n🏆 Top 15 Features mais importantes (Permutation Importance):")
    display(fi_df)

    # Plot bar chart
    print("   • Criando gráfico de barras...")
    plt.figure(figsize=(10, 6))
    fi_df.sort_values('importance', ascending=True).plot.barh(
        x='feature', y='importance', legend=False
    )
    plt.title(f"Permutation Importance (Amostra: {results['sample_size']})")
    plt.xlabel('Importância')
    plt.tight_layout()
    plt.show()

print("\n✅ Análise de explicabilidade concluída!")
print(f"💾 Resultados salvos em: {SHAP_CACHE_FILE}")
print("=" * 60)

# Cell 9/10 - Clustering com 10 seeds, k=3; salvar e analisar clusters


df_proc = pd.read_parquet('hotel_bookings_processed.parquet')  # do cell 3
numeric_cols_cluster = df_proc.select_dtypes(include=[np.number]).columns.tolist()

X_cluster = df_proc[numeric_cols_cluster].copy()
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()
X_cluster_imp = imputer.fit_transform(X_cluster)
X_cluster_scaled = scaler.fit_transform(X_cluster_imp)

# Testar KMeans com 10 seeds e fixar n_clusters=3
seeds = [0, 7, 13, 21, 42, 99, 123, 2023, 327, 999]
n_clusters = 3
kmeans_models = {}
sil_scores = {}

for seed in seeds:
    k = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = k.fit_predict(X_cluster_scaled)
    sil = silhouette_score(X_cluster_scaled, labels)
    sil_scores[seed] = sil
    kmeans_models[seed] = {'model': k, 'labels': labels}
    print(f"seed {seed}: silhouette={sil:.4f}")

# Escolher seed com melhor silhouette
best_seed = max(sil_scores, key=sil_scores.get)
best_kmeans = kmeans_models[best_seed]['model']
df_proc['cluster'] = kmeans_models[best_seed]['labels']

print(f"Melhor seed: {best_seed} com silhouette {sil_scores[best_seed]:.4f}")
# Análise por cluster
cluster_analysis = df_proc.groupby('cluster').agg({
    'is_canceled':'mean',
    'adr':'mean',
    'lead_time':'mean',
    'total_guests':'mean',
    'total_nights':'mean',
    'total_of_special_requests':'mean'
}).round(3)
display(cluster_analysis)

# Salvar objetos
joblib.dump(best_kmeans, 'kmeans_best_seed.pkl')
joblib.dump(scaler, 'cluster_scaler.pkl')
joblib.dump(imputer, 'cluster_imputer.pkl')
df_proc.to_parquet('hotel_bookings_analyzed.parquet', index=False)
print("KMeans salvo: kmeans_best_seed.pkl, dados salvos: hotel_bookings_analyzed.parquet")


# Cell 10/10 - DR: PCA, t-SNE, UMAP (visualização)

# Use X_cluster_scaled from previous cell
# PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_cluster_scaled)
print(f"PCA explained var: PC1 {pca.explained_variance_ratio_[0]*100:.2f}% PC2 {pca.explained_variance_ratio_[1]*100:.2f}%")

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df_proc['cluster'], palette='tab10', s=20)
plt.title('PCA - Visualização de 3 clusters')
plt.show()

# t-SNE (pode ser lento) - usar perplexity ~30
tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
X_tsne = tsne.fit_transform(X_cluster_scaled)
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=df_proc['cluster'], palette='tab10', s=20)
plt.title('t-SNE - Visualização de 3 clusters')
plt.show()

# UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_cluster_scaled)
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_umap[:,0], y=X_umap[:,1], hue=df_proc['cluster'], palette='tab10', s=20)
plt.title('UMAP - Visualização de 3 clusters')
plt.show()