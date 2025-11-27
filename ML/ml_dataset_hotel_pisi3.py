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
import time

# sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate, RandomizedSearchCV
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
from sklearn.metrics import silhouette_score, RocCurveDisplay, ConfusionMatrixDisplay, silhouette_samples

# display
from IPython.display import display

# Links úteis (documentação)
print("Docs úteis:")
print(" scikit-learn: https://scikit-learn.org/stable/")
print(" imbalanced-learn (SMOTE): https://imbalanced-learn.org/stable/")
print(" shap: https://shap.readthedocs.io/")
print(" umap-learn: https://umap-learn.readthedocs.io/")

script_dir = pathlib.Path(__file__).parent
files = {
    'parquet': script_dir / 'hotel_bookings.parquet',
    'csv': script_dir / 'data' / 'hotel_bookings.csv'
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

#Treinamento: cross-val por 10 seeds (agregação de métricas)

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

# Fit final (usar seed 42) e diagnóstico treino vs teste (overfitting)
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

# Otimização acelerada do Random Forest

# Carregar pipeline original
pipelines = joblib.load('pipelines_initial.pkl')
pipeline_rf_tuned = pipelines['rf']

# 📊 ESTRATÉGIA 1: AMOSTRAGEM INTELIGENTE PARA BUSCA DE HIPERPARÂMETROS
print("📊 Criando amostra estratégica (30% dos dados)...")
X_sample, _, y_sample, _ = train_test_split(
    X, y,
    train_size=0.3,           # 30% = 36K registros - suficiente para estimar parâmetros
    stratify=y,               # Manter proporção original das classes
    random_state=42,          # Reprodutibilidade
    shuffle=True
)
print(f"✅ Amostra criada: {X_sample.shape[0]:,} registros de {X.shape[0]:,} originais")

# 🎯 ESTRATÉGIA 2: ESPAÇO DE PARÂMETROS OTIMIZADO PARA DATASETS GRANDES
param_dist_optimized = {
    # ÁRVORES: Balance entre performance e tempo
    'classifier__n_estimators': [100, 120],           # Reduzido - ganho marginal diminui acima de 100

    # PROFUNDIDADE: Controlar overfitting em dados grandes
    'classifier__max_depth': [15, 20],                # Valores moderados para 120K registros

    # REGULARIZAÇÃO: Prevenir overfitting com valores maiores
    'classifier__min_samples_split': [20, 30],        # Aumentado - força generalização
    'classifier__min_samples_leaf': [10, 15],         # Aumentado - folhas mais robustas

    # FEATURES: Diversidade com menos opções
    'classifier__max_features': ['sqrt', 0.3],        # sqrt (default) e 30% - bons trade-offs

    # BOOTSTRAP: Manter para reduzir overfitting
    'classifier__bootstrap': [True]
}

print("🎯 Configuração de parâmetros otimizada:")
print(f"   • Espaço de busca reduzido: 32 combinações (vs 432 original)")
print(f"   • Parâmetros mais restritivos para dataset grande")

# ⚡ ESTRATÉGIA 3: RANDOMIZEDSEARCHCV ACELERADO
print("⚡ Configurando RandomizedSearchCV otimizado...")

random_search = RandomizedSearchCV(
    pipeline_rf_tuned,
    param_dist_optimized,
    n_iter=15,                 # Testar 15 combinações aleatórias (vs 432)
    cv=3,                      # 3-fold CV (vs 5) - suficiente para dados grandes
    scoring='accuracy',         # Métrica clara de avaliação
    n_jobs=-1,                 # Paralelizar em TODOS os cores
    random_state=42,           # Reprodutibilidade
    verbose=2,                 # Monitoramento detalhado
    return_train_score=True    # Analisar overfitting
)

# 🕒 EXECUÇÃO COM TIMING
print("🚀 Iniciando busca de hiperparâmetros...")
print("⏰ Estimativa: 30-90 minutos (vs 6+ horas original)")
start_time = time.time()

random_search.fit(X_sample, y_sample)

end_time = time.time()
execution_minutes = (end_time - start_time) / 60
print(f"✅ Busca concluída em {execution_minutes:.1f} minutos")

# 📈 ESTRATÉGIA 4: TREINO FINAL COM TODOS OS DADOS
print("📈 Treinando modelo final com melhores parâmetros...")
best_pipeline = random_search.best_estimator_

# Agora sim usar TODOS os dados com os melhores parâmetros encontrados
best_pipeline.fit(X, y)

# 💾 SALVAR RESULTADOS
print("💾 Salvando resultados...")
import joblib

# Salvar modelo otimizado
joblib.dump(best_pipeline, 'random_forest_optimized.pkl')
joblib.dump(random_search, 'random_search_results.pkl')

# 📊 RELATÓRIO FINAL
print("\n" + "="*60)
print("🎉 OTIMIZAÇÃO CONCLUÍDA COM SUCESSO!")
print("="*60)
print(f"🏆 Melhores parâmetros encontrados:")
for param, value in random_search.best_params_.items():
    print(f"   • {param}: {value}")

print(f"📊 Melhor score na validação: {random_search.best_score_:.4f}")
print(f"⏰ Tempo total: {execution_minutes:.1f} minutos")
print(f"💾 Modelo salvo: 'random_forest_optimized.pkl'")

# 🔍 ANALISAR OVERFITTING
print("\n🔍 Análise de overfitting:")
train_score = random_search.cv_results_['mean_train_score'][random_search.best_index_]
test_score = random_search.best_score_
gap = train_score - test_score
print(f"   • Score treino: {train_score:.4f}")
print(f"   • Score validação: {test_score:.4f}")
print(f"   • Gap (overfitting): {gap:.4f}")

print("\n✅ Processo concluído! Use o modelo salvo para fazer previsões.")

# Carregar o modelo otimizado
best_rf_model = joblib.load('random_forest_optimized.pkl')

# Avaliar no conjunto de treino
y_train_pred_tuned = best_rf_model.predict(X_train)
y_train_proba_tuned = best_rf_model.predict_proba(X_train)[:,1]
train_f1_tuned = f1_score(y_train, y_train_pred_tuned)
train_acc_tuned = accuracy_score(y_train, y_train_pred_tuned)

# Avaliar no conjunto de teste
y_test_pred_tuned = best_rf_model.predict(X_test)
y_test_proba_tuned = best_rf_model.predict_proba(X_test)[:,1]
test_f1_tuned = f1_score(y_test, y_test_pred_tuned)
test_acc_tuned = accuracy_score(y_test, y_test_pred_tuned)

print(f"\n--- RandomForest Otimizado ---")
print(f" Train Acc: {train_acc_tuned:.4f} F1: {train_f1_tuned:.4f}")
print(f" Test Acc: {test_acc_tuned:.4f} F1: {test_f1_tuned:.4f}")

gap_tuned = train_f1_tuned - test_f1_tuned
if gap_tuned > 0.05: # Um gap menor que 0.10 é geralmente aceitável, vamos usar 0.05 para ser mais rigoroso
    print(f"  ⚠️ Possível overfitting ainda presente (gap F1 train - test = {gap_tuned:.4f})")
else:
    print(f"  ✓ Gap aceitável (gap F1 train - test = {gap_tuned:.4f})")

print("\nRelatório de Classificação no Teste:")
print(classification_report(y_test, y_test_pred_tuned))

# ROC, AUC e Confusion Matrix (melhor modelo)
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

# SHAP Explainability (OTIMIZADO COM SALVAMENTO)
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

# Preparação de Dados para Clusterização: Carregamento, Imputação e Escalonamento

# 1. Carregue o arquivo 'hotel_bookings_processed.parquet'
df_cluster = pd.read_parquet('hotel_bookings_processed.parquet')
print(f"DataFrame 'df_cluster' carregado com {df_cluster.shape[0]} linhas e {df_cluster.shape[1]} colunas.")

# 2. Selecione apenas as colunas numéricas
X_cluster_raw = df_cluster.select_dtypes(include=[np.number])
print(f"'X_cluster_raw' criado com {X_cluster_raw.shape[1]} colunas numéricas.")

# 3. Instancie e aplique SimpleImputer
imputer = SimpleImputer(strategy='median')
X_cluster_imputed = imputer.fit_transform(X_cluster_raw)
print("Valores ausentes imputados usando a mediana.")

# 4. Instancie e aplique StandardScaler
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster_imputed)
print("Dados escalados usando StandardScaler.")

# 5. Salve o SimpleImputer e o StandardScaler
joblib.dump(imputer, 'cluster_imputer.pkl')
joblib.dump(scaler, 'cluster_scaler.pkl')
print("Imputer e Scaler salvos como 'cluster_imputer.pkl' e 'cluster_scaler.pkl'.")

print("Preparação de dados para clusterização concluída.")

# Cálculo do SSE (Elbow Method) e do Silhouette Score para K-Means para determinar o número ideal de clusters.
sse = []
silhouette_scores = []

# Define o range de K valores para testar
k_range_sse = range(1, 21)
k_range_silhouette = range(2, 21)

print("Calculando SSE para K-Means...")
for k in k_range_sse:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_cluster_scaled)
    sse.append(kmeans.inertia_)
    print(f"  K={k}: SSE={kmeans.inertia_:.2f}")

print("\nCalculando o Silhouette Scores para K-Means...")
for k in k_range_silhouette:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_cluster_scaled)
    if len(np.unique(labels)) > 1:
        score = silhouette_score(X_cluster_scaled, labels)
        silhouette_scores.append(score)
        print(f"  K={k}: Silhouette Score={score:.4f}")
    else:
        silhouette_scores.append(0.0)
        print(f"  K={k}: Silhouette Score=N/A (apenas um cluster encontrado)")

print("Cálculos completos. Pronto para plotar.")

# Plote dos resultados do Método do Cotovelo (SSE) e do Silhouette Score para K-Means.

# Plotando o Método do Cotovelo (SSE)
plt.figure(figsize=(10, 6))
plt.plot(k_range_sse, sse, marker='o', linestyle='--')
plt.title('Elbow Method para Determinar K Ótimo (SSE)')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Soma dos Quadrados dos Erros (SSE)')
plt.xticks(list(k_range_sse))
plt.grid(True)
plt.show()

# Plotando o Silhuette Score
plt.figure(figsize=(10, 6))
plt.plot(k_range_silhouette, silhouette_scores, marker='o', linestyle='--')
plt.title('Silhouette Score para Determinar K Ótimo')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Silhouette Score')
plt.xticks(list(k_range_silhouette))
plt.grid(True)
plt.show()

print("Visualização dos scores SSE e Silhouette concluída.")

# CLUSTERIZAÇÃO COM TÉCNICAS K-MEANS E DBSCAN

print("\nIniciando clusterização com K-Means e DBSCAN…")

# =========================================================
# 1) K-MEANS (com 3 clusters e random_state fixo)
# =========================================================

from sklearn.cluster import KMeans # Garante que KMeans está importado
from sklearn.metrics import silhouette_score # Garante que silhouette_score está importado

print("\nExecutando K-Means com 3 clusters...")
n_clusters_kmeans = 3
kmeans_final = KMeans(n_clusters=n_clusters_kmeans, random_state=42, n_init=10)
labels_k = kmeans_final.fit_predict(X_cluster_scaled)
sil_kmeans = silhouette_score(X_cluster_scaled, labels_k)
df_cluster["cluster_kmeans"] = labels_k

print(f"K-Means \u2192 Silhouette Score: {sil_kmeans:.4f}")


# =========================================================
# 2) DBSCAN
# =========================================================

from sklearn.cluster import DBSCAN

eps_fixed = 2.4
min_s_fixed = 10

print(f"\nExecutando DBSCAN com parâmetros fixos: eps={eps_fixed}, min_samples={min_s_fixed}...")

db = DBSCAN(eps=eps_fixed, min_samples=min_s_fixed, n_jobs=-1)
labels_d = db.fit_predict(X_cluster_scaled)

# DBSCAN pode retornar tudo como ruídos (-1) ou um único cluster (0) evitar isso para silhouette
unique_labels = len(set(labels_d)) - (1 if -1 in labels_d else 0) # Ignora ruídos
if unique_labels > 1: # Precisa de pelo menos 2 clusters válidos para Silhouette Score
    sil_dbscan_fixed = silhouette_score(X_cluster_scaled, labels_d)
    df_cluster["cluster_dbscan"] = labels_d
    print(f"DBSCAN  (eps={eps_fixed}, min_samples={min_s_fixed}) | Silhouette: {sil_dbscan_fixed:.4f}")
else:
    print(f"DBSCAN  (eps={eps_fixed}, min_samples={min_s_fixed}) | Insuficiente clusters válidos para Silhouette.")
    df_cluster["cluster_dbscan"] = -1 # Atribui -1 para todos se nenhum cluster válido for encontrado

# Salvar clusterização
df_cluster.to_parquet("hotel_bookings_clustered.parquet", index=False)
print("\nClusterização concluída. Resultados salvos em 'hotel_bookings_clustered.parquet'.")

# VISUALIZAÇÃO DOS CLUSTERS
print("\nGerando visualizações dos clusters…")

df_cluster = pd.read_parquet("hotel_bookings_clustered.parquet")
X_scaled = X_cluster_scaled

# ============================================
# PCA para projeção 2D
# ============================================
pca = PCA(n_components=2, random_state=42)
Xpca = pca.fit_transform(X_scaled)

# ---------------------------------------------------
# K-Means
# ---------------------------------------------------
plt.figure(figsize=(7,5))
sns.scatterplot(x=Xpca[:,0], y=Xpca[:,1], hue=df_cluster["cluster_kmeans"], palette="tab10")
plt.title("K-Means — PCA Visualization")
plt.show()

# ============================================================
# Gráfico de DENSIDADE (DBSCAN)
# ============================================================

plt.figure(figsize=(8,6))
sns.kdeplot(
    x=Xpca[:,0],
    y=Xpca[:,1],
    fill=True,
    cmap="viridis",
    thresh=0.03,
    levels=50
)
plt.scatter(Xpca[:,0], Xpca[:,1], c=df_cluster["cluster_dbscan"], s=5, cmap="tab10")
plt.title("DBSCAN — Densidade + PCA")
plt.show()

# ============================================================
# Visualização t-SNE
# ============================================================

tsne = TSNE(n_components=2, random_state=42)
Xtsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(7,5))
sns.scatterplot(x=Xtsne[:,0], y=Xtsne[:,1], hue=df_cluster["cluster_kmeans"], palette="tab10")
plt.title("K-Means — t-SNE Visualization")
plt.show()

# ============================================================
# Visualização UMAP
# ============================================================

um = umap.UMAP(n_components=2, random_state=42)
Xum = um.fit_transform(X_scaled)

plt.figure(figsize=(7,5))
sns.scatterplot(x=Xum[:,0], y=Xum[:,1], hue=df_cluster["cluster_kmeans"], palette="tab10")
plt.title("K-Means — UMAP Visualization")
plt.show()

print("\nVisualizações finalizadas com sucesso.")

# Plot detalhado da Silhueta para 2, 3, 6, 10 e 13 clusters

import matplotlib.cm as cm
# Valores de K para os quais queremos gerar o plot detalhado da silhueta
requested_ks = [2, 3, 6, 10, 13]

for n_clusters in requested_ks:
    # Cria uma subplot com 1 linha e 1 coluna para o plot da silhueta
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(12, 7)

    # O primeiro plot é o gráfico da silhueta
    # A escala do eixo x vai de -1 a 1, mas como os scores de silhueta são geralmente positivos,
    # vamos usar uma escala mais apropriada para a maioria dos casos.
    ax1.set_xlim([-0.1, 1])
    #adicionar espaço em branco entre os clusters para o plot
    ax1.set_ylim([0, len(X_cluster_scaled) + (n_clusters + 1) * 10])

    # Inicializa o clusterizador com n_clusters e um random_state para reprodutibilidade
    clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = clusterer.fit_predict(X_cluster_scaled)

    # Calcula o silhouette_score médio para todos os dados
    silhouette_avg = silhouette_score(X_cluster_scaled, cluster_labels)
    print(f"Para n_clusters = {n_clusters}, o score médio de silhueta de: {silhouette_avg:.4f}")

    # Calcula os scores de silhueta para cada amostra
    sample_silhouette_values = silhouette_samples(X_cluster_scaled, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Agregue os scores de silhueta para as amostras pertencentes ao cluster i, e ordene-os
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Rotula os plots de silhueta com seus números de cluster no meio
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Calcula o novo y_lower para o próximo plot
        y_lower = y_upper + 10  # 10 para as 0 amostras

    ax1.set_title(f"Plot de Silhueta para {n_clusters} Clusters", fontweight='bold')
    ax1.set_xlabel("Coeficientes de Silhueta")
    ax1.set_ylabel("Rótulo do Cluster")

    # A linha vertical para o score de silhueta médio de todos os valores
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--", label=f'Média: {silhouette_avg:.2f}')
    ax1.legend()

    plt.suptitle(f"Análise de Silhueta para k = {n_clusters}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'silhouette_plot_k{n_clusters}_standalone.png', dpi=300, bbox_inches='tight')
    print(f"   \u2705 Gráfico salvo: silhouette_plot_k{n_clusters}_standalone.png")
    plt.show()

# análise dos clusters

df_proc = pd.read_parquet('hotel_bookings_processed.parquet')  # do cell 3
numeric_cols_cluster = df_proc.select_dtypes(include=[np.number]).columns.tolist()

X_cluster = df_proc[numeric_cols_cluster].copy()
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()
X_cluster_imp = imputer.fit_transform(X_cluster)
X_cluster_scaled = scaler.fit_transform(X_cluster_imp)

n_clusters = 3
# Aplicar KMeans diretamente com 3 clusters
kmeans_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_cluster_scaled)
df_proc['cluster'] = cluster_labels

# Calcular silhouette score para o modelo final
silhouette_avg = silhouette_score(X_cluster_scaled, cluster_labels)
print(f"KMeans com {n_clusters} clusters (random_state=42) - Silhouette Score: {silhouette_avg:.4f}")

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
joblib.dump(kmeans_final, 'kmeans_final_3_clusters.pkl')
joblib.dump(scaler, 'cluster_scaler.pkl')
joblib.dump(imputer, 'cluster_imputer.pkl')
df_proc.to_parquet('hotel_bookings_analyzed.parquet', index=False)
print("KMeans salvo: kmeans_final_3_clusters.pkl, dados salvos: hotel_bookings_analyzed.parquet")
