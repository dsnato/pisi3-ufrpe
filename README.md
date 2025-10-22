# 🏨 Hotel Booking Demand - Análise e Previsão de Cancelamentos

> Projeto da Disciplina de PISI 3 - UFRPE  
> **Análise Exploratória de Dados (EDA) + Machine Learning (ML) + Dashboard Interativo (Dash)**

## 📋 Índice

1. [Sobre o Projeto](#sobre-o-projeto)
2. [Tecnologias Utilizadas](#tecnologias-utilizadas)
3. [Estrutura do Projeto](#estrutura-do-projeto)
4. [Como Começar](#como-começar)
   - [Pré-requisitos](#pré-requisitos)
   - [Instalação](#instalação)
5. [Como Executar](#como-executar)
   - [EDA - Análise Exploratória](#eda---análise-exploratória)
   - [ML - Machine Learning](#ml---machine-learning)
   - [DASH - Dashboard Interativo](#dash---dashboard-interativo)
6. [Dataset](#dataset)
7. [Estrutura de Pastas](#estrutura-de-pastas)
8. [Contribuidores](#contribuidores)

---

## 🎯 Sobre o Projeto

Este projeto realiza uma **análise completa** de dados de reservas de hotéis, com foco em prever cancelamentos de reservas. O projeto está dividido em três módulos principais:

### 📊 **EDA (Exploratory Data Analysis)**
Análise exploratória dos dados para entender padrões, tendências e características do dataset de reservas de hotéis. Inclui:
- Análise estatística descritiva
- Visualizações de distribuições
- Identificação de correlações
- Análise de valores faltantes
- Identificação de outliers

### 🤖 **ML (Machine Learning)**
Desenvolvimento de modelos de Machine Learning para prever cancelamentos de reservas. Inclui:
- Pré-processamento de dados
- Feature Engineering
- Treinamento de múltiplos modelos (Random Forest, XGBoost, Logistic Regression, Gradient Boosting)
- Otimização de hiperparâmetros
- Análise de importância de features
- Avaliação de métricas (Acurácia, Precisão, Recall, F1-Score)

### 📈 **DASH (Dashboard Interativo)**
Dashboard web interativo para visualização dos resultados da análise e do modelo. Inclui:
- Visão geral dos dados
- Visualizações interativas com Plotly
- Análise de desempenho dos modelos
- Análise de clusters de clientes
- Interface responsiva e intuitiva

---

## 🛠️ Tecnologias Utilizadas

### **Linguagem**
- Python 3.12+

### **Principais Bibliotecas**

#### Data Science & Machine Learning
- **pandas** - Manipulação e análise de dados
- **numpy** - Operações numéricas
- **scikit-learn** - Algoritmos de Machine Learning
- **xgboost** - Algoritmo de Gradient Boosting otimizado

#### Visualização
- **matplotlib** - Gráficos estáticos
- **seaborn** - Visualizações estatísticas
- **plotly** - Gráficos interativos
- **missingno** - Visualização de dados faltantes

#### Dashboard
- **dash** - Framework para criação de dashboards
- **dash-bootstrap-components** - Componentes Bootstrap para Dash

#### Auxiliares
- **scipy** - Funções científicas e estatísticas
- **joblib** - Serialização de modelos
- **pyarrow** - Leitura/escrita de arquivos Parquet

---

## 📁 Estrutura do Projeto

```
pisi3-ufrpe/
│
├── 📂 EDA/                          # Análise Exploratória de Dados
│   ├── eda_dataset_pisi3.py         # Script principal de EDA
│   └── notebook/                    # Notebooks Jupyter
│       └── eda_dataset_hotel_pisi3.ipynb
│
├── 📂 ML/                           # Machine Learning
│   ├── ml_dataset_hotel_pisi3.py    # Script de treinamento ML
│   └── data/                        # Dados do projeto
│       └── hotel_bookings.csv       # Dataset original
│
├── 📂 DASH/                         # Dashboard Interativo
│   └── dash_interativo_ml.py        # Aplicação Dash
│
├── 📄 requirements.txt              # Dependências do projeto
└── 📄 README.md                     # Este arquivo
```

---

## 🚀 Como Começar

### Pré-requisitos

Antes de começar, você precisa ter instalado em sua máquina:

1. **Python 3.12 ou superior**
   - Download: https://www.python.org/downloads/
   - Durante a instalação, marque a opção "Add Python to PATH"

2. **Git** (para clonar o repositório)
   - Download: https://git-scm.com/downloads

3. **Editor de código** (recomendado)
   - VS Code: https://code.visualstudio.com/
   - PyCharm: https://www.jetbrains.com/pycharm/

### Instalação

Siga estes passos para configurar o projeto em sua máquina:

#### 1️⃣ Clone o repositório

Abra o terminal (PowerShell no Windows, Terminal no Mac/Linux) e execute:

```bash
git clone https://github.com/dsnato/pisi3-ufrpe.git
cd pisi3-ufrpe
```

#### 2️⃣ Crie um ambiente virtual

O ambiente virtual isola as dependências do projeto. Execute:

**No Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**No Mac/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

> 💡 **Nota:** Quando o ambiente virtual estiver ativo, você verá `(.venv)` no início da linha de comando.

#### 3️⃣ Instale as dependências

Com o ambiente virtual ativo, instale todas as bibliotecas necessárias:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⏱️ **Isso pode levar alguns minutos**, pois várias bibliotecas serão baixadas e instaladas.

#### 4️⃣ Verifique a instalação

Para garantir que tudo foi instalado corretamente, execute:

```bash
pip list
```

Você deve ver todas as bibliotecas listadas no arquivo `requirements.txt`.

---

## ▶️ Como Executar

### 📊 EDA - Análise Exploratória

A análise exploratória examina o dataset e gera visualizações para entender melhor os dados.

#### **Opção 1: Executar o script Python**

```bash
cd EDA
python eda_dataset_pisi3.py
```

**O que este script faz:**
1. Carrega o dataset de reservas de hotéis
2. Realiza análise estatística descritiva
3. Gera gráficos de distribuições
4. Identifica correlações entre variáveis
5. Analisa valores faltantes e outliers
6. Salva visualizações em arquivos de imagem

**Saída esperada:**
- Gráficos salvos na pasta `EDA/`
- Estatísticas impressas no console
- Arquivo `hotel_bookings.parquet` (formato otimizado)

#### **Opção 2: Usar o Jupyter Notebook (Recomendado para exploração)**

```bash
cd EDA/notebook
jupyter notebook eda_dataset_hotel_pisi3.ipynb
```

O notebook permite executar o código célula por célula e visualizar os resultados interativamente.

---

### 🤖 ML - Machine Learning

O módulo de Machine Learning treina modelos para prever cancelamentos de reservas.

#### **Executar o pipeline completo de ML**

```bash
cd ML
python ml_dataset_hotel_pisi3.py
```

**O que este script faz:**

**Etapa 1: Pré-processamento**
- Carrega e limpa os dados
- Cria features derivadas (`total_guests`, `total_nights`, etc.)
- Trata valores faltantes
- Remove outliers
- Divide dados em treino (80%) e teste (20%)

**Etapa 2: Treinamento de Modelos**
- Random Forest Classifier
- XGBoost Classifier
- Logistic Regression
- Gradient Boosting Classifier

**Etapa 3: Avaliação**
- Calcula métricas: Acurácia, Precisão, Recall, F1-Score
- Gera matriz de confusão
- Identifica o melhor modelo

**Etapa 4: Otimização**
- Otimiza hiperparâmetros do melhor modelo
- Usa RandomizedSearchCV para busca eficiente

**Etapa 5: Análise de Features**
- Identifica features mais importantes
- Gera visualizações de importância

**Saída esperada:**
- Modelos salvos em arquivos `.pkl`
- Gráficos de comparação de modelos
- Relatório de classificação
- Análise de importância de features

**Arquivos gerados:**
```
ML/
├── X_train.pkl                      # Dados de treino
├── X_test.pkl                       # Dados de teste
├── y_train.pkl                      # Labels de treino
├── y_test.pkl                       # Labels de teste
├── preprocessor.pkl                 # Pipeline de pré-processamento
├── best_model.pkl                   # Melhor modelo treinado
├── optimized_rf_model.pkl           # Modelo otimizado
├── feature_importance.csv           # Importância das features
├── confusion_matrix.png             # Matriz de confusão
├── feature_importance.png           # Gráfico de importância
└── model_comparison.png             # Comparação de modelos
```

---

### 📈 DASH - Dashboard Interativo

O dashboard oferece uma interface web interativa para explorar os dados e resultados dos modelos.

#### **Executar o dashboard**

```bash
cd DASH
python dash_interativo_ml.py
```

**Aguarde a mensagem:**
```
Dash is running on http://127.0.0.1:8050/
```

#### **Acessar o dashboard**

Abra seu navegador e acesse:
```
http://127.0.0.1:8050/
```

ou

```
http://localhost:8050/
```

**O que você verá no Dashboard:**

🏠 **Tab: Visão Geral**
- Distribuição de tipos de hotéis
- Reservas por mês
- Análise de ADR (Average Daily Rate)
- Top 10 países de origem
- Relação entre lead time e cancelamentos

🤖 **Tab: Análise de Modelos**
- Comparação de desempenho entre modelos
- Métricas de avaliação (Acurácia, Precisão, Recall, F1-Score)
- Top 10 features mais importantes
- Matriz de confusão

🔮 **Tab: Análise de Clusters**
- Segmentação de clientes usando K-Means
- Características de cada cluster
- Visualização PCA 2D dos clusters
- Análise de perfis de cancelamento

**Para parar o dashboard:**
- Pressione `Ctrl + C` no terminal

---

## 📊 Dataset

### **Hotel Booking Demand**

- **Fonte:** Kaggle - Hotel Booking Demand Dataset
- **Link:** https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
- **Tamanho:** ~119.000 reservas
- **Período:** 2015-2017
- **Tipos de hotéis:** Resort Hotel e City Hotel

### **Principais variáveis:**

| Variável | Descrição |
|----------|-----------|
| `is_canceled` | Se a reserva foi cancelada (0 = Não, 1 = Sim) - **TARGET** |
| `hotel` | Tipo de hotel (Resort Hotel ou City Hotel) |
| `lead_time` | Dias entre a reserva e a chegada |
| `arrival_date_*` | Data de chegada (ano, mês, semana, dia) |
| `stays_in_*_nights` | Número de noites (fim de semana e semana) |
| `adults`, `children`, `babies` | Número de hóspedes |
| `country` | País de origem |
| `market_segment` | Segmento de mercado |
| `distribution_channel` | Canal de distribuição |
| `is_repeated_guest` | Cliente repetido |
| `previous_cancellations` | Cancelamentos anteriores |
| `reserved_room_type` | Tipo de quarto reservado |
| `assigned_room_type` | Tipo de quarto atribuído |
| `booking_changes` | Número de alterações na reserva |
| `deposit_type` | Tipo de depósito |
| `adr` | Average Daily Rate (taxa diária média) |
| `total_of_special_requests` | Número de pedidos especiais |

### **Como obter o dataset:**

O dataset já está incluído na pasta `ML/data/`. Caso precise baixá-lo novamente:

1. Instale a API do Kaggle:
   ```bash
   pip install kaggle
   ```

2. Configure suas credenciais do Kaggle:
   - Acesse: https://www.kaggle.com/settings
   - Clique em "Create New API Token"
   - Salve o arquivo `kaggle.json` em `~/.kaggle/` (Linux/Mac) ou `C:\Users\<seu-usuario>\.kaggle\` (Windows)

3. Baixe o dataset:
   ```bash
   kaggle datasets download -d jessemostipak/hotel-booking-demand
   ```

---

## 🎓 Para Estudantes

### **Conceitos abordados no projeto:**

#### **Análise de Dados (EDA)**
- Estatística descritiva
- Visualização de dados
- Identificação de padrões
- Tratamento de dados faltantes
- Detecção de outliers
- Análise de correlação

#### **Machine Learning**
- Aprendizado supervisionado
- Classificação binária
- Validação cruzada
- Divisão treino/teste
- Métricas de avaliação
- Otimização de hiperparâmetros
- Feature engineering
- Ensemble methods

#### **Engenharia de Software**
- Controle de versão (Git)
- Ambientes virtuais
- Gerenciamento de dependências
- Modularização de código
- Documentação

#### **Visualização e Dashboard**
- Gráficos interativos
- Interface web com Python
- UX/UI para análise de dados

---

## 🐛 Solução de Problemas Comuns

### **Problema 1: Erro ao ativar ambiente virtual no Windows**
```
.venv\Scripts\Activate.ps1 : File cannot be loaded because running scripts is disabled
```

**Solução:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### **Problema 2: Módulo não encontrado**
```
ModuleNotFoundError: No module named 'pandas'
```

**Solução:**
- Certifique-se de que o ambiente virtual está ativo (`(.venv)` deve aparecer no terminal)
- Reinstale as dependências: `pip install -r requirements.txt`

### **Problema 3: Dashboard não abre no navegador**

**Solução:**
- Verifique se a porta 8050 está livre
- Tente acessar manualmente: http://127.0.0.1:8050/
- Verifique se não há erros no terminal

### **Problema 4: Dataset não encontrado**

**Solução:**
- Certifique-se de que o arquivo `hotel_bookings.csv` está em `ML/data/`
- Ou execute o script a partir da pasta correta (use `cd` para navegar)

---

## 📚 Referências

- [Documentação Pandas](https://pandas.pydata.org/docs/)
- [Documentação Scikit-learn](https://scikit-learn.org/stable/documentation.html)
- [Documentação Plotly](https://plotly.com/python/)
- [Documentação Dash](https://dash.plotly.com/)
- [Dataset Original - Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)

---

## 👥 Contribuidores
- **Disciplina:** PISI 3 - UFRPE
- **Instituição:** Universidade Federal Rural de Pernambuco
- **Membros:** Douglas Wesley, Elton Oliveira, Júlia Karolyne, Renato Samico, Weslley Gabriel

---
