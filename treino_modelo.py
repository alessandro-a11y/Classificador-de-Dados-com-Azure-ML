import pandas as pd
import numpy as np
import joblib
import os
import argparse
import logging
from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler

from azureml.core import Workspace, Run
from azureml.core.model import Model
from azureml.data.datareference import DataReference
from azureml.data.dataset_factory import TabularDatasetFactory

# --- Configuração de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Funções para MLOps ---

def get_azure_run_context() -> Run:
    """Obtém e retorna o contexto de execução do Azure Machine Learning."""
    try:
        run = Run.get_context()
        logging.info("Contexto de execução do Azure ML obtido com sucesso.")
        return run
    except Exception as e:
        logging.error(f"Erro ao obter o contexto de execução: {e}")
        return None

def load_data(data_path: str) -> pd.DataFrame:
    """Carrega os dados de um arquivo CSV ou de um Dataset do Azure ML."""
    try:
        if data_path.startswith('azureml://'):
            ws = Workspace.from_config()
            dataset = TabularDatasetFactory.from_delimited_files(path=[(ws.datastores['workspaceblobstore'], data_path.replace('azureml://', ''))])
            df = dataset.to_pandas_dataframe()
        else:
            df = pd.read_csv(data_path)
        logging.info(f"Dados carregados com sucesso. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Erro ao carregar os dados de '{data_path}': {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Realiza o pré-processamento dos dados, incluindo a separação de
    features e target, e o tratamento de valores nulos (se necessário).
    """
    if df.isnull().values.any():
        df = df.dropna()  # Exemplo simples de tratamento de nulos
        logging.warning("Valores nulos encontrados e removidos.")

    # Normalização/Escalonamento
    scaler = StandardScaler()
    features = ['idade', 'renda']
    df[features] = scaler.fit_transform(df[features])

    X = df[features]
    y = df['comprou']
    logging.info("Pré-processamento e escalonamento dos dados concluídos.")
    return X, y

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """Treina o modelo de Regressão Logística."""
    logging.info("Iniciando o treinamento do modelo...")
    modelo = LogisticRegression(solver='liblinear', random_state=42)
    modelo.fit(X_train, y_train)
    logging.info("Treinamento do modelo concluído.")
    return modelo

def evaluate_model(modelo: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series, run: Run):
    """
    Avalia o modelo e registra métricas detalhadas no Azure ML Run.
    """
    y_pred = modelo.predict(X_test)
    y_pred_proba = modelo.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    logging.info(f"Acurácia: {acc:.4f}")
    logging.info(f"F1-Score: {f1:.4f}")
    logging.info(f"AUC-ROC: {roc_auc:.4f}")

    # Log de métricas no Azure ML
    if run:
        run.log('accuracy', acc)
        run.log('f1_score', f1)
        run.log('roc_auc', roc_auc)
        run.log_table('classification_report', classification_report(y_test, y_pred, output_dict=True))

    logging.info("Métricas registradas com sucesso no Azure ML.")

def save_and_register_model(modelo: LogisticRegression, model_name: str, run: Run):
    """Salva o modelo e o registra no Azure ML Workspace."""
    model_path = os.path.join('outputs', f'{model_name}.pkl')
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(modelo, model_path)
    logging.info(f"Modelo salvo em '{model_path}'")

    if run:
        # Registrar o modelo no Azure ML
        run.upload_file(model_path, model_path)
        registered_model = Model.register(
            workspace=run.experiment.workspace,
            model_path=model_path,
            model_name=model_name,
            description="Modelo de Regressão Logística para previsão de compra."
        )
        logging.info(f"Modelo registrado com o nome '{registered_model.name}' e versão {registered_model.version}.")

# --- Função Principal de Execução ---
def main():
    parser = argparse.ArgumentParser(description='Script de treinamento de modelo de ML.')
    parser.add_argument('--data_path', type=str, default='dados.csv', help='Caminho para o arquivo de dados.')
    args = parser.parse_args()

    run = get_azure_run_context()

    try:
        # 1. Carregar e Pré-processar os dados
        df = load_data(args.data_path)
        X, y = preprocess_data(df)

        # 2. Dividir os dados
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        logging.info("Dados divididos em conjuntos de treino e teste.")

        # 3. Treinar o modelo
        modelo = train_model(X_train, y_train)

        # 4. Avaliar o modelo
        evaluate_model(modelo, X_test, y_test, run)

        # 5. Salvar e Registrar o modelo
        save_and_register_model(modelo, "modelo-compra", run)

    except Exception as e:
        logging.error(f"Ocorreu um erro no pipeline de execução: {e}")
        raise

if __name__ == "__main__":
    main()