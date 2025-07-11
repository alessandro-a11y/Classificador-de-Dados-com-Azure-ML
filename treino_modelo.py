import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# Azure ML
from azureml.core import Workspace, Run

# Conecta ao Workspace
ws = Workspace.from_config()

# Carrega e prepara os dados
df = pd.read_csv("dados.csv")
X = df[["idade", "renda"]]
y = df["comprou"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Treina o modelo
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

# Avalia o modelo
y_pred = modelo.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Acurácia: {acc:.2f}")

# Salva o modelo
os.makedirs("outputs", exist_ok=True)
joblib.dump(modelo, "outputs/modelo.pkl")

# Log no Azure
run = Run.get_context()
run.log("acuracia", acc)
