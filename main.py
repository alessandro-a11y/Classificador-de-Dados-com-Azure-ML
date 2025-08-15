import joblib
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
import pandas as pd
import os

# --- 1. Inicializa o FastAPI ---
# Define metadados para a documentação automática do Swagger UI
app = FastAPI(
    title="API de Predição de Compra",
    description="""
    Uma API simples para prever a probabilidade de um cliente comprar um produto,
    usando um modelo de Machine Learning.
    
    A API foi aprimorada para incluir:
    - Validação de entrada mais robusta.
    - Endpoints de exemplo para facilitar o uso.
    - Melhor documentação no código.
    """,
    version="1.1.0"
)

# --- 2. Define o Schema de Entrada (Pydantic) ---
# Usa Pydantic para validar e estruturar os dados de entrada
class ClientData(BaseModel):
    idade: int
    renda: float

# --- 3. Carrega o Modelo ---
# O modelo é carregado apenas uma vez, na inicialização da API
# Isso otimiza o desempenho para múltiplas requisições
try:
    model_path = "modelo.pkl"
    # Adiciona um caminho relativo mais seguro
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Arquivo do modelo não encontrado: {model_path}")
    
    model = joblib.load(model_path)
    print("Modelo carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}. A API será iniciada, mas o endpoint de predição não funcionará.")
    model = None

# --- 4. Define os Endpoints da API ---

@app.get("/health")
def health_check():
    """Endpoint para verificar a saúde da API e a disponibilidade do modelo."""
    if model:
        return {"status": "ok", "message": "API está funcionando e o modelo foi carregado com sucesso."}
    else:
        raise HTTPException(status_code=500, detail="A API está funcionando, mas o modelo não foi carregado. Verifique os logs de erro.")

@app.get("/example_predict")
def example_predict():
    """
    Endpoint para obter um exemplo de JSON de entrada para a rota /predict.
    
    Isso é útil para quem for testar ou consumir a API.
    """
    example_data = {
        "idade": 35,
        "renda": 50000.0
    }
    return example_data

@app.post("/predict")
def predict(client: ClientData):
    """
    Endpoint para prever a probabilidade de um cliente comprar um produto.
    
    - Recebe um objeto JSON com `idade` (int) e `renda` (float).
    - Retorna a probabilidade de compra, um eco dos dados de entrada e uma explicação.
    """
    # Verifica se o modelo foi carregado corretamente
    if not model:
        raise HTTPException(status_code=500, detail="O modelo de predição não está disponível. Por favor, entre em contato com o administrador.")
    
    # Valida valores de entrada
    if client.idade <= 0 or client.renda <= 0:
        raise HTTPException(
            status_code=400,
            detail="A idade e a renda devem ser valores positivos."
        )
    
    try:
        # Cria um DataFrame a partir dos dados recebidos
        input_data = pd.DataFrame([[client.idade, client.renda]], columns=["idade", "renda"])
        
        # Faz a predição da probabilidade (predict_proba retorna [[prob_classe_0, prob_classe_1]])
        probabilities = model.predict_proba(input_data)[0]
        
        # A probabilidade de compra é a da classe 1
        purchase_probability = probabilities[1]
        
        return {
            "input_data": {
                "idade": client.idade,
                "renda": client.renda
            },
            "probabilidade_de_compra": round(purchase_probability, 4),
            "explicação": "A probabilidade de compra é baseada em idade e renda. Um valor de 1.0 representa 100% de chance de compra, e 0.0 representa 0%."
        }
    except Exception as e:
        # Captura erros de predição e retorna uma mensagem de erro genérica para o usuário
        raise HTTPException(status_code=500, detail=f"Erro interno na predição. Detalhes técnicos: {str(e)}")

# --- 5. Executa a API com Uvicorn (para testes locais) ---
if __name__ == "__main__":
    # O "reload=True" permite que o servidor reinicie automaticamente
    # quando você salva o código, útil para o desenvolvimento.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)