
# 📊 Classificador de Dados com Azure ML

Este projeto demonstra como treinar um modelo de machine learning usando o **Azure Machine Learning** e Python. Ele inclui leitura de dados, pré-processamento e treino de um modelo de classificação.

---

## 📁 Estrutura

- `dados.csv` — Base de dados de entrada.
- `treino_modelo.py` — Script principal de treino do modelo.
- `requirements.txt` — Dependências Python do projeto.

---

## 🚀 Como executar

### 1. Instale os requisitos:
```bash
pip install -r requirements.txt
```

### 2. Execute o script de treino:
```bash
python treino_modelo.py
```

O script realiza:
- Carregamento dos dados (`dados.csv`)
- Pré-processamento
- Divisão em treino/teste
- Treinamento de modelo (ex: Random Forest ou similar)
- Avaliação de desempenho
- Registro do modelo (caso Azure ML esteja configurado)

---

## ☁️ Integração com Azure Machine Learning

Você pode adaptar esse projeto para rodar diretamente em um workspace do **Azure ML**, incluindo:
- Envio de experimentos
- Registro de modelos no workspace
- Deploy como serviço web

---

## 🧪 Requisitos

- Python 3.8+
- Bibliotecas: scikit-learn, pandas, azureml-core (opcional)

---

## ✍️ Contribuições

Sinta-se à vontade para abrir issues, sugerir melhorias ou contribuir com novos recursos.

---

## 📄 Licença

Este projeto está licenciado sob a [MIT License](LICENSE).
