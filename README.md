# Support Copilot — Datathon Fase 5 (FIAP MLET)

Plataforma de suporte híbrido: um **modelo clássico** classifica o chamado em uma de 10 filas e
decide se ele vai para o **assistente RAG** sobre um manual em PDF, com **guardrails**,
**monitoramento de drift** e empacotamento em **Docker**.

> Em construção. Estado atual: dados (D1), modelo + MLflow (D2) e API de triagem (D3).
> O que já foi feito, dia a dia, está em [`docs/diario.md`](docs/diario.md); os achados com
> números, em [`docs/findings.md`](docs/findings.md).

## Como rodar (macOS / Linux)

Pré-requisitos: [uv](https://docs.astral.sh/uv/) e Python 3.11 (o `uv` instala se faltar).
No macOS, o XGBoost precisa do OpenMP: `brew install libomp`.

```bash
uv sync                                            # ambiente
uv run python -m support_copilot.data.prepare      # baixa o dataset (SHA-256) e gera o split
uv run uvicorn support_copilot.api.main:app --port 8000
```

Swagger em http://localhost:8000/docs. O modelo campeão está versionado em `models/`, então a API
sobe sem treinar nada.

Para retreinar e ver os experimentos no MLflow (porta **5001**, não 5000: no macOS a 5000 é do
AirPlay):

```bash
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5001
uv run python -m support_copilot.model.train      # outro terminal; ~8 min
```

## Endpoints

| método | rota | o que faz |
|---|---|---|
| GET | `/health` | status e versão do modelo |
| POST | `/predict` | `{"text": "..."}` (1 a 1000 caracteres) → fila, confiança, `is_technical`, rota (`rag` ou `queue`) |

## Resultado atual do classificador

| | valor |
|---|---|
| F1 macro (teste sem vazamento de paráfrases) | 0,371 |
| F1 do roteamento técnico × não técnico | 0,693 |
| Classe majoritária | 0,045 |

Por que esses números são menores que os de um split aleatório, e por que este é o número honesto:
[`docs/findings.md` §1](docs/findings.md).

**Dataset:** [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets)
(CC BY-NC 4.0). Decisão e justificativa em [`docs/dataset.md`](docs/dataset.md).
