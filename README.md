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
| POST | `/chat` | `{"question": "..."}` → resposta baseada no manual técnico, com fontes (página, seção, score, trecho); recusa se o manual não cobre a pergunta |

### `/chat` (RAG sobre o manual)

O manual é a documentação do OPNsense (BSD 2-Clause), já versionado em `data/manuals/`. Na primeira
chamada o índice FAISS é montado com embeddings locais (o modelo de ~70 MB é baixado uma vez) e fica
em cache em `data/index/`.

```bash
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"question": "How do I reset the root password?"}'
```

Por padrão o provedor é `fake`: monta a resposta com o trecho recuperado e a página, sem chamar
nenhuma API. Para usar um LLM de verdade, copie `.env.example` para `.env` (nunca versionado) e preencha:

```bash
LLM_PROVIDER=gemini          # ou openai
GOOGLE_API_KEY=...           # para gemini (modelo padrão gemini-2.5-flash, troque com GEMINI_MODEL)
# OPENAI_API_KEY=...         # para openai (modelo padrão gpt-4.1-mini, troque com OPENAI_MODEL)
```

E suba a API lendo o arquivo: `uv run --env-file .env uvicorn support_copilot.api.main:app --port 8000`.

Ajustes opcionais: `RAG_TOP_K` (4), `RAG_MIN_SCORE` (0,67), `RAG_CHUNK_SIZE` (1500),
`RAG_CHUNK_OVERLAP` (225), `LLM_MAX_TOKENS` (400), `LLM_TIMEOUT_S` (30).
Avaliação da recuperação: `uv run python scripts/eval_retrieval.py` → `reports/retrieval_eval.md`.

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
