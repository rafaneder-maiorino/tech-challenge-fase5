# Diário do projeto — Datathon Fase 5

Registro diário do que foi feito, das decisões tomadas e dos números obtidos. A entrada mais
recente fica no topo. O plano completo, com a matriz de requisitos e o checklist, fica fora do
repositório.

**Meta interna:** 27/11/2026 · **Prazo oficial:** 29/01/2027

---

## 13/11/2026 (sex, tarde) — D2 adiantado: modelo + MLflow

**Feito**
- Servidor MLflow local na porta 5001 (backend SQLite, artefatos em `mlartifacts/`).
- `support_copilot.model.train`: 4 candidatos com validação cruzada pareada (mesmos folds), uma execução no MLflow por candidato (parâmetros, métricas, matriz de confusão, classification report, modelo com signature). Campeão registrado como `ticket-classifier` com alias `@champion` e exportado para `models/ticket_classifier.joblib` + `.json`.
- `reports/model_selection.md`, `reports/confusion_matrix.png`, `reports/tuning_linear.md`, `docs/findings.md` §1–§3 e captura do MLflow em `docs/img/`.
- `prepare` refeito com split por família de paráfrases, e schema Pandera validando os 3 arquivos processados.

**Decisões**
- **Split por família de paráfrases** (componentes conexos com cosseno ≥ 0,7) no split e na validação cruzada. O split aleatório deixava 57,4% do teste com uma paráfrase no treino.
- **Campeão = regressão logística (C=4)**, escolhida pela validação cruzada por família, sem usar o teste.
- **Roteamento pela fila mais provável.** Rotear pela soma das probabilidades técnicas não melhora (0,694 contra 0,693).
- **XGBoost em configuração leve** (150 árvores, 20k features): continua como candidato, mas perde dos lineares e custa ~7× o tempo.
- MLflow 3.18 serializa com `skops`. Os tipos do XGBoost e da calibração do sklearn entram numa lista explícita de tipos confiáveis, por candidato.

**Números (teste por família)**
| métrica | valor |
|---|---|
| F1 macro na validação cruzada / no teste (campeão) | **0,332 ± 0,008 / 0,371** |
| Acurácia no teste | 0,383 |
| F1 do roteamento técnico × não técnico | 0,693 |
| Margem sobre o 2º colocado (XGBoost), pareada por fold | +0,018 ± 0,014, vence em 5 de 5 |
| Classe majoritária (F1 macro) | 0,045 |
| Mesmo modelo com split aleatório (com vazamento) | 0,610 / 0,658 |

**Validação:** ruff limpo · 17 testes passando.

**Hipótese refutada:** os erros não ficam dentro do grupo técnico: 45,8% atravessam a fronteira do roteamento (findings §2).

**Pendente:** colar no Makefile os alvos `mlflow-server` e `train` (bloqueio de escrita remota).

---

## 13/11/2026 (sex) — D1: repositório e dataset

**Feito**
- Repositório `tech-challenge-fase5` criado (uv, Python 3.11, ruff, pytest, `.gitattributes` com LF).
- `scripts/dataset_audit.py`: auditoria de aceite do dataset → `reports/dataset_audit.md`.
- `support_copilot.data.source`: download com URL e SHA-256 fixados, idempotente.
- `support_copilot.data.prepare`: limpeza, split estratificado e pool de drift → `reports/prepare_summary.md`.
- `docs/dataset.md` com a decisão e a justificativa. Makefile com `download`, `audit`, `prepare`, `lint` e `test`.
- PR #1 mergeado em `main` (`eec0e55`).

**Decisões**
- **Dataset:** `Tobi-Bueck/customer-support-tickets`, arquivo `5-2-50-version` (CC BY-NC 4.0).
- **Rejeitado:** Kaggle suraj520. Descrições são templates com placeholder literal e o rótulo contradiz o assunto.
- **Arquivo v5 em vez do 4-20k:** mais linhas em inglês (16.338 × 11.923), mais sinal (F1 0,545 × 0,444) e menos textos acima de 1000 caracteres (0,1% × 2,0%).
- **Features = `subject + body`.** `answer` e `tag_*` ficam fora porque não existem no momento da triagem.
- **Rótulos:** as 10 filas originais do recorte em inglês, sem classe "Other".
- **Roteamento para o RAG:** Technical Support, Product Support, IT Support e Service Outages and Maintenance.
- **Textos com mais de 1000 caracteres são descartados,** o mesmo limite que a API vai aplicar (9 em inglês, 36 em alemão).
- **O recorte em alemão não entra no treino:** fica guardado como pool para o cenário de drift do D8.
- **CI básico** passou do D1 para o D9, junto com o build do Docker.

**Números**
| métrica | valor |
|---|---|
| F1 macro, TF-IDF(1,2) + LR, 10 filas (CV 5-fold) | **0,545 ± 0,016** |
| F1 macro, classe majoritária | 0,045 |
| F1 macro, técnico × não técnico | 0,762 |
| F1 macro, só `subject` | 0,266 |
| Sondagem de vazamento: `answer` / `tag_1` / `priority` | 0,325 / 0,221 / 0,111 |
| Treino / teste / pool alemão | 13.063 / 3.266 / 12.213 |
| Parcela técnica (treino e teste) | 63,8% |
| Textos repetidos entre treino e teste | 0 |

**Validação:** `make lint` limpo · `make test` com 5 testes passando.

**Hipótese em aberto (D2):** a distância entre o F1 de 10 classes (0,545) e o binário (0,762)
sugere que a confusão se concentra entre as filas técnicas. Confirmar com a matriz de confusão.

**Pendente:** apagar a branch remota `feat/d1-dataset-audit`.

---

## 12/11/2026 (qui) — D0: análise e plano

**Feito**
- Material da Fase 5 baixado da plataforma e organizado no Drive por capítulo: 19 aulas liberadas, mais o enunciado do Datathon e o capítulo de projeto.
- Enunciado analisado: 27 requisitos mapeados para dias e evidências, cobrindo os 5 entregáveis e os 5 critérios da rubrica.
- Plano de 14 dias até 27/11, com prioridades P0/P1/P2.

**Decisões**
- Meta interna de entrega em 27/11, com o prazo oficial de 29/01/2027 como margem de segurança.
- RAG + Guardrails (30% da nota) recebe 4 blocos de trabalho e entra no vídeo.
- LLM com provider `fake` determinístico, para o avaliador conseguir rodar sem chave de API.
- Embeddings locais, para a busca funcionar offline e para minimizar os dados enviados ao LLM.

**Observação:** os capítulos "Avaliação e Observabilidade em LLMOps" (bloqueado) e "Segurança,
Guardrails e Conformidade" (PDFs "em breve") ainda não estavam liberados.
