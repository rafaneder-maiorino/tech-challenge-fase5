# Dataset — decisão e justificativa

**Escolhido:** `Tobi-Bueck/customer-support-tickets` (Hugging Face), arquivo
`aa_dataset-tickets-multi-lang-5-2-50-version.csv`, SHA-256
`f187c090e59581c2bbf3aa1377c8db4dd647464ecf2ae51bf8966e42e0ed6bc0`.

**Licença:** CC BY-NC 4.0, que permite uso acadêmico não comercial com atribuição. O repositório
não redistribui o CSV: `make download` baixa o arquivo da origem e confere o hash.

Todos os números abaixo saem de `reports/dataset_audit.md`, gerado por
`scripts/dataset_audit.py` (validação cruzada estratificada em 5 folds, F1 macro, seed 42).

## Critérios de aceite (plano, seção 4.1)

| critério | resultado | status |
|---|---|---|
| Licença compatível com uso acadêmico | CC BY-NC 4.0 | ✅ |
| Sinal real no texto | F1 macro 0,545 ± 0,016 (TF-IDF + LR) contra 0,045 da classe majoritária | ✅ |
| Sem vazamento de rótulo nas features | só `subject + body` entram no modelo (ver abaixo) | ✅ |
| Classe "técnica" bem definida para o roteamento | 4 filas, 63,8% dos chamados; binário técnico × não técnico com F1 macro 0,762 | ✅ |
| Compatível com o contrato da API (≤ 1000 caracteres) | p95 = 649, máx. = 1191; só 0,1% passa de 1000 | ✅ |

## Por que este arquivo e não o outro

| | `5-2-50-version` (escolhido) | `4-20k` |
|---|---|---|
| Linhas em inglês, limpas | **16.338** | 11.923 |
| F1 macro TF-IDF + LR | **0,545** | 0,444 |
| > 1000 caracteres | **0,1%** | 2,0% |

O arquivo escolhido tem mais dados e mais sinal, e quase não perde linhas para o limite da API.

## Por que não o Kaggle "Customer Support Ticket Dataset" (suraj520)

As descrições são templates com placeholder literal ("I'm having an issue with the
{product_purchased}. Please assist."), e o rótulo contradiz o assunto: há `Ticket Type: Technical
issue` com `Ticket Subject: Payment issue`. Não existe sinal para o modelo aprender.

## Features: só `subject + body`

`answer` (resposta do atendente) e `tag_*` **não existem no momento da inferência**: a resposta é
escrita depois da triagem. Esse é o motivo da exclusão. A sondagem de vazamento confirma que
nenhuma delas é um atalho escondido: `answer` prevê a fila com F1 macro 0,325, `tag_1` com 0,221
e `priority` com 0,111. Todas ficam abaixo do próprio texto (0,545).

`subject` é nulo em 13,4% das linhas e fica vazio nesses casos. Sozinho, ele dá só 0,266: o
`body` carrega a maior parte do sinal.

## Espaço de rótulos: as 10 filas originais

O dataset completo tem 52 valores distintos de fila, mas o recorte em inglês tem exatamente
**10**. Elas são mantidas como estão, sem classe "Other": consolidar com mínimo de 300 só
renomearia `General Inquiry` (236), o que não agrega nada. O desbalanceamento (29,0% a 1,4%) é
tratado no treino (`class_weight`) e a métrica principal é o **F1 macro**.

| fila | % (en) | técnica |
|---|---|---|
| Technical Support | 29,0% | sim |
| Product Support | 18,8% | sim |
| Customer Service | 14,8% | |
| IT Support | 11,9% | sim |
| Billing and Payments | 9,8% | |
| Returns and Exchanges | 5,0% | |
| Service Outages and Maintenance | 4,1% | sim |
| Sales and Pre-Sales | 3,1% | |
| Human Resources | 2,1% | |
| General Inquiry | 1,4% | |

**Hipótese para o D2:** o F1 de 10 classes (0,545) fica bem abaixo do binário técnico × não
técnico (0,762). Isso sugere que a confusão se concentra **dentro** do grupo técnico (Technical ×
Product × IT Support), que é semanticamente sobreposto. Se a matriz de confusão confirmar, o
roteamento para o RAG é robusto mesmo quando a fila exata erra, e esse é o número que importa
para o produto.

## Roteamento para o RAG

As filas técnicas são `Technical Support`, `Product Support`, `IT Support` e
`Service Outages and Maintenance` (63,8%): o grupo de "algo precisa ser consertado". O manual em
PDF cobre um domínio só (rede/roteador). Perguntas técnicas fora dele recebem "não encontrei no
manual" pelo guardrail de contexto, e o LLM nem é chamado.

## Idiomas e drift

O recorte em alemão (12.249 linhas) **não entra no treino**. Ele é limpo com as mesmas regras e
salvo como `drift_pool_de.parquet`, para servir no D8 como cenário de drift **realista**
(mudança de idioma e vocabulário), em vez de ruído artificial.

## Limitações

- **Provavelmente sintético.** A página não diz como os dados foram gerados, e há placeholders como
  `<name>` nas respostas. As métricas medem o que o modelo faz **neste** dataset, não em tickets
  reais.
- **Licença não comercial.** Um uso em produção exigiria outro dataset.
- **Só inglês no treino.** O desempenho em outro idioma é exatamente o que o D8 vai medir como drift.
