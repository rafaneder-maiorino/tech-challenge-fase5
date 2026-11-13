# Dataset audit — Tobi-Bueck/customer-support-tickets

License: CC BY-NC 4.0 (non-commercial; OK for academic use, cite the source).
Language filter: `en` · min class size: 300 · seed: 42

## `aa_dataset-tickets-multi-lang-5-2-50-version.csv`

- SHA-256: `f187c090e59581c2bbf3aa1377c8db4dd647464ecf2ae51bf8966e42e0ed6bc0`
- Rows: 28,587
- Columns: subject, body, answer, type, queue, priority, language, version, tag_1, tag_2, tag_3, tag_4, tag_5, tag_6, tag_7, tag_8
- Null %: subject=13.4, tag_3=0.5, tag_4=10.7, tag_5=49.1, tag_6=79.5, tag_7=92.9, tag_8=98.0
- Languages: en=16,338, de=12,249
- Exact duplicate texts (subject+body): 0 (0.0%)
- Usable rows (en, non-empty, deduplicated): 16,338
- Text length (chars): p50=418, p95=649, max=1191, >1000=0.1%

### Queues (10 distinct, en)

| queue | n | % | technical? |
|---|---|---|---|
| Technical Support | 4,737 | 29.0% | yes |
| Product Support | 3,073 | 18.8% | yes |
| Customer Service | 2,410 | 14.8% |  |
| IT Support | 1,942 | 11.9% | yes |
| Billing and Payments | 1,595 | 9.8% |  |
| Returns and Exchanges | 820 | 5.0% |  |
| Service Outages and Maintenance | 664 | 4.1% | yes |
| Sales and Pre-Sales | 513 | 3.1% |  |
| Human Resources | 348 | 2.1% |  |
| General Inquiry | 236 | 1.4% |  |

- Classes after consolidation (min 300): 10 → Billing and Payments, Customer Service, Human Resources, IT Support, Other, Product Support, Returns and Exchanges, Sales and Pre-Sales, Service Outages and Maintenance, Technical Support
- Share routed to RAG (technical): 63.8%
- type: Incident=6,571, Request=4,665, Problem=3,397, Change=1,705

### Signal (stratified 5-fold, macro F1)

| model | macro F1 |
|---|---|
| majority class | 0.045 ± 0.000 |
| TF-IDF(1,2) + LR, subject+body | **0.545 ± 0.016** |
| TF-IDF + LR, subject only | 0.266 ± 0.005 |
| TF-IDF + LR, technical vs not (binary) | 0.762 ± 0.009 |

### Leakage probe (columns NOT used as features)

| column | macro F1 predicting queue |
|---|---|
| answer | 0.325 |
| tag_1 | 0.221 |
| tag_2 | 0.184 |
| priority | 0.111 |

**Signal verdict:** PASS (gap over majority = +0.501)

## `dataset-tickets-multi-lang-4-20k.csv`

- SHA-256: `9be3bf810584fe01e8e83383e83dfd33f4c3910938ecad03ef151da79d8f0635`
- Rows: 20,000
- Columns: subject, body, answer, type, queue, priority, language, tag_1, tag_2, tag_3, tag_4, tag_5, tag_6, tag_7, tag_8
- Null %: subject=7.3, tag_2=0.2, tag_3=0.5, tag_4=7.7, tag_5=34.5, tag_6=63.2, tag_7=80.4, tag_8=90.5
- Languages: en=11,923, de=8,077
- Exact duplicate texts (subject+body): 0 (0.0%)
- Usable rows (en, non-empty, deduplicated): 11,923
- Text length (chars): p50=389, p95=815, max=1827, >1000=2.0%

### Queues (10 distinct, en)

| queue | n | % | technical? |
|---|---|---|---|
| Technical Support | 3,412 | 28.6% | yes |
| Product Support | 2,232 | 18.7% | yes |
| Customer Service | 1,859 | 15.6% |  |
| IT Support | 1,391 | 11.7% | yes |
| Billing and Payments | 1,302 | 10.9% |  |
| Returns and Exchanges | 582 | 4.9% |  |
| Service Outages and Maintenance | 442 | 3.7% | yes |
| Sales and Pre-Sales | 330 | 2.8% |  |
| Human Resources | 205 | 1.7% |  |
| General Inquiry | 168 | 1.4% |  |

- Classes after consolidation (min 300): 9 → Billing and Payments, Customer Service, IT Support, Other, Product Support, Returns and Exchanges, Sales and Pre-Sales, Service Outages and Maintenance, Technical Support
- Share routed to RAG (technical): 62.7%
- type: Incident=4,642, Request=3,498, Problem=2,498, Change=1,285

### Signal (stratified 5-fold, macro F1)

| model | macro F1 |
|---|---|
| majority class | 0.049 ± 0.000 |
| TF-IDF(1,2) + LR, subject+body | **0.444 ± 0.009** |
| TF-IDF + LR, subject only | 0.276 ± 0.014 |
| TF-IDF + LR, technical vs not (binary) | 0.730 ± 0.007 |

### Leakage probe (columns NOT used as features)

| column | macro F1 predicting queue |
|---|---|
| answer | 0.299 |
| tag_1 | 0.249 |
| tag_2 | 0.184 |
| priority | 0.119 |

**Signal verdict:** PASS (gap over majority = +0.394)

