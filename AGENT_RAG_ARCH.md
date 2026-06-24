# Agentic RAG Architecture

## 1. Почему линейный RAG не подходит

Линейный RAG обычно выглядит так:

```text
question -> retrieve -> generate -> answer
```

Для юридического домена этого недостаточно.

Основные проблемы линейного RAG:

1. **Нет планирования.** Юридический вопрос часто требует разложения на подзадачи: юрисдикция, тип документа, период действия нормы, роль клиента, факты кейса, исключения.
2. **Нет multi-hop reasoning.** Один вопрос может требовать поиска в договоре, внутренней политике, законе и истории кейса одновременно.
3. **Нет проверки релевантности.** Retriever может вернуть похожие, но юридически нерелевантные фрагменты.
4. **Нет проверки актуальности.** Для законов и нормативных актов важно учитывать дату действия нормы и версию документа.
5. **Нет permission-aware reasoning.** Система не должна использовать документ, если пользователь не имеет права доступа.
6. **Нет надежного fallback loop.** Если ответ не найден, линейный RAG часто либо галлюцинирует, либо дает слабый общий ответ.
7. **Нет отдельного hallucination control.** Генератор сам по себе не должен оценивать собственную правдивость.

Agentic RAG решает эти проблемы через граф агентов: каждый агент отвечает за отдельное решение, а состояние запроса передается между узлами графа.

## 3. High-Level Architecture

```text
Client / Frontend
  -> Laravel API
    -> Auth / Tenant / Permissions / Case Context
    -> Request Validation
    -> Python AI Layer call

Python LangGraph AI Layer
  -> Agent State
  -> Router Agent
  -> Permission Guard
  -> Planner Agent
  -> Query Rewriter
  -> Retrieval Orchestrator
       -> Internal DB Retriever
       -> Legal Corpus Retriever
       -> Contract/Policy Retriever
       -> Case Context Retriever
       -> Lexical Search Tool
       -> Vector Search Tool
       -> Reranker Tool
  -> Document Relevance Grader
  -> Legal Source Grader
  -> Hallucination Grader
  -> Generator Agent
  -> Citation Verifier
  -> Risk / Escalation Agent
  -> Final Response Formatter

Storage / Infra
  -> PostgreSQL source of truth
  -> Vector index service or pgvector cluster
  -> OpenSearch / Elasticsearch for BM25
  -> Redis for cache, locks, rate limits, hot queries
  -> Object storage for original documents
  -> Observability: traces, metrics, audit logs
```

## 4. Основные агенты

### 4.1 Router Agent

Router Agent определяет тип запроса и выбирает дальнейший путь в графе.

Примеры классов запросов:

- простой Q&A по базе знаний;
- вопрос по договору;
- вопрос по внутренней политике;
- вопрос по нормативному акту;
- multi-hop legal analysis;
- запрос на подготовку черновика;
- запрос, требующий уточнения;
- запрос вне зоны покрытия;

Router не отвечает пользователю только классифицирует запрос и выбирает маршрут.

### 4.2 Permission Guard

Permission Guard проверяет, какие данные пользователь имеет право использовать.

Это обязательный слой для юридического бизнеса.

Он отвечает за:

- tenant isolation;
- user permissions;
- case-level access;
- client-level access;
- запрет retrieval по недоступным документам;
- проверку, что все источники в финальном ответе доступны пользователю;
- audit trail: кто, когда, по какому делу получил доступ к какому источнику.

Важно: permission filtering должен применяться до retrieval или внутри retrieval tool, а не только после поиска. Иначе есть риск, что недоступный документ повлияет на ответ.

### 4.3 Planner Agent

Planner Agent разбивает сложный вопрос на подзадачи.

Например, вопрос:

```text
Можно ли расторгнуть договор с клиентом без штрафа, если он нарушил SLA?
```

Planner может разложить его так:

- найти договор клиента;
- найти разделы termination / penalties / SLA;
- найти внутреннюю policy по расторжению;
- найти применимый закон или нормативный акт;
- проверить факты кейса;
- определить, хватает ли данных для ответа;

Planner не генерирует финальный ответ. Он формирует retrieval plan.

### 4.4 Query Rewriter

Query Rewriter переписывает пользовательский вопрос в несколько поисковых форм.

Он нужен потому, что вопросы часто сформулированы бытовым языком, а документы написаны юридическим или англоязычным стилем.

Функции:

- перевод между украинским/русским/английским, если документы мультиязычные;
- расширение синонимами;
- выделение юридических терминов;
- выделение дат, сторон, сумм, типов документов;
- генерация нескольких retrieval-запросов для multi-hop поиска.

Пример:

```text
медичний документ / лікарняний
-> sick leave
-> medical certificate
-> paid sick leave
-> absence cannot be treated as paid sick leave
```

### 4.5 Retrieval Orchestrator

Retrieval Orchestrator выполняет поиск по нескольким источникам и инструментам.

Он не является одним retriever. Это координатор поиска.

Инструменты:

- vector search по embeddings;
- BM25 / lexical search через OpenSearch;
- metadata filters;
- date/version filters;
- tenant/case filters;
- exact identifier search;
- document-specific search;
- legal corpus search;
- case context search.

Результаты объединяются и передаются в reranker.

### 4.6 Reranker Agent / Tool

Reranker пересортировывает найденные кандидаты перед передачей в генерацию.

Для production лучше использовать отдельный сервис:

- BGE reranker large;

Reranker должен работать только на ограниченном top-N candidate set, например top 30-100, а не по всему корпусу.

### 4.7 Document Relevance Grader

Document Relevance Grader проверяет, отвечают ли найденные документы на вопрос.

Он оценивает:

- релевантен ли документ вопросу;
- содержит ли документ прямой ответ;
- является ли фрагмент просто похожим по словам;
- относится ли документ к нужной юрисдикции;
- относится ли документ к нужной дате или версии;
- не противоречит ли документ другим найденным источникам.

Возможные решения:

- `relevant`;
- `partially_relevant`;
- `irrelevant`;
- `needs_more_context`;
- `conflicting_sources`.

Если релевантность низкая, граф возвращается к Query Rewriter или Retrieval Orchestrator.

### 4.8 Legal Source Grader

Legal Source Grader оценивает качество и юридическую пригодность источников.

Он отличается от обычного relevance grader.

Он проверяет:

- является ли источник официальным;
- есть ли дата действия;
- не устарела ли версия;
- является ли документ внутренней политикой, законом, договором или заметкой;
- есть ли конфликт между договором и политикой;
- какой источник имеет приоритет.

Например:

- подписанный договор клиента может иметь приоритет над общей policy;
- актуальная редакция закона имеет приоритет над старой;
- внутренний memo не должен восприниматься как нормативный акт.

### 4.9 Generator Agent

Generator Agent формирует ответ только после того, как context прошел проверки.

Требования к генератору:

- отвечать на языке пользователя;
- использовать только подтвержденный контекст;
- явно ссылаться на источники;
- не придумывать нормы, даты, суммы, исключения;
- указывать ограничения ответа;
- не выдавать финальное юридическое заключение в v1;
- если данных недостаточно, честно сказать об этом.

Generator не должен сам решать, что источник релевантен. Это задача grader-агентов.

### 4.10 Hallucination Grader

Hallucination Grader проверяет готовый ответ относительно найденных источников.

Он отвечает на вопрос:

```text
Подтверждается ли каждое существенное утверждение ответа источниками?
```

Проверяются:

- даты;
- суммы;
- сроки;
- обязанности сторон;
- условия договора;
- юридические выводы;
- исключения;
- формулы;
- ссылки на нормы.

Возможные решения:

- `grounded`;
- `partially_grounded`;
- `unsupported_claims`;
- `contradiction`;
- `needs_revision`;
- `escalate_to_human`.

Если найдено unsupported claim, граф возвращает ответ Generator Agent на исправление или отправляет запрос в fallback/human review.

### 4.11 Citation Verifier

Citation Verifier проверяет, что sources в ответе действительно содержат утверждения, на которые ссылается система.

Он нужен потому, что модель может дать правильный текст, но сослаться на неправильный источник.

Проверки:

- каждый source существует;
- пользователь имеет доступ к source;
- source был в retrieved context;
- source действительно подтверждает соответствующую часть ответа;
- нет ссылок на документы, которые не использовались.

### 4.12 Risk  Agent

Risk Agent определяет, можно ли показывать ответ пользователю напрямую.

Критерии high-risk:

- вопрос требует финального юридического заключения;
- есть финансовые или процессуальные последствия;
- источники конфликтуют;
- недостаточно фактов;
- нет актуальной нормы;
- вопрос касается судебной стратегии;
- пользователь просит подготовить юридически значимый документ;
- confidence низкий;
- hallucination grader нашел неподтвержденные утверждения.

### 4.13 Response Formatter (ЕСЛИ ЛОГИ РЕАЛЬНО НУЖНЫ В JSON, ЛУЧШЕ В ЛОКИ)

Response Formatter приводит результат к стабильному structured JSON.

Он не меняет смысл ответа.

Финальный ответ должен содержать:

- answer;
- sources;
- confidence;
- legal_disclaimer;
- fallback_reason;
- escalation_required;
- trace_id;
- latency;
- agent_path;
- verification_status.

## 5. Agent Graph

Целевой граф:

```text
Start
  -> Router
  -> Permission Guard
  -> Planner
  -> Query Rewriter
  -> Retrieval Orchestrator
  -> Reranker
  -> Document Relevance Grader
      -> if irrelevant: Query Rewriter / Retrieval retry
      -> if needs_more_context: Retrieval retry
      -> if relevant: Legal Source Grader
  -> Legal Source Grader
      -> if outdated/conflict: Retrieval retry or Human Review
      -> if valid: Generator
  -> Generator
  -> Hallucination Grader
      -> if unsupported: Generator revision
      -> if contradiction: Human Review / fallback
      -> if grounded: Citation Verifier
  -> Citation Verifier
      -> if invalid citations: Generator revision
      -> if valid: Risk Agent
  -> Risk Agent
      -> if high risk: Human Review / safe fallback
      -> if acceptable: Response Formatter
  -> Final Answer
```

## 6. Циклы возврата и fallback

В Agentic RAG fallback — это не один `if`.

Нужны несколько типов возврата.

### 6.1 Retrieval retry loop

Если Document Relevance Grader говорит, что документы нерелевантны:

```text
Relevance Grader -> Query Rewriter -> Retrieval -> Reranker -> Relevance Grader
```

Ограничения:

- максимум 2-3 retry;
- каждый retry должен менять запрос;
- trace должен показывать причину повторного поиска.

### 6.2 More context loop

Если документы частично релевантны, но не хватает фактов:

```text
Grader -> Planner -> targeted retrieval -> Grader
```

Например, найден договор, но не найдено приложение с SLA.

### 6.3 Generation revision loop

Если Hallucination Grader нашел неподтвержденное утверждение:

```text
Hallucination Grader -> Generator revision -> Hallucination Grader
```

Ограничения:

- максимум 1-2 revision;
- если снова unsupported, ответ не публикуется;
- система возвращает fallback

### 6.4 Clarification loop

Если вопрос невозможно обработать без дополнительного факта:

```text
Risk/Planner -> ask clarification -> wait user -> restart graph with state
```

Пример:

- неизвестна юрисдикция;
- неизвестна дата договора;
- неизвестно, какой клиент или кейс;
- не указан документ.

## 7. Проверка релевантности документов

Document Relevance Grader должен оценивать не только similarity score.

Similarity говорит:

```text
Этот документ похож на вопрос.
```

Но legal relevance говорит:

```text
Этот документ действительно отвечает на вопрос в нужном контексте.
```

Критерии:

- совпадает topic;
- совпадает legal domain;
- совпадает юрисдикция;
- совпадает tenant/case;
- совпадает дата или версия;
- есть direct answer;
- источник не является просто общим disclaimer;
- source не конфликтует с более приоритетным source.

Результат grader должен сохраняться в agent state и trace.

## 8. Проверка на галлюцинации

Hallucination Grader должен работать после генерации, но до отдачи пользователю.

Он сравнивает ответ с verified context.

Проверка должна быть claim-level:

1. Разбить ответ на утверждения.
2. Для каждого утверждения найти supporting source.
3. Проверить, что source действительно подтверждает утверждение.
4. Проверить, что нет противоречащего source.
5. Пометить unsupported claims.

Примеры unsupported claims:

- модель назвала точную дату, которой нет в источнике;
- модель добавила исключение;
- модель рассчитала сумму без формулы;
- модель сослалась на закон, которого нет в context;
- модель сделала финальный legal conclusion, хотя источники дают только общую информацию.

Если unsupported claims есть, ответ должен быть исправлен или заблокирован.

## 9. Управление состоянием агента

Для управления состоянием лучше использовать LangGraph.

Причина: LangChain удобен для chains/tools, но LangGraph лучше подходит для stateful agent workflows с циклами, условиями и checkpointing.

Agent state должен хранить:

- request_id;
- trace_id;
- user_id;
- tenant_id;
- case_id;
- permissions;
- original_question;
- normalized_question;
- route;
- plan;
- rewritten_queries;
- retrieval_attempts;
- retrieved_documents;
- reranked_documents;
- relevance_grades;
- legal_source_grades;
- generated_answer;
- hallucination_grade;
- citation_grade;
- risk_grade;
- fallback_reason;
- escalation_required;
- final_response.

Нужны checkpoint-и:

- после Router;
- после Retrieval;
- после Grader;
- после Generator;
- перед Human Review;
- перед Final Response.

Это позволяет:

- дебажить ответы;
- повторять pipeline;
- продолжать workflow после уточнения пользователя;
- восстанавливать выполнение после ошибки;
- строить audit trail.

## 10. Интеграция Laravel API + Python LangGraph AI Layer

### Laravel API

Laravel остается главным backend/API слоем.

Он отвечает за:

- authentication;
- authorization;
- tenants;
- users;
- clients;
- cases;
- billing;
- request limits;
- business audit;
- UI/API contracts;
- сохранение финального ответа и trace_id.

Laravel не должен напрямую вызывать LLM для этого workflow. Иначе grounding, retrieval, hallucination checks и trace будут раздроблены.

### Python AI Layer

Python AI Layer отвечает за:

- LangGraph agent orchestration;
- retrieval tools;
- embeddings;
- reranking;
- graders;
- generation;
- hallucination checks;
- citation verification;
- risk classification;
- structured AI response.

### Контракт между Laravel и Python

Laravel передает:

- question;
- user_id;
- tenant_id;
- case_id;
- locale;
- permissions scope;
- request_id;
- optional selected document ids.

Python возвращает:

- answer;
- sources;
- confidence;
- legal_disclaimer;
- fallback_reason;
- escalation_required;
- human_review_reason;
- trace_id;
- agent_path;
- verification_status.

## 11. Инструменты агентов

Минимальный набор tools:

- `internal_document_search`;
- `legal_corpus_search`;
- `contract_search`;
- `case_context_search`;
- `metadata_filter`;
- `permission_check`;
- `rerank`;
- `citation_lookup`;
- `version_check`;
- `conflict_check`;
- `trace_write`.

Каждый tool должен быть deterministic там, где это возможно. LLM не должен сам решать permission, access control или audit.

## 12. Confidence и risk

Confidence не должен быть просто ответом LLM.

Он должен считаться pipeline-side.

Факторы:

- качество retrieval;
- оценка Document Relevance Grader;
- оценка Legal Source Grader;
- hallucination grade;
- citation grade;
- наличие конфликтов;
- количество retry;
- необходимость fallback;
- risk level.

Пример:

- `high`: источники релевантны, актуальны, не конфликтуют, ответ grounded, citations valid.
- `medium`: источники релевантны частично или есть ограничения.
- `low`: данных недостаточно, есть fallback, есть unsupported claims.

## 13. Observability и audit

trace — это не только debugging, но и полноценный артефакт.

Нужно логировать:

- кто задал вопрос;
- tenant/case context;
- какие tools вызывались;
- какие documents были найдены;
- какие documents были отброшены и почему;
- какие graders сработали;
- какие claims были проверены;
- где был fallback;

Production traces должны уходить не только в локальные JSONL, а в:

- OpenTelemetry;
- Loki / ELK;
- metrics dashboard.

