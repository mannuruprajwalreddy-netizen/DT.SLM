---
license: cc0-1.0
language: en
task_categories: [question-answering, text-generation]
tags: [netflix, instruction-tuning]
---

# netflix-qa-dataset

Instruction/QA pairs generated from the Kaggle "Netflix Movies and TV Shows" metadata (`netflix_titles.csv`).

## Structure
JSONL with fields: `system`, `question`, `answer`, `kind` (question type).
Files: `train.jsonl` (90% of titles), `val.jsonl` (10% of titles, held out by title), `stats.json`.

## Construction
Rows missing title/description dropped; titles de-duplicated; ~9 templated Q&A pairs per title; genre recommendation prompts; out-of-scope refusal examples.

## Limitations
Templated (synthetic) questions; factual content only as accurate as the source CSV.

## Source
Kaggle: Netflix Movies and TV Shows (shivamb).
