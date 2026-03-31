# Dataset Transformation Specification

A clean and unified specification for converting the selected source datasets into a consistent **BEIR-style retrieval format** for experimentation, evaluation, and future extension.

---

## Table of Contents

- [1. Goal](#1-goal)
- [2. Source Datasets](#2-source-datasets)
- [3. Target Standard Format](#3-target-standard-format)
- [4. Source Directory Layout](#4-source-directory-layout)
- [5. Conversion Summary](#5-conversion-summary)
- [6. Dataset-Specific Conversion Rules](#6-dataset-specific-conversion-rules)
  - [6.1 Natural Questions](#61-natural-questions)
  - [6.2 SciFact](#62-scifact)
  - [6.3 DBpedia-Entity](#63-dbpedia-entity)
  - [6.4 CISI](#64-cisi)
  - [6.5 SituatedQA](#65-situatedqa)
- [7. CISI Graph Extension](#7-cisi-graph-extension)
- [8. SituatedQA Split and ID Policy](#8-situatedqa-split-and-id-policy)
- [9. Final Output Structure](#9-final-output-structure)
- [10. Notes](#10-notes)

---

## 1. Goal

The purpose of this transformation is to convert all selected datasets into one **common retrieval format** so they can be accessed, indexed, evaluated, and compared through a single pipeline.

The target format follows a **BEIR-style structure**:

- `corpus.jsonl`
- `queries.jsonl`
- `qrels/dev.tsv`
- `qrels/test.tsv`

For datasets that contain useful structural information beyond standard retrieval text, an **additional auxiliary file** may be preserved. In this project, CISI uses an extra `edges.jsonl` file to retain document cross-references.

---

## 2. Source Datasets

The following datasets are used:

- **Natural Questions**  
  `https://ai.google.com/research/NaturalQuestions`
- **SciFact**  
  `https://github.com/allenai/scifact`
- **DBpedia-Entity**  
  `https://github.com/iai-group/DBpedia-Entity/`
- **CISI**  
  `https://github.com/GianRomani/CISI-project-MLOps`
- **SituatedQA**  
  `https://situatedqa.github.io/`

---

## 3. Target Standard Format

All datasets are normalized into the following schema.

### 3.1 `corpus.jsonl`

Each line stores one retrievable document or chunk.

```json
{"_id":"","title":"","text":"","metadata":{}}
```

### 3.2 `queries.jsonl`

Each line stores one query.

```json
{"_id":"","text":"","metadata":{}}
```

### 3.3 `dev.tsv`

Used for experimentation, tuning, and intermediate evaluation.

```tsv
query_id	corpus_id	score
```

### 3.4 `test.tsv`

Used for final evaluation.

```tsv
query_id	corpus_id	score
```

### 3.5 Relevance convention

Unless otherwise required by the original dataset:

- `score = 1` means relevant
- Only positive relevance links are preserved in the normalized qrels files

---

## 4. Source Directory Layout

```text
├── cisi
│   ├── CISI.ALL
│   ├── CISI.QRY
│   └── CISI.REL
├── dbpedia-entity
│   ├── qrels
│   │   ├── dev.tsv
│   │   └── test.tsv
│   ├── corpus.jsonl
│   └── queries.jsonl
├── nq
│   ├── qrels
│   │   └── test.tsv
│   ├── corpus.jsonl
│   └── queries.jsonl
├── scifact
│   ├── qrels
│   │   ├── test.tsv
│   │   └── train.tsv
│   ├── corpus.jsonl
│   └── queries.jsonl
└── situatedQA/qa_data
    ├── geo.dev.jsonl
    ├── geo.test.jsonl
    ├── geo.train.jsonl
    ├── temp.dev.jsonl
    ├── temp.test.jsonl
    └── temp.train.jsonl
```

---

## 5. Conversion Summary

| Dataset | Already BEIR-style? | Required Changes |
|---|---:|---|
| Natural Questions | Partial | Create `dev.tsv` as a subset of `test.tsv` for experimentation |
| SciFact | Partial | Reassign available qrels so the working dataset has both `dev.tsv` and `test.tsv` |
| DBpedia-Entity | Yes | No structural change needed |
| CISI | No | Convert `CISI.ALL`, `CISI.QRY`, and `CISI.REL` into BEIR-style files; preserve cross-references in `edges.jsonl` |
| SituatedQA | No | Convert Geo and Temp into separate BEIR-style datasets using normalized intermediate files |

---

## 6. Dataset-Specific Conversion Rules

## 6.1 Natural Questions

### Original structure

#### `corpus.jsonl`

```json
{"_id":"","title":"","text":"","metadata":""}
```

#### `queries.jsonl`

```json
{"_id":"","text":"","metadata":""}
```

#### `test.tsv`

```text
{query_id:"",corpus_id:"",score:""}
```

### Conversion rule

Natural Questions already follows the target retrieval format for:

- `corpus.jsonl`
- `queries.jsonl`
- `test.tsv`

The only missing file is:

- `dev.tsv`

### Action

Create `dev.tsv` as a subset of `test.tsv` for experimentation.

### Result

```text
nq/
├── corpus.jsonl
├── queries.jsonl
└── qrels/
        ├── dev.tsv
        └── test.tsv
```

---

## 6.2 SciFact

### Original structure

#### `corpus.jsonl`

```json
{"_id":"","title":"","text":"","metadata":""}
```

#### `queries.jsonl`

```json
{"_id":"","text":"","metadata":""}
```

#### `test.tsv`

```text
{query_id:"",corpus_id:"",score:""}
```

#### `train.tsv`

```text
{query_id:"",corpus_id:"",score:""}
```

### Conversion rule

SciFact already has the required content, but the split names are adjusted for this project’s working convention.

### Action

- Use the original `test.tsv` as `dev.tsv`
- Use the original `train.tsv` as `test.tsv`

### Result

```text
scifact/
├── corpus.jsonl
├── queries.jsonl
└── qrels/
        ├── dev.tsv
        └── test.tsv
```

> This is a project-level split reassignment for experimentation convenience.

---

## 6.3 DBpedia-Entity

### Original structure

#### `corpus.jsonl`

```json
{"_id":"","title":"","text":"","metadata":""}
```

#### `queries.jsonl`

```json
{"_id":"","text":"","metadata":""}
```

#### `test.tsv`

```text
{query_id:"",corpus_id:"",score:""}
```

#### `dev.tsv`

```text
{query_id:"",corpus_id:"",score:""}
```

### Conversion rule

DBpedia-Entity already matches the required format.

### Action

No structural changes are needed.

### Result

```text
dbpedia-entity/
├── corpus.jsonl
├── queries.jsonl
└── qrels/
        ├── dev.tsv
        └── test.tsv
```

---

## 6.4 CISI

CISI is not provided in BEIR format and must be fully transformed.

### Original files

#### `CISI.ALL`

A file of 1,460 documents. Each record contains:

- `.I` = document ID
- `.T` = title
- `.A` = author
- `.W` = abstract / main text
- `.X` = cross-references to other documents

Example conceptual structure:

```text
{.I:"",.T:"",.A:"",.W:"",.X:""}
```

#### `CISI.QRY`

A file containing 112 queries. Each record contains:

- `.I` = query ID
- `.W` = query text

```text
{.I:"",.W:""}
```

#### `CISI.REL`

A relevance mapping between queries and documents.

Example rows:

```text
1     28  0    0.000000
1     35  0    0.000000
1     38  0    0.000000
1     42  0    0.000000
1     43  0    0.000000
```

### Conversion rule

CISI is parsed as a structured classical IR collection.

#### `CISI.ALL` -> `corpus.jsonl`

Field mapping:

| CISI field | Target field |
|---|---|
| `.I` | `_id` |
| `.T` | `title` |
| `.W` | `text` |
| `.A` | `metadata.authors` |
| `.X` | moved to `edges.jsonl` |

### Why `.X` is not kept directly inside `metadata`

The `.X` field contains document-to-document references. These are structurally important, but they are not ideal as inline metadata for retrieval.

To keep the corpus fully compatible with standard BEIR-style retrieval while still preserving structural relationships, the dataset is split into:

- `corpus.jsonl` for retrievable text documents
- `edges.jsonl` for document cross-reference links

### Example `corpus.jsonl`

```json
{
  "_id": "1",
  "title": "18 Editions of the Dewey Decimal Classifications",
  "text": "The present study is a history of the DEWEY Decimal Classification...",
  "metadata": {
    "authors": ["Comaromi, J.P."]
  }
}
```

#### `CISI.QRY` -> `queries.jsonl`

Field mapping:

| CISI field | Target field |
|---|---|
| `.I` | `_id` |
| `.W` | `text` |

Since the query file contains no additional metadata, an empty metadata object is retained for consistency.

### Example `queries.jsonl`

```json
{"_id":"1","text":"information retrieval systems","metadata":{}}
```

#### `CISI.REL` -> `test.tsv`

Only the first two columns are useful:

- column 1 -> `query_id`
- column 2 -> `corpus_id`

The last two columns do not carry useful relevance information for this project and are discarded.

A new third column is added:

- `score = 1`

### Resulting qrels format

```tsv
query_id	corpus_id	score
1	28	1
1	35	1
1	38	1
```

### Additional action

Create `dev.tsv` as a subset of the resulting `test.tsv` for experimentation.

### Result

```text
cisi/
├── corpus.jsonl
├── queries.jsonl
└── qrels/
        ├── dev.tsv
        ├── test.tsv
        └── edges.jsonl
```

---

## 6.5 SituatedQA

SituatedQA is transformed into **two separate BEIR-style datasets**:

- `geo`
- `temp`

This is necessary because the two parts represent different kinds of contextual reasoning:

- **Geo** uses location-sensitive context
- **Temp** uses time-sensitive context

### Original files

#### Geographic files

- `geo.train.jsonl`
- `geo.dev.jsonl`
- `geo.test.jsonl`

Each row has the form:

```json
{"question":"","id":"","edited_question":"","location":"","answer":"","any_answer":""}
```

#### Temporal files

- `temp.train.jsonl`
- `temp.dev.jsonl`
- `temp.test.jsonl`

Each row has the form:

```json
{"question":"","id":"","edited_question":"","date":"","date_type":"","answer":"","any_answer":""}
```

### Normalization strategy

The three splits are first merged into intermediate normalized files:

- `geo.train + geo.dev + geo.test -> geo.jsonl`
- `temp.train + temp.dev + temp.test -> temp.jsonl`

This merge is used only for unified processing.

### Important rule

Original split membership is preserved for every row during normalization.

That means:

- `dev.tsv` is generated only from rows originally belonging to the `dev` split
- `test.tsv` is generated only from rows originally belonging to the `test` split

So the evaluation files are **not** created by modifying one another.

### Output format per dataset

Each of `geo` and `temp` becomes:

```text
<dataset>/
├── corpus.jsonl
├── queries.jsonl
└── qrels/
        ├── dev.tsv
        └── test.tsv
```

### Corpus design choice

Each source row becomes:

- one corpus record
- one query record
- one positive qrel mapping from query to the matching corpus item

This is necessary because the same original `id` may appear multiple times under different contextual versions.

Instead of using the repeated source `id` directly as the BEIR `_id`, a new local unique `_id` is assigned for each generated record.

The original source identifier is preserved in metadata as `source_id`.

---

## 7. CISI Graph Extension

CISI’s cross-references are preserved separately in `edges.jsonl`.

Each record stores one reference relation from a source document to a target document.

### `edges.jsonl` schema

```json
{"source_id":"","target_id":"","relation":"cross_reference","raw":[]}
```

### Fields

| Field | Meaning |
|---|---|
| `source_id` | current document ID |
| `target_id` | referenced document ID |
| `relation` | relation type |
| `raw` | original tuple extracted from `.X` |

### Example

```json
{"source_id":"1","target_id":"92","relation":"cross_reference","raw":[92,1,1]}
{"source_id":"1","target_id":"262","relation":"cross_reference","raw":[262,1,1]}
{"source_id":"1","target_id":"556","relation":"cross_reference","raw":[556,1,1]}
```

### Why this matters

This design keeps the main corpus compatible with standard lexical and dense retrieval pipelines, while also preserving structural data for optional:

- graph-based retrieval
- citation traversal
- HNSW or graph-augmented experiments

---

## 8. SituatedQA Split and ID Policy

## 8.1 Repeated source IDs in Geo

A single source `id` may appear multiple times with different contextual interpretations.

### Example source rows

```json
{"question": "what is the value of the currency", "id": -4036207256798544363, "edited_question": "what is the value of the currency in paraguay", "location": "paraguay", "answer": ["126 PYG to 1 USD"], "any_answer": ["7.84 Hong Kong Dollar to 1 USD", "126 PYG to 1 USD", ".2017 Pound Sterling to 1 USD"]}
{"question": "what is the value of the currency", "id": -4036207256798544363, "edited_question": "what is the value of the currency in england", "location": "England", "answer": [".2017 Pound Sterling to 1 USD"], "any_answer": ["7.84 Hong Kong Dollar to 1 USD", "126 PYG to 1 USD", ".2017 Pound Sterling to 1 USD"]}
{"question": "what is the value of the currency", "id": -4036207256798544363, "edited_question": "what is the value of the currency in hong kong", "location": "Hong Kong", "answer": ["7.84 Hong Kong Dollar to 1 USD"], "any_answer": ["7.84 Hong Kong Dollar to 1 USD", "126 PYG to 1 USD", ".2017 Pound Sterling to 1 USD"]}
```

### Converted `corpus.jsonl`

```json
{"_id":"1","title":"what is the value of the currency","text":"Question: what is the value of the currency in paraguay\nAnswer: 126 PYG to 1 USD","metadata":{"location":"paraguay","source_id":"-4036207256798544363"}}
{"_id":"2","title":"what is the value of the currency","text":"Question: what is the value of the currency in england\nAnswer: .2017 Pound Sterling to 1 USD","metadata":{"location":"England","source_id":"-4036207256798544363"}}
{"_id":"3","title":"what is the value of the currency","text":"Question: what is the value of the currency in hong kong\nAnswer: 7.84 Hong Kong Dollar to 1 USD","metadata":{"location":"Hong Kong","source_id":"-4036207256798544363"}}
```

### Converted `queries.jsonl`

```json
{"_id":"1","text":"what is the value of the currency in paraguay","metadata":{"location":"paraguay","source_id":"-4036207256798544363"}}
{"_id":"2","text":"what is the value of the currency in england","metadata":{"location":"England","source_id":"-4036207256798544363"}}
{"_id":"3","text":"what is the value of the currency in hong kong","metadata":{"location":"Hong Kong","source_id":"-4036207256798544363"}}
```

### Converted `test.tsv`

```tsv
query_id	corpus_id	score
1	1	1
2	2	1
3	3	1
```

---

## 8.2 Repeated source IDs in Temp

A single temporal question may also appear multiple times with different dates.

### Example source rows

```json
{"question": "where will the next summer and winter olympics be held", "id": 2098168902147822379, "edited_question": "where will the next summer and winter olympics be held as of 2021", "date": "2021", "date_type": "sampled_year", "answer": ["Japan and China"], "any_answer": ["Brazil and S. Korea", "Japan and China"]}
{"question": "where will the next summer and winter olympics be held", "id": 2098168902147822379, "edited_question": "where will the next summer and winter olympics be held as of 2018", "date": "2018", "date_type": "sampled_year", "answer": ["Brazil and S. Korea"], "any_answer": ["Brazil and S. Korea", "Japan and China"]}
{"question": "where will the next summer and winter olympics be held", "id": 2098168902147822379, "edited_question": "where will the next summer and winter olympics be held as of 2017", "date": "2017", "date_type": "sampled_year", "answer": ["Brazil and S. Korea"], "any_answer": ["Brazil and S. Korea", "Japan and China"]}
{"question": "where will the next summer and winter olympics be held", "id": 2098168902147822379, "edited_question": "where will the next summer and winter olympics be held as of 2020", "date": "2020", "date_type": "start", "answer": ["Japan and China"], "any_answer": ["Brazil and S. Korea", "Japan and China"]}
{"question": "where will the next summer and winter olympics be held", "id": 2098168902147822379, "edited_question": "where will the next summer and winter olympics be held as of 2016", "date": "2016", "date_type": "start", "answer": ["Brazil and S. Korea"], "any_answer": ["Brazil and S. Korea", "Japan and China"]}
```

### Converted `corpus.jsonl`

```json
{"_id":"1","title":"where will the next summer and winter olympics be held","text":"Question: where will the next summer and winter olympics be held as of 2021\nAnswer: Japan and China","metadata":{"date":"2021","source_id":"2098168902147822379"}}
{"_id":"2","title":"where will the next summer and winter olympics be held","text":"Question: where will the next summer and winter olympics be held as of 2018\nAnswer: Brazil and S. Korea","metadata":{"date":"2018","source_id":"2098168902147822379"}}
{"_id":"3","title":"where will the next summer and winter olympics be held","text":"Question: where will the next summer and winter olympics be held as of 2017\nAnswer: Brazil and S. Korea","metadata":{"date":"2017","source_id":"2098168902147822379"}}
{"_id":"4","title":"where will the next summer and winter olympics be held","text":"Question: where will the next summer and winter olympics be held as of 2020\nAnswer: Japan and China","metadata":{"date":"2020","source_id":"2098168902147822379"}}
{"_id":"5","title":"where will the next summer and winter olympics be held","text":"Question: where will the next summer and winter olympics be held as of 2016\nAnswer: Brazil and S. Korea","metadata":{"date":"2016","source_id":"2098168902147822379"}}
```

### Converted `queries.jsonl`

```json
{"_id":"1","text":"where will the next summer and winter olympics be held as of 2021","metadata":{"date":"2021","source_id":"2098168902147822379"}}
{"_id":"2","text":"where will the next summer and winter olympics be held as of 2018","metadata":{"date":"2018","source_id":"2098168902147822379"}}
{"_id":"3","text":"where will the next summer and winter olympics be held as of 2017","metadata":{"date":"2017","source_id":"2098168902147822379"}}
{"_id":"4","text":"where will the next summer and winter olympics be held as of 2020","metadata":{"date":"2020","source_id":"2098168902147822379"}}
{"_id":"5","text":"where will the next summer and winter olympics be held as of 2016","metadata":{"date":"2016","source_id":"2098168902147822379"}}
```

### Converted `test.tsv`

```tsv
query_id	corpus_id	score
1	1	1
2	2	1
3	3	1
4	4	1
5	5	1
```

### ID policy summary

For both Geo and Temp:

- repeated original source IDs are allowed in the raw data
- generated BEIR records must receive new unique local `_id` values
- the original raw identifier is preserved in `metadata.source_id`

Optional bookkeeping ranges may be used if desired:

- Geo: `1, 2, 3, ...`
- Temp: `1000001, 1000002, 1000003, ...`

This is optional and used only for implementation convenience.

---

## 9. Final Output Structure

A normalized project layout may look like this:

```text
normalized_datasets/
├── nq/
│   ├── corpus.jsonl
│   ├── queries.jsonl
│   └── qrels/
│       ├── dev.tsv
│       └── test.tsv
├── scifact/
│   ├── corpus.jsonl
│   ├── queries.jsonl
│   └── qrels/
│       ├── dev.tsv
│       └── test.tsv
├── dbpedia-entity/
│   ├── corpus.jsonl
│   ├── queries.jsonl
│   └── qrels/
│       ├── dev.tsv
│       └── test.tsv
├── cisi/
│   ├── corpus.jsonl
│   ├── queries.jsonl
│   └── qrels/
│       ├── dev.tsv
│       ├── test.tsv
│       └── edges.jsonl
├── situatedqa-geo/
│   ├── corpus.jsonl
│   ├── queries.jsonl
│   └── qrels/
│       ├── dev.tsv
│       └── test.tsv
└── situatedqa-temp/
    ├── corpus.jsonl
    ├── queries.jsonl
│   └── qrels/
│       ├── dev.tsv
│       └── test.tsv
```

---

## 10. Notes

- The objective is **uniform access**, not forced loss of dataset-specific structure.
- BEIR-style files are used as the main retrieval interface.
- Dataset-specific structure is preserved separately when it adds value:
  - CISI keeps document links in `edges.jsonl`
  - SituatedQA keeps original source context in metadata such as `location`, `date`, and `source_id`
- `dev.tsv` is used as a working experimental split wherever the original dataset does not provide one directly.
- Geo and Temp are treated as **separate normalized datasets**, not as one merged benchmark.

