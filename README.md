# Semantic Book Recommender: An Emotion-Aware NLP Pipeline

**Author:** Syed Adnan Ali  
**Technical Focus:** Semantic Search, Representation Learning, Zero-Shot Classification, Sentiment Analysis

---

## Project Overview

This repository features an end-to-end Data Science pipeline that transforms raw bibliographic data into an interactive, semantic-aware recommendation engine. The project demonstrates the integration of multiple Large Language Model (LLM) capabilities to solve traditional data challenges, such as sparse category mapping and nuanced user intent.

---

### Key Data Science Competencies
* **Representation Learning:** Utilized **OpenAI Embeddings** and **Cosine Similarity** to implement semantic retrieval, moving beyond keyword-based search.
* **Zero-Shot Classification:** Employed `facebook/bart-large-mnli` to simplify a high-cardinality dataset (500+ categories) into 4 meaningful genres, ensuring 100% coverage of "long-tail" data.
* **Emotion Analytics:** Integrated a **DistilRoBERTa-based** classifier to enrich the dataset with human-centric emotional dimensions (Ekman’s basic emotions).
* **System Design:** Developed a modular architecture connecting exploratory data analysis (EDA), a vector database (ChromaDB), and a web-based UI (Gradio).

---

## Technical Pipeline & Architecture

The system follows a rigorous four-stage pipeline designed for accuracy and scalability:

1.  **EDA & Probabilistic Cleaning:** Conducted missingness heatmaps and correlation analysis to justify description-length cutoffs without introducing bias into other dimensions, such as ratings or book age.
2.  **Category Engineering:** Simplified the taxonomy using a hybrid approach: high-frequency categories were retained, while low-frequency entries were mapped via **Zero-Shot Inference** using BART-MNLI.
3.  **Vector Store Integration:** Constructed a persistent vector index using **ChromaDB** and **LangChain**, allowing for sub-second similarity search across 5,000+ titles.
4.  **Emotional Re-ranking:** Implemented a filtering layer that uses per-emotion scores (joy, surprise, fear, etc.) to re-sort semantic results based on user-defined "tone".

---

## Performance & Results

* **Semantic Accuracy:** Successfully retrieves conceptually related titles (e.g., querying "survival in the wilderness" returns survivalist nonfiction and adventure fiction).
* **Pipeline Efficiency:** Pre-calculated embeddings and emotion scores are cached in `data/books_with_emotions.csv` to minimize runtime latency and API costs.
* **User Interface:** The **Gradio dashboard** provides a responsive gallery view, displaying cover art, metadata, and truncated summaries.

---

## 📌 TL;DR
- **Input:** Kaggle "7k+ Books" dataset (titles, subtitles, categories, descriptions, ISBN, etc.).  
- **Process:** EDA & cleaning → category simplification → zero-shot mapping → emotion scoring → vector DB → cosine similarity retrieval.  
- **Output:** A **Gradio dashboard** that returns **semantic recommendations** for any user query, filterable by **genre** (4 categories) and **emotion** (Ekman + neutral).

## 🎥 Demo

<p align="center">
  <img src="assets/gradio-dashboard.png" alt="Semantic Book Recommender – Gradio dashboard" width="100%">
</p>

*Example:* Query: “A book about World War One” · Category: **Nonfiction** · Emotion: **Suspenseful**.

---

## 🧭 Project Structure
```
semantic-book-recommender/
├── app/
│   └── gradio-dashboard.py          # End-user app: semantic retrieval + filters + gallery UI
├── notebooks/
│   ├── data-exploration.ipynb       # EDA: missingness, correlations, desc length cutoff, merges, save cleaned data
│   ├── vector-search.ipynb          # Practice: LangChain + OpenAIEmbeddings + similarity_search (scratchpad)
│   ├── text-classification.ipynb    # Category pruning (>50 freq), map into 4 genres, zero-shot classify long-tail
│   └── sentiment-analysis.ipynb     # Emotion scoring via j-hartmann/emotion-english-distilroberta-base
├── data/
│   ├── books_cleaned.csv
│   ├── books_with_categories.csv
│   ├── books_with_emotions.csv
│   └── tagged_description_Updated.txt
├── assets/
│   ├── cover-not-found.jpg          # Fallback image for missing book covers
│   └── gradio-dashboard.png         # README demo screenshot
├── scripts/
│   ├── run_app.sh                   # Helper scripts to launch the app
│   └── run_app.bat
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🏗️ Architecture (Mermaid)
```mermaid
flowchart LR
  A[Kaggle dataset] --> B[EDA & Cleaning<br/>Missingness heatmap, correlation, desc length cutoff]
  B --> C[Category Simplification<br/>Keep freq>50 + 4 super-categories]
  C --> D[Zero-shot Classification<br/>facebook/bart-large-mnli for long-tail]
  D --> E[Emotion Scoring<br/>j-hartmann/emotion-english-distilroberta-base]
  E --> F[Vector DB & Embeddings<br/>OpenAIEmbeddings + LangChain]
  F --> G[Gradio App<br/>Cosine similarity, filters & gallery]
```

---

## ✨ Key Features
- **Semantic Retrieval:** OpenAI embeddings + cosine similarity surface conceptually similar books to a natural-language query.
- **Smart Filtering:** Filter by **Genre** (Fiction, Non-Fiction, Juvenile Fiction, Non-Juvenile Fiction) and **Emotion** (anger, fear, sadness, disgust, joy, surprise, neutral).
- **Robust EDA:** Missingness heatmaps, correlation analysis to justify dropping short/empty descriptions w/out biasing other dimensions.
- **Zero-Shot Mapping:** Collapse ~500 raw categories into 4 genres; unsupervised mapping for low-frequency categories with BART-MNLI.
- **Emotion Enrichment:** DistilRoBERTa-based classifier adds per-emotion scores as columns for nuanced filtering.
- **Simple UI:** Gradio dashboard renders a clean, responsive gallery with title + subtitle + cover (if present) + meta.

---

## 🔧 Setup

### 1) Python env
```bash
# Python 3.10+ recommended
pip install -r requirements.txt
```

### 2) Environment variables
Copy the template and fill your keys:
```bash
cp .env.example .env
```
**Required (for embeddings):**
- `OPENAI_API_KEY` — for `OpenAIEmbeddings` used via `langchain-openai`

**Optional (if pulling models from HF hub programmatically):**
- `HUGGINGFACE_HUB_TOKEN`

**If you download via Kaggle programmatically (kagglehub):**
- `KAGGLE_USERNAME`, `KAGGLE_KEY` (or a kaggle.json in the standard location)

> You can also run the notebooks without internet if you already have the dataset locally.

---

## 📂 Data Files (used by notebooks & app)

The repo includes CSVs produced at each stage so reviewers can run the app immediately:

- `data/books_cleaned.csv` — output of **`notebooks/data-exploration.ipynb`** after EDA, description-length filtering, title–subtitle merge, and ISBN postfix for lookup.
- `data/books_with_categories.csv` — output of **`notebooks/text-classification.ipynb`** after pruning raw categories (>50 frequency) and mapping all titles into **4 simplified genres** via zero-shot **BART-MNLI**.
- `data/books_with_emotions.csv` — output of **`notebooks/sentiment-analysis.ipynb`** after scoring **7 emotions** (anger, fear, sadness, disgust, joy, surprise, neutral) with **DistilRoBERTa**; includes per-emotion columns and/or a dominant label. **This is the default file consumed by the app.**
- `data/tagged_description_Updated.txt` — line-separated descriptions (used for quick embedding/vectorization experiments in `vector-search.ipynb`).

**Assets**

- `assets/cover-not-found.jpg` — fallback image used by the Gradio gallery when a book cover is unavailable.
- `assets/gradio-dashboard.png` — screenshot used in this README.

---

## 🧪 Reproduce the Pipeline (Notebook flow)

1. **`notebooks/data-exploration.ipynb`**
   - Load dataset (via `kagglehub` or local file)
   - Visualize **missingness heatmap** and **correlation** of key attributes (ratings, num_pages, book_age)
   - Inspect description **length distribution**; decide a cutoff for “meaningful” descriptions
   - Merge `title` + `subtitle` into a canonical display field
   - Append `ISBN` into description text to simplify lookup from search results
   - Save **cleaned** dataframe to `data/books_cleaned.csv`

2. **`notebooks/text-classification.ipynb`**
   - Keep high-frequency categories (threshold > 50)
   - Map into **4 simplified genres**: *fiction*, *non-fiction*, *juvenile fiction*, *non-juvenile fiction*
   - Use **zero-shot classification** with `facebook/bart-large-mnli` to map low-frequency/unknown categories
   - Save **genre-augmented** dataframe to `data/books_with_categories.csv`

3. **`notebooks/sentiment-analysis.ipynb`**
   - Run `j-hartmann/emotion-english-distilroberta-base` to score **7 emotions** (anger, fear, sadness, disgust, joy, surprise, neutral)
   - Keep **max emotion** per description (and/or retain all scores as columns)
   - Save **emotion-enriched** dataframe to `data/books_with_emotions.csv`

4. **`notebooks/vector-search.ipynb`** (optional dev scratchpad)
   - Create embeddings with **OpenAIEmbeddings**
   - Build a vector store (e.g., **Chroma** via LangChain)
   - `similarity_search` experiments and evaluation notes

---

## ▶️ Run the App

By default the app expects `data/books_with_emotions.csv`. Adjust the path in **`app/gradio-dashboard.py`** if you change filenames.

```bash
python app/gradio-dashboard.py
```

Example query ideas:
- “cozy mystery with a hopeful tone” — Genre: *Fiction*, Emotion: *joy*
- “survival story with dread and suspense” — Genre: *Non-Fiction* or *Fiction*, Emotion: *fear*

---

## 🧠 Design Decisions & Justifications
- **Dropping short/empty descriptions:** Backed by **correlation analysis** showing negligible bias introduced w.r.t. ratings, page count, and age.  
- **Genre simplification:** Reduces cognitive load in the UI and avoids sparse/imbalanced categories; **zero-shot** mapping preserves coverage.  
- **Emotion layer:** Adds a *human-centric* axis to re-ranking and filtering (e.g., “joyful coming-of-age mystery”).  
- **Cosine similarity over embeddings:** Standard, interpretable metric for semantic proximity in embedding space.  
- **Gradio front-end:** Lightweight, fast to iterate, easy to demo during interviews or SOP portfolios.

---

## 🧩 Extensibility
- Swap **OpenAIEmbeddings** for local embeddings (e.g., `sentence-transformers`) to run entirely offline.
- Add **re-ranking** with cross-encoder or LLM re-score for improved precision on the top-k.
- Support **faceted search** (authors, year, rating) and add **diversity/novelty** heuristics.
- Cache embeddings & model outputs to cut costs and latency.

---

## Academic Reflection
Building this project provided a rigorous opportunity to bridge the transition from traditional descriptive statistics to semantic enrichment. By leveraging correlation analysis to justify data filtering and employing zero-shot models for data cleaning, I adopted a 'Data-First' approach to AI development. The final system strikes a balance between technical complexity and user-centric design, offering a comprehensive view of the data science lifecycle from initial ingestion to final deployment.

---

## 🗺️ Tech Stack
**Python**, **Pandas**, **NumPy**, **Matplotlib/Seaborn**, **LangChain**, **OpenAIEmbeddings**, **Chroma** (or similar vector store), **Hugging Face Transformers**, **PyTorch**, **Gradio**, **kagglehub**, **python-dotenv**.

---

*This project was developed as an educational deep-dive. For admissions review only.*
