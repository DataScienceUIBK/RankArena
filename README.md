# 🔍 RankArena

A unified platform for evaluating **retrieval**, **reranking**, and **RAG** with **human** and **LLM-as-a-judge** feedback.

> **Paper:** *RankArena: A Unified Platform for Evaluating Retrieval, Reranking and RAG with Human and LLM Feedback* — CIKM ’25, Seoul, Nov 10–14, 2025.

- **Live demo:** `https://rankarena.ngrok.io/`
- **Demo video:** `https://youtu.be/jIYAP4PaSSI`
- **Contact:** abdelrahman.abdallah@uibk.ac.at



## 🎉News
- **[2025-08-01]** Our paper accepted in 🎉 CIKM 2025 🎉.



## ✨ Features

- **Retriever + Reranker Arena** (offline BM25/DPR/BGE/ColBERT/Contriever on Wiki/MSMARCO; optional online search)
- **RAG Arena**: dual reranking → dual LLM answers (Together AI) → side-by-side comparison
- **LLM Judge**: automatic reranking comparisons saved alongside user votes
- **Leaderboard & Analytics**: ELO-style ranking + plots (Plotly)
- **Annotation Tab**: manual relevance ranking, movement metrics & persisted logs

---

## 🔐 Secrets & Environment

We no longer hard-code API keys. Configure them via environment variables (optionally through a `.env` file).

Create a `.env` file in the repo root (do **not** commit it):

```
SERPER_API_KEY=your-serper-key-here
TOGETHER_API_KEY=your-together-key-here
```

> **What uses these keys?**
>
> - `SERPER_API_KEY` — required **only** for the **Online Retriever** (web search via Serper). Offline retrievers do not need it.
> - `TOGETHER_API_KEY` — required for **RAG generation** (answers from Together’s Llama 3.3 70B Instruct). The UI can still retrieve/rerank without it; only answer generation needs it.

Also add to `.gitignore`:
```
.env
```

Both keys are loaded at runtime using `os.getenv(...)` (and `python-dotenv` if installed). The app shows a friendly error if a required key is missing when you choose a feature that needs it.

---

## 🧰 Requirements

- Python 3.9+
- Recommended: create and activate a virtualenv
- Install deps:
  ```bash
  pip install -r requirements.txt
  ```
  If you don’t have a `requirements.txt`, ensure at least:
  ```bash
  pip install gradio python-dotenv together plotly pandas numpy scipy scikit-learn
  ```
  Plus the project’s own packages/modules (e.g., `rankify`).

---

## 🚀 Quick Start

1) **Set keys** (optional for fully offline usage):
   - Put them in `.env` (see above) or export as env vars:
     ```bash
     export SERPER_API_KEY=...
     export TOGETHER_API_KEY=...
     ```

2) **Run the app**:
   ```bash
   python app.py
   ```
   The app serves on **http://0.0.0.0:7860** by default.

3) **Use it**:
   - **Offline Retriever**: choose method + corpus, no keys needed.
   - **Online Retriever**: requires `SERPER_API_KEY`.
   - **RAG Arena**: requires `TOGETHER_API_KEY` for answer generation.

---

## 🧭 Tabs Overview

- **💬 RAG Arena**: Retrieve → Dual Rerank → Generate answers with Together AI → Compare & Vote  
- **🛠️ Retriever + Reranker**: Retrieve, then run two rerankers; compare rankings & vote  
- **⚡ Direct Reranker / Anonymous Reranker**: Quick reranking flows (no user metadata)  
- **📚 BEIR**: Benchmark-oriented workflows  
- **✏️ Annotation**: Manually rank retrieved contexts, save movement metrics  
- **📊 Leaderboard**: Combined ELO + BEIR/DL stats + rich analytics  
- **ℹ️ Information**: Help texts / about

---

## ⚙️ Configuration Notes

- **Online Retriever**: guarded by a key check. If `SERPER_API_KEY` is missing, you’ll get a clear UI error instead of a crash.
- **Together AI**: the RAG step calls:
  ```
  meta-llama/Llama-3.3-70B-Instruct-Turbo-Free
  ```
  via the `together` SDK. If `TOGETHER_API_KEY` is unset, RAG answer generation will show an informative error.

- **Optional per-session key input**: You can (optionally) re-enable the Together-key textbox in the RAG tab and pass it through; env var fallback still works.


---

## 🧩 Troubleshooting

- **“Missing SERPER_API_KEY”** when using Online Retriever  
  Set `SERPER_API_KEY` in your environment or `.env`. Use Offline Retriever if you don’t need web search.

- **“Please provide a valid Together AI API key” / “Together AI library not installed”** when running RAG  
  Set `TOGETHER_API_KEY` and install the SDK:
  ```bash
  pip install together
  ```

- **Plotly/Sklearn not found**  
  Some analytics are optional; install:
  ```bash
  pip install plotly scikit-learn
  ```

- **Port or host change**  
  Update `demo.launch(server_name="0.0.0.0", server_port=7860, share=False)` in `app.py`.

---

## 🗂️ Data & Logs

- User interactions and evaluations are saved under `user_data/<session_id>/...`:
  - `votes/`, `votellm/`, `annotate/`, etc.
- Leaderboard reads from `leaderboard/result.csv` (if present) and augments with arena stats.

---


## :star2: Citation

Please kindly cite our paper if helps your research:

```BibTex
@article{abdallah2025rankarena,
  title={Rankarena: A unified platform for evaluating retrieval, reranking and rag with human and llm feedback},
  author={Abdallah, Abdelrahman and Abdalla, Mahmoud and Piryani, Bhawna and Mozafari, Jamshid and Ali, Mohammed and Jatowt, Adam},
  journal={arXiv preprint arXiv:2508.05512},
  year={2025}
}
```