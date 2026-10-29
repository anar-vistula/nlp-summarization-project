# Long Document Summarization System

A multilingual text summarization system comparing extractive (TextRank) and transformer-based abstractive methods with comprehensive evaluation metrics. Supports English and Turkish with multiple models including GPU-accelerated inference via Google Colab.

## Architecture

```
├── backend/
│   ├── preprocessing/       # Language detection, tokenization, chunking
│   ├── summarizers/         # Extractive (TextRank), Abstractive (Transformers), Paid APIs
│   ├── evaluation/          # ROUGE, coherence, coverage, density, topic analysis
│   ├── api/                 # FastAPI REST endpoints + SSE streaming
│   ├── comparison.py        # Multi-model comparison service
│   └── service.py           # Main orchestration service
├── frontend/                # Web UI (vanilla HTML/CSS/JS)
├── colab_gpu_server.ipynb   # Google Colab notebook for GPU models
├── requirements.txt
├── .env.example             # Environment variable template
└── run.py                   # Server entry point
```

## Models

| Model | Type | Runs On | Language |
|-------|------|---------|----------|
| TextRank | Extractive | Local CPU | All |
| DistilBART-CNN | Abstractive | Local CPU | English |
| BART-Large-CNN | Abstractive | Local CPU | English |
| Pegasus-CNN/DailyMail | Abstractive | Local CPU | English |
| mT5-XLSum | Abstractive | Local CPU | 44 languages |
| mT5-Turkish (fine-tuned) | Abstractive | Local CPU | Turkish (optional, see below) |
| LED-Large-BookSum | Abstractive | Colab GPU | English (long docs) |
| Flan-T5-Large | Abstractive | Colab GPU | English |
| mBART-Turkish | Abstractive | Colab GPU | Turkish |
| Qwen2.5-7B-Instruct | Abstractive | Colab GPU (4-bit) | Multilingual |
| GPT-4o-mini / GPT-4 | Abstractive | OpenAI API | All |

## Setup

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download NLP data
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

### 2. Configure Environment

Copy `.env.example` to `.env` and fill in:

```bash
cp .env.example .env
```

```
OPENAI_API_KEY=sk-...          # For GPT-4/GPT-4o-mini (optional)
ANTHROPIC_API_KEY=sk-ant-...   # For Claude models (optional)
COLAB_GPU_URL=https://...      # ngrok URL from Colab notebook (optional)
```

The system works without any keys — TextRank and local models are always available.

### 3. Start the Server

```bash
python run.py
```

Open `http://localhost:8000` in your browser. The frontend is served automatically.

### 4. (Optional) Start Colab GPU Server

For the heavy models (LED-Large, Flan-T5, mBART-Turkish):

1. Open `colab_gpu_server.ipynb` in Google Colab
2. Set runtime to **T4 GPU** (Runtime → Change runtime type)
3. Paste your ngrok token in Cell 6
4. Run all cells — you'll get a public URL
5. Put that URL in your `.env` as `COLAB_GPU_URL=https://...`
6. Restart your local server

The Colab notebook must stay open while using GPU models.

The v2 notebook hosts **every** heavy model, so the local machine only runs TextRank and the
evaluation metrics. Models are loaded on demand and evicted under a VRAM budget; T5-family
models load in fp32 because they emit NaNs in fp16 on a T4. Pass `--prefer-colab` to
`scripts/run_experiment.py` to route all hosted models to the GPU.

### 5. (Optional) Fine-Tuned Turkish Model

Run `fine_tuning_notebook.ipynb` on Colab, download the resulting zip, and unzip it into
`models/mt5-turkish-finetuned/` in the project root. It is detected automatically on the
next server start and used for Turkish text in place of mT5-XLSum.

## Usage

- **Summarize & Compare** — Quick comparison of extractive vs abstractive
- **Compare All Models** — Run all available models and see detailed metrics for each
- **Extractive Only / Abstractive Only** — Single method results

Sample texts are provided (Climate Change, AI, Space Exploration, Public Health).

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/summarize/stream` | POST | SSE streaming (preprocessing → extractive → abstractive → metrics) |
| `/summarize/compare` | POST | Both methods with full metrics |
| `/summarize/extractive` | POST | TextRank only |
| `/summarize/abstractive` | POST | Transformer only |
| `/compare/all` | POST | All available models with comparison |
| `/models` | GET | List available models for a language |
| `/colab/status` | GET | Check if Colab GPU server is reachable |
| `/samples` | GET | List sample texts |
| `/metrics/metadata` | GET | Every metric's definition, direction, source and caveat |
| `/experiment/documents` | GET | Documents available for the experiment runner |
| `/experiment/run` | POST | SSE stream: all models × all documents, then aggregate statistics |
| `/experiment/last` | GET | Full result of the last experiment run |
| `/experiment/last.csv` | GET | Per-document, per-model scores as CSV |

## Evaluation Metrics

Two families, reported separately and never mixed.

**Reference-based** (require a human reference summary; omitted without one)

- **ROUGE-1/2/L/Lsum** — n-gram and LCS overlap with the reference (Lin, 2004)
- **METEOR** — unigram alignment with stem/synonym matching (Banerjee & Lavie, 2005)
- **BERTScore** — embedding similarity (optional, `use_bertscore=True`; Zhang et al., 2020)

**Intrinsic** (computed against the source document)

- **Compression Ratio** — summary length relative to the source
- **Information Density** — retention of the source's top-50 TF-IDF terms
- **Topic Coverage** — retention of the top-20 key phrases
- **Source Coverage** — fraction of source sentences represented
- **Lexical Cohesion** — adjacent-sentence similarity (proxy for coherence)
- **Redundancy / distinct-n** — internal repetition and decoder looping
- **Abstractiveness** — novel n-gram ratio (See et al., 2017)
- **Extractive Coverage & Density** — how much is copied, and in how long runs (Grusky et al., 2018)
- **Factual Consistency (proxy)** — numbers and proper nouns in the summary that do not appear in the source
- **Readability** — Flesch Reading Ease (English) and LIX (any language)
- **Overall Quality** — weighted composite of each family; weights are reported with the score

Tokenization is Unicode-aware throughout, so Turkish is scored on its actual characters
rather than the ASCII-stripped text the default ROUGE tokenizer produces.

## Statistical Evaluation

The **Experiment** tab runs every model over a set of documents and reports:

- per-model means with **95% percentile bootstrap confidence intervals**
- **paired bootstrap resampling tests** (Koehn, 2004), cross-checked with **Wilcoxon signed-rank**
- **Holm–Bonferroni** correction across all pairwise comparisons, and Cohen's d effect sizes
- a CSV export of the raw per-document scores (`GET /experiment/last.csv`)

Languages are aggregated separately, because the available model sets differ by language.
Below five documents per language, differences are reported descriptively and **no p-values are
given** — a paired bootstrap over two documents returns an artefact of the resampling procedure
rather than evidence.

## Tests

```bash
./venv/bin/python -m pytest tests -q
```

## Tech Stack

- **Backend**: Python 3.9, FastAPI, PyTorch, Hugging Face Transformers, NLTK, scikit-learn, rouge-score
- **Frontend**: Vanilla HTML/CSS/JS with Server-Sent Events
- **GPU Server**: Flask, pyngrok, Google Colab (T4 GPU)
- **APIs**: OpenAI, Anthropic (optional)
