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
| mT5-XLSum | Abstractive | Local CPU | 44 languages |
| LED-Large-BookSum | Abstractive | Colab GPU | English (long docs) |
| Flan-T5-Large | Abstractive | Colab GPU | English |
| mBART-Turkish | Abstractive | Colab GPU | Turkish |
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

## Evaluation Metrics

- **ROUGE-1/2/L** — Lexical overlap (precision, recall, F1)
- **Information Density** — Key term retention from source
- **Coherence** — Sentence-to-sentence flow quality
- **Redundancy** — Internal repetition detection
- **Coverage** — Source content representation
- **Topic Coverage** — Key topic retention (top-20 phrases)
- **Overall Quality** — Weighted composite score

## Tech Stack

- **Backend**: Python 3.9, FastAPI, PyTorch, Hugging Face Transformers, NLTK, scikit-learn, rouge-score
- **Frontend**: Vanilla HTML/CSS/JS with Server-Sent Events
- **GPU Server**: Flask, pyngrok, Google Colab (T4 GPU)
- **APIs**: OpenAI, Anthropic (optional)
