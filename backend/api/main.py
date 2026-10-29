"""
FastAPI REST API for the Text Summarization System.
Exposes endpoints for summarization and evaluation.
"""

import json
import time
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.responses import StreamingResponse

from backend.api.schemas import (
    SummarizeRequest,
    SummarizeResponse,
    ExtractiveResponse,
    AbstractiveResponse,
    HealthResponse,
    ExperimentRequest,
)
from backend.service import SummarizationService
from backend.comparison import ComparisonService
from backend.evaluation.experiment import ExperimentRunner
from backend.evaluation.metrics import SummaryEvaluator
from backend.sample_texts import get_sample, get_all_samples

app = FastAPI(
    title="Text Summarization API",
    description=(
        "A REST API for long document summarization using extractive (TextRank) "
        "and abstractive (Transformer-based) methods with comprehensive evaluation metrics."
    ),
    version="1.0.0",
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services (lazy model loading)
service = SummarizationService(model_name="auto", use_bertscore=False)
comparison_service = ComparisonService()
experiment_runner = ExperimentRunner(comparison_service, evaluator=service.evaluator)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and available methods."""
    return HealthResponse(
        status="healthy",
        model_loaded=service.abstractive._pipeline is not None,
        available_methods=["extractive_textrank", "abstractive_transformer"],
    )


@app.post("/summarize/compare", response_model=SummarizeResponse)
async def summarize_and_compare(request: SummarizeRequest):
    """
    Generate summaries using both extractive and abstractive methods,
    evaluate them, and return a comparative analysis.
    This is the primary endpoint for the thesis evaluation.
    """
    try:
        result = service.summarize_and_compare(
            text=request.text,
            num_sentences=request.num_sentences,
            ratio=request.ratio,
            reference_summary=request.reference_summary,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


@app.post("/summarize/extractive", response_model=ExtractiveResponse)
async def summarize_extractive(request: SummarizeRequest):
    """Generate an extractive summary using TextRank."""
    try:
        result = service.summarize_extractive(
            text=request.text,
            num_sentences=request.num_sentences,
            ratio=request.ratio,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extractive summarization failed: {str(e)}")


@app.post("/summarize/abstractive", response_model=AbstractiveResponse)
async def summarize_abstractive(request: SummarizeRequest):
    """Generate an abstractive summary using a Transformer model."""
    try:
        result = service.summarize_abstractive(
            text=request.text,
            max_length=request.max_length,
            min_length=request.min_length,
            ratio=request.ratio,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Abstractive summarization failed: {str(e)}")


@app.post("/evaluate")
async def evaluate_summary(
    text: str,
    summary: str,
    reference_summary: str = None,
):
    """Evaluate a pre-generated summary against the source text."""
    try:
        metrics = service.evaluator.evaluate(text, summary, reference_summary)
        return {"metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.post("/summarize/stream")
async def summarize_stream(request: SummarizeRequest):
    """
    Stream summarization results via Server-Sent Events.
    Sends incremental updates as each stage completes:
    1. preprocessing → input stats
    2. extractive → extractive summary
    3. abstractive → abstractive summary
    4. metrics → evaluation metrics for both
    5. done → final signal
    """

    def event_generator():
        try:
            # Stage 1: Preprocessing (includes language detection)
            preprocessed = service.preprocessor.preprocess(request.text)
            lang = preprocessed["language"]
            yield f"data: {json.dumps({'stage': 'preprocessing', 'data': {'word_count': preprocessed['word_count'], 'sentence_count': preprocessed['num_sentences'], 'chunk_count': preprocessed['num_chunks'], 'language': lang}})}\n\n"

            # Stage 2: Extractive summarization (timed, for the efficiency comparison)
            started = time.perf_counter()
            extractive_result = service.extractive.summarize(
                request.text,
                num_sentences=request.num_sentences,
                ratio=request.ratio,
                lang=lang,
            )
            extractive_result["runtime_seconds"] = round(time.perf_counter() - started, 3)
            yield f"data: {json.dumps({'stage': 'extractive', 'data': extractive_result})}\n\n"

            # Stage 3: Abstractive summarization
            started = time.perf_counter()
            abstractive_result = service.abstractive.summarize(
                request.text, ratio=request.ratio, lang=lang
            )
            abstractive_result["runtime_seconds"] = round(time.perf_counter() - started, 3)
            yield f"data: {json.dumps({'stage': 'abstractive', 'data': abstractive_result})}\n\n"

            # Stage 4: Metrics evaluation
            summaries = {
                "extractive_textrank": extractive_result["summary"],
                "abstractive_transformer": abstractive_result["summary"],
            }
            comparison = service.evaluator.compare_summaries(
                request.text, summaries, reference_summary=request.reference_summary, lang=lang
            )
            yield f"data: {json.dumps({'stage': 'metrics', 'data': comparison})}\n\n"

            # Stage 5: Done
            yield f"data: {json.dumps({'stage': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'stage': 'error', 'data': {'message': str(e)}})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/models")
async def get_available_models(lang: str = "en", cross_lingual: bool = False):
    """
    Available models for a language.

    With cross_lingual=true, models trained on other languages are included as
    well, each labelled with its language_match.
    """
    models = comparison_service.get_available_models(lang, cross_lingual=cross_lingual)
    return {"models": models, "language": lang, "cross_lingual": cross_lingual}


@app.get("/colab/status")
async def colab_status():
    """Check if the Colab GPU server is reachable."""
    from backend.summarizers.paid_models import ColabGPUClient
    client = ColabGPUClient()
    if not client.is_available():
        return {"connected": False, "reason": "COLAB_GPU_URL not set in .env"}
    models = client.get_available_models()
    if models:
        return {"connected": True, "models": models, "url": client.base_url}
    return {"connected": False, "reason": "Server not reachable", "url": client.base_url}


@app.post("/compare/all")
async def compare_all_models(request: SummarizeRequest):
    """
    Run ALL available models on the same text and compare results.
    This is the primary endpoint for the thesis multi-model comparison.
    """
    try:
        result = comparison_service.compare_all(
            text=request.text,
            reference_summary=request.reference_summary,
            ratio=request.ratio,
            cross_lingual=request.cross_lingual,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@app.get("/metrics/metadata")
async def metrics_metadata():
    """
    Description of every metric: what it measures, which direction is better,
    its source in the literature, and its caveat. The UI renders its labels
    and tooltips from this, so documentation cannot drift from the code.
    """
    return {"metrics": SummaryEvaluator.metric_metadata()}


@app.get("/experiment/documents")
async def experiment_documents():
    """Documents available as experiment material."""
    return {"documents": ExperimentRunner.list_documents()}


@app.post("/experiment/run")
async def experiment_run(request: ExperimentRequest):
    """
    Run the controlled experiment: every available model over every selected
    document, then means with bootstrap confidence intervals and pairwise
    significance tests.

    Streams Server-Sent Events (`plan`, `result`, `error`, `aggregate`,
    `done`) because a full run over several documents takes minutes on CPU.
    """

    def event_generator():
        try:
            events = experiment_runner.run(
                document_keys=request.document_keys,
                model_keys=request.model_keys,
                ratio=request.ratio,
                extra_documents=[d.model_dump() for d in (request.extra_documents or [])],
            )
            for event in events:
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'stage': 'error', 'data': {'message': str(e)}})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/experiment/last.csv")
async def experiment_last_csv():
    """Per-document, per-model scores of the last run as CSV."""
    # Fall back to the newest run saved on disk: the dev server reloads on every
    # code edit, which would otherwise discard a long run held in memory.
    csv_text = experiment_runner.last_run_csv() or experiment_runner.latest_saved_csv()
    if not csv_text:
        raise HTTPException(status_code=404, detail="No experiment has been run yet")
    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="experiment_results.csv"'},
    )


@app.get("/experiment/last")
async def experiment_last():
    """Full result of the last experiment run, including aggregates."""
    if not experiment_runner.last_run:
        raise HTTPException(status_code=404, detail="No experiment has been run yet")
    return experiment_runner.last_run


@app.get("/samples")
async def list_samples():
    """List available sample texts."""
    return {"samples": get_all_samples()}


@app.get("/samples/{sample_key}")
async def get_sample_text(sample_key: str):
    """Get a specific sample text with reference summary."""
    sample = get_sample(sample_key)
    if not sample:
        raise HTTPException(status_code=404, detail=f"Sample '{sample_key}' not found")
    return sample


# Serve frontend static files
FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def serve_frontend():
        """Serve the frontend UI."""
        return FileResponse(str(FRONTEND_DIR / "index.html"))
