# Experimental Methodology

Written to be transferable into the thesis's methodology chapter. Every procedure below states
what was done and **why**, because several choices exist specifically to prevent a known bias.

---

## 1. Systems under evaluation

Ten systems spanning four paradigms, so that differences can be attributed to architecture and
training data rather than to incidental implementation details.

| System | Paradigm | Parameters | Context | Trained for |
|---|---|---|---|---|
| TextRank | Unsupervised graph ranking | — | unbounded | language-agnostic |
| DistilBART-CNN | Distilled encoder-decoder | 306M | 1,024 | English |
| BART-Large-CNN | Encoder-decoder | 406M | 1,024 | English |
| Pegasus-CNN/DailyMail | Gap-sentence pretraining | 568M | 1,024 | English |
| mT5-XLSum | Multilingual encoder-decoder | 582M | 512 | 44 languages |
| LED-Large-BookSum | Long-context encoder-decoder | 460M | 16,384 | English |
| Flan-T5-Large | Instruction-tuned | 783M | 1,024 | English |
| mBART-Turkish | Encoder-decoder, fine-tuned | 611M | 1,024 | Turkish |
| Qwen2.5-7B-Instruct | Decoder-only LLM (4-bit) | 7.6B | 3,072 | multilingual |
| GPT-4o-mini | Commercial LLM API | undisclosed | 128,000 | multilingual |

TextRank is the non-neural baseline. GPT-4 (legacy) was **excluded**: it costs ~$0.16/document and
its 8k context fails on 9 of the 100 English documents, since API models are not chunked.

---

## 1a. Why these systems, and why not others

Selection aimed at **paradigm coverage**: each system is present to answer a question no other
system in the set can answer.

| System | Why it is in the study |
|---|---|
| TextRank | Non-neural baseline. If a transformer cannot beat unsupervised graph ranking, the added cost is unjustified — and on faithfulness it does not. |
| DistilBART-CNN | The practical default: distilled, fast enough for CPU, the model most people would actually deploy. |
| BART-Large-CNN | The undistilled version of the same architecture. Isolates **the cost of distillation** by holding architecture and training data fixed. |
| Pegasus-CNN/DailyMail | Pretrained with gap-sentence generation, an objective designed *for* summarization. Isolates **pretraining objective** from architecture. |
| mT5-XLSum | Multilingual, trained on 44 languages. Tests whether multilinguality substitutes for language-specific training. |
| LED-Large-BookSum | 16,384-token context. The direct test of **long context versus hierarchical chunking** — research question 2. |
| Flan-T5-Large | Instruction-tuned, summarizes via prompt rather than fine-tuning. Also the largest open seq2seq here, which tests the size-versus-training-data question. |
| mBART-Turkish | Purpose-built for Turkish. The counterpart to mT5: **language-specific fine-tuning versus broad multilinguality**. |
| Qwen2.5-7B-Instruct | Open decoder-only LLM. Represents the LLM paradigm without a commercial dependency, and is the only system here an examiner can reproduce at zero cost. |
| GPT-4o-mini | Commercial API baseline, 128k context, no chunking required. The practical upper bound most users would reach for. |

**Systems deliberately excluded, with reasons:**

- **GPT-4 (legacy)** — ~$0.16/document, and its 8,192-token context fails on 9 of 100 English
  documents since API models are not chunked. A model that errors on the longest documents cannot
  participate in a long-document study. Cost alone would have been ~$41 for the full corpus
  against ~$0.25 for GPT-4o-mini.
- **Claude (Sonnet/Haiku)** — no API key available. Qwen2.5-7B-Instruct fills the same
  "instruction-tuned LLM" slot at zero cost, so the paradigm is still represented.
- **Fine-tuned mT5-Turkish** — the fine-tuning notebook exists and the loader detects the model
  automatically, but the trained weights were never produced. Listed as future work rather than
  silently omitted.

**Why Qwen2.5-7B-Instruct specifically:** it had to fit alongside seven other models on one 16 GB
T4, so it runs in 4-bit NF4 quantization (~5 GB). Pre-quantized weights are used rather than
quantizing at load time, because the full-precision repository downloads 15.2 GB per session
against 5.5 GB — and Colab re-downloads everything on every restart. *Limitation to report:*
4-bit quantization costs some quality relative to full precision, so Qwen is a lower bound on
open-LLM performance.

**Its context is 3,072 tokens, not the 8,192 the model supports.** The declared 5.56 GB weight
footprint proved to be 9.26 GB in measurement, leaving ~5.3 GB of a 14.56 GB T4 for the KV cache;
at 8,192 tokens the two longest English documents exhausted it and died before producing a chunk.
Every Qwen figure in this study therefore comes from the 3,072-token configuration, which chunks
Turkish documents ~3.2 times. This is not a detail: chunk count is what erodes output-language
fidelity (§4a), so Qwen is evaluated as *model plus deployment*, and the contrast with
GPT-4o-mini's single-pass 128k context confounds model with context length.

---

## 2a. Why these corpora, and why not others

**Requirement:** documents of 4–6 pages — long enough to exceed the context window of most systems
under test — each paired with a **human-written** reference summary.

**English — selected:**

- **arXiv** (`ccdv/arxiv-summarization`, Cohan et al., 2018) — 203,037 documents, median ~4,900
  words, author abstracts as references. Established in the summarization literature, so results
  are comparable to prior work.
- **PubMed** (`ccdv/pubmed-summarization`, same authors) — 119,924 documents, ~3,000 words.
  Included **alongside** arXiv rather than instead of it, so findings are not an artefact of a
  single domain's conventions.

**English — considered and rejected:** GovReport (median ~5,757 words — valid, but longer than the
4–6 page target and would have skewed the length distribution); BookSum (narrative chapters,
~5,340 words — a different discourse structure that would confound the comparison); CNN/DailyMail
and XSum (news-length, far too short for a long-document study).

**Turkish — the problem.** Candidate datasets were measured directly rather than assumed:

| Candidate | Median document | Verdict |
|---|---|---|
| `batubayk/TR-News` | **341 words** | Too short — news-length |
| `musabg/wikipedia-tr-summarization` | **277 words** | Too short |
| MLSum-tr, XLSum-tr | news-length, 1–2 sentence summaries | Too short |

**No public Turkish long-document summarization dataset exists.** This is a finding, not an
inconvenience, and it is reported as one.

**Turkish — how the corpus was built.** Three selection strategies were tried and measured:

1. **Random article sampling** — rejected. Turkish Wikipedia's median article is a stub; the yield
   of articles over 1,200 words was effectively zero.
2. **Longest pages by byte size** (`Special:LongPages`) — rejected. The longest pages are
   football-fixture and discography tables: 1.8 MB of markup containing almost no prose.
3. **Featured ("Seçkin maddeler") and good ("Kaliteli maddeler") articles** — selected. 574
   candidates, community-reviewed for quality, 1,200–8,500 words of genuine prose.

The **lead section is used as the reference summary** — human-written, standalone, and the
convention used by WikiSum-style work. It is **stripped from the document body** so no system can
copy its own reference out of the input. Collection respects Wikipedia's rate limits with
exponential backoff (37 backoffs occurred during the build of 100 documents).

*Limitation:* a Wikipedia lead is descriptive rather than abstractive, and is shorter relative to
its document than an academic abstract. This differs from the English references and is a
confound between the two language sets — it is not a controlled comparison of languages, but of
language-plus-genre.

## 2. Corpus

Long documents are the object of study, so every document exceeds the context window of most
systems under test — the condition the thesis is about.

| Source | Language | n | Median | Reference summary |
|---|---|---|---|---|
| arXiv | English | 50 | 3,201 w | author abstract |
| PubMed | English | 50 | 2,827 w | author abstract |
| Turkish Wikipedia (featured/good) | Turkish | 100 | 2,287 w | lead section |

Documents are filtered to 1,200–6,000 words. Sampling uses fixed seeds and records provenance per
document; each experiment snapshots the exact corpus it used.

**Turkish required building a corpus.** No public Turkish long-document summarization dataset
exists: TR-News averages 341 words, `musabg/wikipedia-tr-summarization` 277. Turkish Wikipedia's
reviewed articles supply long, edited prose whose lead section is a human-written standalone
summary (the WikiSum convention). The lead is **removed from the body** so no system can copy its
reference. *Limitation:* a lead is descriptive rather than a true abstract, and is shorter relative
to its document than an academic abstract.

## 3. Procedure

**Cross-lingual application.** Every system is run on every document regardless of its training
language. Each result is labelled `in_language`, `multilingual`, or `cross_language`. Mismatched
pairs are evidence about generalization, not errors.

**Length control.** Summary length is set per document toward a common target (200 words) rather
than a fixed ratio of source length. *Why:* under a fixed ratio TextRank produced 1,883-word
summaries against DistilBART's 274 on the same document, and coverage and density scale with
length, so the comparison was unfair by construction. Exact parity is impossible across
architectures, so length is recorded per run and used as a covariate.

**Chunking.** Systems whose context is shorter than the document use hierarchical chunking:
split with sentence-aligned overlap, summarize each chunk, concatenate, then summarize the
concatenation, recursing while it remains too long. API models receive the whole document, since
their context permits it — **this asymmetry is deliberate and must be reported**, as it compares
each system at its native capability.

**Execution.** One model is loaded and run across all documents before the next is loaded (10
loads rather than 1,000). Each completed document × model pair is written immediately, making runs
resumable across GPU session expiry. Inference runs on a single NVIDIA T4; evaluation runs locally.

## 4. Metrics

Two families, reported separately and never combined into a single ranking.

**Reference-based** — only computed when a human reference exists:
ROUGE-1/2/L/Lsum (Lin, 2004), METEOR (Banerjee & Lavie, 2005), BERTScore (Zhang et al., 2020).

**BERTScore is reported standalone and is not a component of any composite.** The weight table
assigns it 0.20, but including it would redefine the composite for every run after the composite had
already been rescored once, and — more importantly — averaging a semantic metric into a surface
composite destroys the comparison it exists to provide. The question it answers is whether a
semantic measure agrees with n-gram overlap, and on Turkish it does: both place the commercial LLM
above the extractive baseline where the composite does not. It uses `roberta-large` for English and
`bert-base-multilingual-cased` for Turkish, so **BERTScore is comparable within a language and not
between languages** — the two columns are different instruments, and the gap between them is not a
language effect.

**Intrinsic** — computed against the source document:
compression ratio; information density (retention of top-50 TF-IDF terms); topic coverage
(top-20 key phrases); source coverage (share of source sentences represented); lexical cohesion
(adjacent-sentence similarity, a proxy for coherence); redundancy and distinct-n; **novel n-gram
ratio** (See et al., 2017); **extractive fragment coverage and density** (Grusky et al., 2018);
**factual-consistency proxy** (numbers and proper nouns absent from the source); readability
(Flesch for English, LIX otherwise).

**ROUGE is reported only against a human reference.** Scoring a summary against its own source is
not ROUGE: recall is bounded by compression and precision measures only copying. That quantity is
still informative and is reported separately as the **copy rate**.

**All tokenization is Unicode-aware.** The reference implementation of ROUGE discards non-ASCII
characters, which silently destroys Turkish; stemming is applied to English only.

**Output language is verified, not assumed.** Every summary's language is detected
(`langdetect`, fixed seed) and compared against the document's language, because several systems
produce fluent, well-formed output in the wrong language — Qwen answered 48 of 100 Turkish
documents in English and 12 in Chinese; LED answered 96 of 100 in English. No reference-based or
intrinsic measure detects this: ROUGE falls because the reference is in another language, which is
correct arithmetic for the wrong reason, since it scores a *faithful translation* and a *fluent
fabrication* at the same ~2–4%. Results are therefore reported conditioned on output language
where it varies, and the aggregate is reported as a mixture rather than a central tendency.

**Length measures are script-dependent and are not comparable across scripts.** The word count
splits on whitespace, so the Chinese summaries register as 1–10 words despite being full
paragraphs. Runs whose output script differs from the corpus are excluded from length-derived
statistics rather than averaged into them.

**Empty and degenerate output is identified before any metric is read.** A summary of zero length
satisfies every containment- and similarity-based measure vacuously — faithfulness 100% (nothing to
contradict), lexical cohesion 100% (nothing incoherent), redundancy 0% (nothing repeated) — and
composites to 25.0%, which is *above* the mean of the system that produced it on documents where it
produced text. Pegasus returned empty output on 21 of 200 documents. Those runs are **excluded from every metric,
interval and significance test, and the documents they occurred on are excluded for all systems**,
so paired comparisons still rest on one identical document set (n = 88 English, 91 Turkish). The
silent-failure rate is reported separately, as a property of the system. The alternative — scoring
degenerate runs as zero — was rejected: it preserves n but invents a value for a measurement never
made, and the choice between 0 and "undefined" would itself drive the comparison. The
underlying point is general: a reference-free suite assembled from containment and similarity
cannot distinguish "said nothing" from "said nothing wrong".

**Composite scores** are weighted means within each family, reported with their weights and
components. The weights are a stated modelling choice, not an empirical result.

## 5. Statistical analysis

Documents are the unit of analysis. For each system and metric:

- **Mean with a 95% percentile bootstrap confidence interval** (10,000 resamples, fixed seed).
- **Paired bootstrap resampling test** (Koehn, 2004) for system comparisons: documents are
  resampled with each system's paired scores kept together, which suits evaluation on a common
  document set and assumes no particular distribution.
- **Wilcoxon signed-rank** as a rank-based cross-check.
- **Holm-Bonferroni correction** across all pairwise comparisons, since comparing k systems means
  k(k−1)/2 tests.
- **Cohen's d** for paired samples as effect size.

**No p-value is reported below five documents.** A paired bootstrap over two documents draws from
three distinct resamples and returns an arbitrarily small p whenever both documents agree. That
number reflects the resampling procedure, not evidence.

Languages are aggregated separately. Only systems that completed every document of a language
enter that language's statistics, so all compared means rest on the same document set.

### 5a. Why 100 documents per language

Sample size was fixed empirically rather than by convention. The rule set *before* the run: begin
at n=100 and extend to ~300 only if confidence intervals proved too wide to separate systems.

Measured at n=100 (before the degenerate-output exclusion described in §4, which reduces the analysed set to n=88 English and n=91 Turkish without materially changing any interval), 95% bootstrap CI widths across the headline metrics are **0.8 to 6.7 percentage
points (median 2.0)**, while between-system differences reach 24.8 points on information density
and 28.2 on topic coverage. Composite-quality intervals do not overlap between the extractive
baseline and either neural system. Extending to 300 documents would narrow intervals by a factor
of √3 ≈ 1.7 — changing no conclusion at three times the computational cost.

Where intervals overlap — factual consistency, 0.1-point spread across systems — the correct
reading is a null result rather than insufficient power: additional documents cannot produce a
difference that does not exist. Claims on metrics whose spread is comparable to their interval
width (ROUGE-L, 2.2-point spread against ~2.4-point intervals) are reported as "not clearly
separated" rather than as a ranking.

## 6. Reproducibility

Each experiment writes a self-contained folder: the configuration and machine fingerprint, a
snapshot of the exact corpus, one row per document × model with all metrics, timing and cost, and
per-document folders holding the source text, the reference, and every system's summary. Any
figure in the thesis can be traced to the text that produced it. Bootstrap seeds are fixed, so
re-analysis reproduces identical intervals.

### 6a. Efficiency measurement

Every run records wall-clock time, process CPU time, GPU time as reported by the inference server,
and **the number of chunk passes performed**. Chunk count is the mechanism linking context window
to cost, so it is recorded as data rather than inferred afterwards.

Measured on the English corpus (median 3,015 words): models with a 1,024-token window require
5.8–5.9 chunk passes per document regardless of parameter count (306M–568M), while a 512-token
window requires 17.1 — the chunk count tracks the window, not the model size. Within a fixed
window, decoding cost still differs (BART-Large 10.8 s/document against Pegasus 4.2 s at identical
chunk counts), so runtime is explained by chunk count *across* window sizes and by model size and
decoding configuration *within* one.

Reported per model: seconds/document, mean chunks/document, context window, and share of total
compute. These are generated from the recorded data by `scripts/aggregate.py`, not transcribed.

## 8. Reporting conventions

How a result is phrased determines whether it survives examination. These rules are applied
throughout, and are stated here so the phrasing is visibly a policy rather than an accident.

**State the mechanism, not the magnitude.** A measurement without a cause is an anecdote.

> ✗ "mT5-XLSum was much slower than the other models."
> ✓ "mT5-XLSum required 47.6 s/document against DistilBART's 6.8. Its 512-token window forces 18.5
> chunk passes per document versus 5.8, so runtime scales with context window rather than with
> parameter count (582M vs 306M)."

**Name the bias in the same sentence as the result it favours.** Raising the objection first is
stronger than answering it later.

> ✗ "TextRank achieved the highest information density (57.9%)."
> ✓ "TextRank achieved the highest information density (57.9% vs 33.1%) — partly by construction,
> since the measure rewards retaining source vocabulary and TextRank copies sentences verbatim."

**When intervals overlap, say so; do not rank.** A ranking the data cannot support is the easiest
error for an examiner to find.

> ✗ "TextRank outperformed DistilBART on ROUGE-L (21.4% vs 19.2%)."
> ✓ "TextRank and DistilBART are not clearly separated on ROUGE-L (21.4% [20.1, 22.8] vs 19.2%
> [18.2, 20.3]); the 2.2-point difference is comparable to the interval width."

**Report a null result as a finding, not as missing power.**

> ✗ "No significant difference in faithfulness was detected."
> ✓ "Faithfulness is ~99.6% for all in-language English systems (0.1-point spread). Hallucination
> does not differentiate these systems on in-language input; it appears under language mismatch."

**Separate descriptive measures from evaluative ones.** Novel n-gram ratio and extractive density
describe *how* a summary was produced. Higher is not better — novelty is simultaneously abstraction
and hallucination.

> ✗ "GPT-4o-mini scored best on abstractiveness."
> ✓ "GPT-4o-mini produced the most novel n-grams (26.4%), i.e. it paraphrases most heavily. Read
> together with its faithfulness score, this indicates genuine rewriting rather than fabrication."

**Give a defect a direction.** "There was a bug" is an apology; a biased measurement is a result.

> ✗ "ROUGE was computed incorrectly and has been fixed."
> ✓ "ROUGE was computed against the source rather than a reference, which rewards copying. The
> defect therefore inflated extractive scores relative to abstractive ones, biasing the earlier
> extractive-vs-abstractive comparison in a known direction."

**Distinguish measured from estimated.** Every number in the results is measured and traceable to
a stored run; projections are labelled as such.

**Attribute causes only where the design isolates them.** BART-Large vs DistilBART isolates
distillation because architecture and training data are held constant. Comparisons across systems
differing in several respects at once are described, not explained.

### 8a. A confound the design does not resolve

LED-Large-BookSum is the study's only long-context model, and it is fine-tuned on narrative book
chapters while the English corpus is scientific papers. Any quality difference between LED and the
chunking models therefore admits two explanations — context handling, or domain fit — which this
design cannot separate.

Consequently LED's results are reported as **evidence on efficiency** (chunk count and runtime,
which are properties of the context window and not of training domain) and **not as evidence on
quality**. Resolving the quality question requires either a long-context model trained on
scientific text, or evaluation on a narrative corpus matching LED's training domain. This is
stated as future work rather than argued around.

### 8b. Automatic metrics presuppose well-formed output

Every metric in this study assumes its input is a summary in the expected language. Under severe
distribution shift that assumption fails silently, and the measures degrade gracefully rather than
signalling breakdown.

Demonstrated case: a Turkish-trained model applied to English papers produced LaTeX fragments and
Turkish function words in broken English, and scored **100% factual consistency** (noise contains
no checkable entities to contradict), **38.5% abstractiveness** (noise diverges maximally from
source wording), and **near-zero redundancy** (random tokens do not repeat). Its composite score,
21.8%, is in the range of a merely weak summarizer.

Consequently: **aggregate metrics alone cannot establish that a system produced language.** Every
condition in this study with low quality and high abstractiveness is verified by reading sampled
outputs before any claim is made about it, and faithfulness is interpreted as "nothing checkable
was contradicted" rather than as evidence of a faithful summary.

## 9. Stated limitations

Composite weights are a modelling choice. Information density and source coverage reward reusing
source vocabulary, systematically favouring extractive systems. Lexical cohesion is a proxy for
coherence and cannot judge logical flow. The factual-consistency measure is a surface check over
numbers and proper nouns and cannot detect distorted relations between facts. LIX penalizes
agglutinative languages. Wikipedia lead sections are descriptive rather than abstractive
references. No human evaluation was conducted. API models receive whole documents while local
models receive chunks — and since chunk count erodes output-language compliance (§4a), that
asymmetry is not only a compute advantage but a correctness one, which the design does not
separate from model quality. Qwen runs at 3,072 of its 8,192 tokens for memory reasons, so its
results describe a deployment rather than the model. Whitespace word counts are invalid for
unspaced scripts. One system returned empty output on 21 of 200 documents while scoring 25.0%
composite on those runs, so every reference-free composite in this study should be read alongside
that system's silent-failure rate rather than on its own. None of these measures substitutes for human judgement.
