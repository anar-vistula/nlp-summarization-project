# Findings index

Every established result, as a one-line claim with its evidence and where the detail lives.
Written for drafting the thesis: each entry is a sentence you could defend, not a number you
would have to re-derive.

Status (27 Oct 2026): **complete — 2,000 of 2,000 document x model cells**, all ten systems on
n=100 English and n=100 Turkish, with zero outstanding failures. The faithfulness fix and the
script-aware word count have been applied uniformly to every stored run (`rescore.py`), and the
aggregate, significance tests and output-language classification are regenerated from it. Numbers
below are post-rescore and include BERTScore. **A note on basis:** the final analysis excludes the
21 documents on which one system returned an empty summary (for all systems, so paired tests rest
on one identical set — see A7), giving n=88 English and n=91 Turkish. Entries written before that
exclusion quote n=100 figures; the difference was measured across all headline claims and is
**at most 0.7 points, with no claim changing direction**, so the arguments below stand as written.
Where an exact figure is needed for the thesis, take it from
`experiments/main_run/aggregate/report.md`, which is authoritative and regenerated from the final
data. The experiment ran
2026-10-19 to 2026-10-27, 16.4 hours of compute, $0.1904 of API spend. **No GPU or API work
remains.**

---

## A. Evaluation methodology — the strongest contribution

**A1. Surface-based summarization metrics fail under distribution shift, in seven independent
ways, with one underlying cause.**

| # | Artifact | Effect |
|---|---|---|
| 1 | ROUGE computed against the source when no reference exists | Inflated extractive scores; rewarded copying |
| 2 | Possessives counted as fabrication ("Einstein's" vs "Einstein") | Deflated abstractive scores; +21.9 pts for GPT-4o-mini once fixed |
| 3 | Degenerate output scored as faithful | 24/100 documents scored ≥90% faithful while <30% quality |
| 4 | Orthographically corrupted output scored as faithful | Substring containment: "Karyal" matches "karyalı" |
| 5 | Corpus hyphen tokenization vs. model hyphenation | **Deflated correct output**: an accurate technical summary scored 44.4%; +28.6 pts for GPT-4o-mini once fixed |
| 6 | Word count splits on whitespace, so Chinese output counts as 1–10 words | Corrupts every length-derived measure; a complete paragraph reads as a truncation (A6) |
| 7 | **An empty summary scores 100% faithfulness, 100% cohesion, 0% redundancy** | **Composite 25.0% — above Pegasus's own mean on documents where it produced text (A7)** |

**Artifacts 3 and 4 reward output that is less well-formed; artifact 5 punishes output that is
more well-formed.** A surface-based faithfulness measure is therefore not merely noisy — it is
systematically biased against fluent paraphrase and in favour of degenerate text.

The common cause: all four compare surface strings while assuming the output is well-formed text
in the expected language. **Artifacts 2 and 4 are the same design decision failing in opposite
directions** — strict matching flags ordinary morphology as fabrication, lenient matching lets
corruption pass. No threshold on surface form separates inflection from corruption.
→ OBSERVATIONS.md, "A second metric defect", "A model scores higher on a language it cannot write"

**A1b. On the most dangerous failure mode, reference-free metrics point the wrong way.**
LED-Large given Turkish input produced fluent, grammatical, entirely fabricated English at exactly
the target length (198 words; script fidelity 0.36% vs 10.55% source; invented entities verified
absent from the source). ROUGE-L caught it at **1.9%**; abstractiveness rated it **84.7%** and
redundancy **0.0%** — i.e. highly creative and perfectly non-repetitive. Reference-free metrics
are usually presented as a practical approximation when references are unavailable; here they do
not merely lose precision, they invert. **BERTScore, computed against the human reference, scores
that condition 38.0% — the lowest of any system in either language.** So the defence against fluent
fabrication is a reference-based semantic metric; where no reference exists, nothing in this suite
detects it. That is the actionable form of this finding.
→ OBSERVATIONS.md, "The fourth and most dangerous failure mode"

**A2. Automatic metrics cannot establish that a system produced language at all — or which
language.** mBART-Turkish on English produced LaTeX residue and Turkish function words, scoring
100% faithfulness and 38.5% abstractiveness on one document, 60.2% faithfulness across 100. Any
evaluation reporting aggregates alone would have recorded it as a weak-but-functional summarizer.
Quantified at n=100 by script-aware classification: **it answers English documents in Turkish 89
times out of 100** (10 English, 1 Danish), and on the 10 English-output runs it scores 10.8%
ROUGE-L against 5.3% on the Turkish-output ones. Language is now verified per run, not assumed
([scripts/output_language.py](scripts/output_language.py)).
→ OBSERVATIONS.md, "The central finding"

**A3. The composite quality score is confounded with summary length (r = 0.69).**
Retention metrics scale with length, so TextRank's lead is partly a length effect. ROUGE-L is
length-robust in this data (seven systems from 86–224 words all score 17–21%) and is the
defensible primary metric.
→ OBSERVATIONS.md, "Length control only partially worked"

**A4. Summary length is a property of the training distribution, not a parameter.**
A 200-word target was met by instruction-following models and the extractive method (95–112%)
and ignored by fine-tuned encoder-decoders (27–53%). `max_length` is a cap, not a target.
→ same section

**A5. Length-instruction compliance holds only in English, and tracks the *output* language.**
GPT-4o-mini hits 95% of the 200-word target in English and 68% in Turkish; Qwen hits 107% on the
English corpus, 96% when it drifts into English on a Turkish document, and **53% when it complies
and answers in Turkish**. Reference imitation is excluded — the Turkish references are *longer*
(median 224 vs 168 words), so the effect runs opposite to them. Candidate mechanism: the target is
honoured against a token budget, and Turkish costs 3.18 tokens/word against English's 1.21.
Consequence: the A3 length covariate must be fitted **within** language, never pooled.
→ OBSERVATIONS.md, "Length-instruction compliance is language-conditional"

**A6. A length measure cannot be trusted across scripts.** `summary_words` splits on whitespace,
so 12 Chinese summaries were recorded as 1–10 words when each is a 200–300 character paragraph.
This corrupts compression ratio, percentage-of-target and the length covariate, and it defeats the
obvious safeguard: a length-based degenerate-output filter reads a complete paragraph as a
truncation and misses the actual defect, which is the language. First artifact in this family where
the broken assumption is the **script** rather than the language. Unfixed; must be corrected before
any length-covariate analysis.
→ OBSERVATIONS.md, "A sixth metric artifact"

**A7. Producing nothing outscores producing something — the limiting case of the whole family.**
Pegasus returned an **empty summary on 21 of 200 documents** (12/100 English, 9/100 Turkish),
recorded `ok: True` with no error. An empty string satisfies faithfulness (nothing to contradict),
lexical cohesion (nothing incoherent) and redundancy (nothing repeated) *perfectly*, scoring
100%/100%/0%, and composites to **25.0% — above the 22.8% Pegasus averages on English documents
where it did produce text.** Two separable defects: a **metric** defect, since any suite built from
containment and similarity measures is vacuously satisfied by the empty set; and a **pipeline**
defect, since "the model returned without raising" admits an empty generation into every mean,
interval and significance test. **Resolved**: degenerate runs are excluded from
every metric, interval and significance test, and the 21 documents are excluded for all systems so
paired comparisons keep one identical document set (n = 88 EN, 91 TR); the silent-failure rate is
reported separately (**12% English, 9% Turkish**). The rejected alternative — scoring empty runs as
zero — would have kept n=100 but invents a value for a measurement never made, and the choice of
value would drive the result. **Removing them moves Pegasus in opposite directions per metric**:
faithfulness 73.2% → 69.5% (the empties scored 100%), ROUGE-L 9.8% → 11.1% (they scored 0%),
composite 23.0% → 22.8%. Every other system moved ≤0.5 points and no significance verdict changed.
→ OBSERVATIONS.md, "The sharpest metric artifact: an empty summary outscores a real one"

**A8. A semantic metric breaks the composite-versus-ROUGE tie, and it sides with ROUGE.** On
Turkish the composite ranks TextRank above GPT-4o-mini (40.9% vs 35.9%) while ROUGE-L reverses it
(11.8% vs 14.4%). **BERTScore agrees with ROUGE-L** (56.5% vs 59.8%). Two metrics sharing no
machinery — surface n-gram overlap and contextual-embedding similarity, both against the human
reference — place the commercial LLM above the extractive baseline; only the length-correlated
composite disagrees. The composite's extractive preference is therefore a demonstrated artifact
(A3, D1), not an argued one.
**A8b. BERTScore is not uniformly sharper — it is sharper where failures are semantic.** The top
five English systems fall inside 0.6 points (83.1–83.7%) where ROUGE-L spreads 18.9–21.5; on
Turkish it spreads 38.0–59.8. It adds almost nothing on English and a great deal on Turkish.
**A8c. It is not comparable across languages.** `bert-score` uses `roberta-large` for English and
`bert-base-multilingual-cased` for Turkish, so the ~27-point gap between the columns is partly two
different instruments. Within-language only — which is how this study aggregates regardless.
Measured cost: 112 s for 1,979 pairs batched per language, and it is reported **standalone, never
folded into the composite** (that would redefine `overall_quality` for every run and destroy the
contrast it was added to provide).
→ OBSERVATIONS.md, "BERTScore: the semantic metric settles the composite-versus-ROUGE disagreement"

---

## B. Cross-lingual transfer — four distinct failure mechanisms

**B0. The cross-lingual penalty must be measured as difference-in-differences.**
TextRank (language-agnostic) drops 4.9 composite points between corpora, establishing the baseline
difficulty difference. Against it: DistilBART loses **6.4 points** to language mismatch (the clean
transfer estimate); mBART-Turkish gains **9.6 points** in-language. Raw per-language scores
conflate transfer, corpus difficulty and domain fit.
→ OBSERVATIONS.md, "Cross-lingual penalty must be measured as difference-in-differences"

**B1. Transfer fails through different components, not by uniform degradation.**

| Mechanism | Example | Component at fault |
|---|---|---|
| Non-language output | mBART-TR on English: LaTeX noise | Decoder language model |
| Orthographic destruction | Pegasus on Turkish: 0.00% Turkish characters vs 10.55% in source | Tokenizer vocabulary coverage |
| Invented morphology | DistilBART on Turkish: "muhafazakârlarlarlar" | Subword segmentation |
| **Language substitution** | **Qwen on Turkish: fluent English, and Chinese on 12/100** | **Instruction compliance, eroded per chunk (B7)** |

Reporting only "cross-lingual performance is worse" would discard this.
→ OBSERVATIONS.md, "Three distinct cross-lingual failure modes"

**B2. An English-only model deletes every Turkish-specific character.**
Pegasus output contains 0.00% of ç/ğ/ı/ö/ş/ü against 10.55% in the source and 10.24% for the
extractive baseline. The content is often right; the script is destroyed. This inflates
"abstractiveness" to 34–41% — the novel n-grams are diacritic-stripped source words, not paraphrase.
→ OBSERVATIONS.md, "Three distinct cross-lingual failure modes"

**B3. Language mismatch costs compute, independent of quality.**
A 1,024-token window holds ~530 English words but only ~203 Turkish. DistilBART needs 15.5 chunks
per Turkish document against 5.8 per English, at *identical* per-chunk cost (1.2 s). The entire
2.7× slowdown is chunk count. **The languages most likely to need chunking are those whose
tokenizers force the most of it.**
→ OBSERVATIONS.md, "Language mismatch has a compute cost"

**B4. Multilinguality buys language-independence by being worse everywhere.**
mT5-XLSum shows almost no language effect (−0.5 points within-model, +4.4 against baseline) but
scores ≈22% in both languages — 16.1 points behind the English-only DistilBART on English. The
curse of multilinguality, measured directly. The strongest system on Turkish is the unsupervised
extractive baseline (41.0%), which has no training data to mismatch.
→ OBSERVATIONS.md, "Multilinguality buys language-independence"

**B5. Only the commercial multilingual LLM genuinely transfers.**
GPT-4o-mini shows no language penalty (+1.4 vs baseline) and is **the only neural system to beat
the extractive baseline on Turkish ROUGE-L** (14.6% vs 11.9%) — on the length-robust metric, not
the length-inflated composite. Open models either degrade sharply (DistilBART −6.4, LED −15.3) or
are uniformly weak (mT5 ≈22% in both languages). Turkish ROUGE-L ordering:
GPT-4o-mini 14.6 > TextRank 11.9 > mBART 7.3 > DistilBART 6.1 > mT5 5.8 > LED 2.0.
*Read with B6: Qwen's aggregate 7.5% belongs in that ordering only as a mixture — conditioned on
its Turkish-language output it reaches 13.9%, so "only the commercial LLM transfers" is a claim
about reliability, not about capability.*
→ OBSERVATIONS.md, "The commercial multilingual LLM is the only system that genuinely transfers"

**B6. Cross-lingual robustness is not an "LLM property" — but the mechanism is output-language
control, not degraded capability.** *(Revised at n=100 TR; the earlier reading was an aggregation
artifact. n=40 EN, in progress.)*
In aggregate Qwen2.5-7B looks exactly like a 306M English-only model on Turkish (26.7% composite,
7.5% ROUGE-L against DistilBART's 27.0% and 6.1%). Splitting by the **output** language of its 100
summaries shows why that average means nothing: it answered in Turkish 40 times, English 48 times
and **Chinese 12 times**. On the **39** documents where it stayed in Turkish it scores **14.3%** ROUGE-L
against GPT-4o-mini's 14.6% across all 100 — i.e. near the commercial system and roughly double
DistilBART's 6.1%. **GPT-4o-mini produced Turkish on 100/100 documents; Qwen on 39/100** (47
English, 14 Chinese; counts from script-aware classification, which corrected an earlier
`langdetect` undercount of the Chinese runs). The paired composite test confirms the split:
GPT-4o-mini and Qwen are **not separated on English** (0.5%, p = 0.73, d = 0.15) and **clearly
separated on Turkish** (9.4%, p = 0.009, d = 1.37) — the substitution claim is now tested, not
asserted. The
practical conclusion stands — an open model cannot be substituted for the commercial one outside
English — but the defect is instruction compliance, not capability, and a 60% rate of answering in
the wrong language is a worse failure than uniform mediocrity. Caveats: conditioning on output
language is post-hoc and selects the shorter documents (2,293 vs 3,115 words), on which every
system scores slightly higher; Qwen is 4-bit at 3,072-token context, so this is
model-plus-deployment.
→ OBSERVATIONS.md, "Correction at n=100: the Qwen result is an aggregation artifact"

**B7. Hierarchical chunking erodes output-language fidelity, and the erosion scales with chunk
count.** Qwen stayed in Turkish on 56% of 2-chunk documents, 38% at 3, 20% at 4 and 22% at 5+
(mean chunks 2.80 when compliant, 3.55 when drifted). Each pass is an opportunity to lose the
language constraint and the merge pass reads already-drifted text, so the failure compounds.
Chunking alone is not sufficient — DistilBART chunks 15.5× per Turkish document and drifts on
3/100, having no instruction to lose and a monolingual decoder to lose it into; LED produces
English on 96/100 by training rather than drift. **Drift requires a multilingual decoder plus an
instruction, with chunking as the eroding agent.** This couples RQ2 to RQ3: the language whose
tokenizer forces the most chunks (B3) pays a correctness penalty on top of the compute penalty.
The clean test — Qwen at full context versus Qwen chunked — was blocked by the T4 memory ceiling
(E1) and is the most valuable follow-up left open.
→ OBSERVATIONS.md, "Language drift is produced by chunking"

---

## C. Long context versus hierarchical chunking (research question 2)

**C1. Native long context is cheaper than chunking, not more expensive.**
LED-Large (16,384 tokens) processed 100/100 documents in a single pass at 7.0 s/document, faster
than BART-Large's 10.8 s across 5.8 chunks. The cost of chunking is the repeated beam searches and
the merge pass, not the reading.
→ OBSERVATIONS.md, "LED-Large: the long-context test"

**C2. The quality half of the question is confounded and is reported as unanswered.**
LED is trained on narrative book chapters and evaluated on scientific papers; its lower
faithfulness (84.3%) admits both a context-handling and a domain-fit explanation, which this
design cannot separate. Stated as future work rather than argued around.
→ same section; METHODOLOGY.md §8a

**C3. Chunk count is set by context window, not parameter count.**
Three 1,024-token models spanning 306M–568M parameters all need 5.8–5.9 chunks; mT5's 512-token
window needs 17.1. Within a fixed window, model size and decoding still differ (BART-Large 10.8 s
vs Pegasus 4.2 s at equal chunk counts), so both effects are needed.
→ OBSERVATIONS.md, "Chunking cost measured directly"

---

## D. System comparison (English, n=100)

**D1. On retention metrics extractive selection beats abstractive transformers by a wide margin —
partly by construction**, since those metrics reward preserving source vocabulary and TextRank
copies verbatim. Topic coverage 77.7% vs 48.8%; information density 57.9% vs 33.5%.

**D2. On agreement with the human abstract the systems are not clearly separated.**
ROUGE-L: TextRank 21.4% [20.1, 22.8] vs DistilBART 19.2% [18.2, 20.3] — a 2.2-point gap against
~2.4-point intervals. Reported as "not clearly separated", not as a ranking.

**D3. Hallucination does not differentiate systems on in-language English.**
Faithfulness ~99.6% for TextRank, DistilBART and BART-Large with a 0.1-point spread — a null
result, not insufficient power. It differentiates sharply under language mismatch.

**D4. n=100 per language is statistically sufficient.**
95% bootstrap CI widths 0.8–6.7 points (median 2.0) against between-system differences up to 28
points. Tripling the corpus would narrow intervals by √3 and change no conclusion.
→ OBSERVATIONS.md, "Interim results and the resolution of the sample-size question"

**D5. Separations are between paradigms, not within them.** At n=100 on English, five pairs
among the open encoder-decoders are null after Holm–Bonferroni: BART-Large vs DistilBART (0.4%,
p = 1.0 — **distillation costs nothing measurable**), BART-Large vs Flan-T5 (0.6%), DistilBART vs
Flan-T5 (0.1%), mT5-XLSum vs Pegasus (−0.7%), LED vs both BART variants. The significant gaps run
between extractive, LLM and encoder-decoder paradigms. Reported as null results, not as missing
power (D4).

**D6. TextRank's composite lead is significant and must still not be read as "extractive is
better."** p = 0.009 against all nine neural systems in both languages (+4.7 points over
GPT-4o-mini on English, +5.0 on Turkish). But the composite correlates with length at r = 0.69 and
its retention components reward preserving source vocabulary, which TextRank does by construction
(A3, D1). On ROUGE-L — length-robust, human-referenced — the English lead sits inside the interval
width (D2) and **on Turkish GPT-4o-mini leads TextRank 14.6% to 11.9%, reversing the ordering.**
The two metric families disagree — and **BERTScore, added afterwards, sides with ROUGE-L**
(Turkish: GPT-4o-mini 59.8% vs TextRank 56.5%), so the disagreement now resolves rather than
merely being reported (A8).

**D7. The faithfulness correction changed which comparisons are significant.** After the fix,
BART-Large and DistilBART fall significantly below both GPT-4o-mini (−2.5, −2.9 points) and Qwen
(−2.0, −2.5) on English, where the pre-rescore numbers had not separated them. The fix raised only
the paraphrasing systems (+28.6 pts for GPT-4o-mini, +24.1 for Qwen; zero for the copy-heavy
models), so the earlier numbers could not have supported these claims. **Every measurement defect
found in this project has biased results toward extractive and copy-heavy output.**
→ OBSERVATIONS.md, "What the faithfulness fix changed"

---

## E. Infrastructure and reproducibility

**E1. Eight models do not fit on one 16 GB T4 once working memory is counted.**
Seven encoder-decoders (306M–783M) coexist under lazy loading; adding a 7B model in 4-bit does
not. The server tracked 5.55 GB resident while the GPU held 9.18 GB — evicted models retained by
exception tracebacks. **Published VRAM figures describe weights, not working memory.**
→ OBSERVATIONS.md, "Operational failure: the 7B model"

**E2. Storing each summary as text, not only its score, made two metric corrections cost minutes
instead of GPU-hours.** `scripts/rescore.py` recomputes every metric from saved output without
invoking a model.

**E3. A reproducibility artefact that is rewritten rather than appended can be silently lost.**
The corpus snapshot was overwritten when the experiment was extended to a second language; caught
only because an analysis script crashed.
→ OBSERVATIONS.md, "Process defect found and fixed"

**E4. Resumability validated, not merely convenient.** 25 of 2,025 runs failed and **all 25 were
recovered** — no outstanding failures across the full 2,000 document × model cells, despite seven
GPU sessions over eight days and three mid-corpus interruptions. 18 failures were transport (dead
tunnel 404s, 503s, a connection reset), arriving in consecutive document-id blocks — the signature
of an expiring session, carrying no information about any model.

**E5. The OOM failures were length-selected, which is what makes them evidence for E1.** Only the
two longest English documents attempted (`en_arxiv_0001`, 5,207 words; `en_arxiv_0003`, 4,308)
ever ran out of memory, both at `chunks=None` after 0 s — dead before the first chunk. Both failed
identically on a **clean GPU** three days later, excluding the memory-leak explanation, then
completed at `chunks=3` in ~180 s once the context was cut from 8,192 to 3,072 tokens. The
understated 5.56 GB figure left ~5.3 GB of a 14.56 GB T4 for an 8k KV cache, and only the longest
documents tried to fill it.
→ OBSERVATIONS.md, "What the 20 failed runs show"

**E6. The E3 defect recurred on the sibling artifact, because the fix was applied where the bug
was found and not beside it.** `snapshot_corpus()` was changed to merge after E3; `write_manifest()`
was not. A one-model, one-language resume therefore replaced an eight-day, ten-model experiment's
provenance record with a description of itself — the report header read `Documents: 100 {'en': 100}`
for a 200-document study. **No data was lost and every table was correct**, which is what makes it
dangerous: a reproducibility artefact that silently *narrows* still looks complete, unlike one that
goes missing. Both fixed — the manifest now merges, appends each invocation to a `sessions` list,
and recomputes counts from the corpus snapshot on disk rather than from the current invocation's
arguments.
→ OBSERVATIONS.md, "E3 recurred, on the sibling artifact"

**E7. Runtime is chunk-bound within a model class and parameter-bound across classes.** Full run:
**16.4 hours** for 2,000 document x model cells. Among chunking encoder-decoders cost tracks chunk
count, not size — mT5-XLSum (582M, 512-token window, 17.3 passes) costs 39.1 s/document against
LED-Large (460M, 16,384 tokens, 1.1 passes) at 8.2 s, a 4.8x difference *inverse* to parameter
count, confirming C1 at full n. **Qwen is the exception that locates the mechanism**: 48% of all
compute at only 2.7 chunks/document, so its cost is 7.6B parameters of autoregressive decoding,
not chunking. A single "runtime is driven by context window" rule would be wrong for the largest
system in the study.
→ OBSERVATIONS.md, "Runtime: the efficiency result, with final totals"
