/* ===========================================================================
   Experiment tab: multi-document, multi-model evaluation with aggregate
   statistics. Streams progress from /experiment/run and renders per-language
   result tables, confidence intervals and pairwise significance tests.
   =========================================================================== */

(function () {
    "use strict";

    const { fmt, ciRows, barRows } = window.Charts;

    const state = {
        loaded: false,
        documents: [],
        selectedDocuments: new Set(),
        customDocuments: [],
        models: [],              // {key, name, type, languages:[]}
        selectedModels: new Set(),
        running: false,
        aborted: false,
    };

    // Metrics shown in the aggregate table, in order.
    const TABLE_METRICS = [
        { key: "overall_quality", label: "Composite", format: fmt.pct, direction: "higher" },
        { key: "rouge_l_f1", label: "ROUGE-L", format: fmt.pct, direction: "higher" },
        { key: "rouge_1_f1", label: "ROUGE-1", format: fmt.pct, direction: "higher" },
        { key: "meteor", label: "METEOR", format: fmt.pct, direction: "higher" },
        { key: "information_density", label: "Density", format: fmt.pct, direction: "higher" },
        { key: "topic_coverage", label: "Topics", format: fmt.pct, direction: "higher" },
        { key: "source_coverage", label: "Coverage", format: fmt.pct, direction: "higher" },
        { key: "factual_consistency", label: "Faithful", format: fmt.pct, direction: "higher" },
        { key: "redundancy", label: "Redundancy", format: fmt.pct, direction: "lower" },
        { key: "abstractiveness", label: "Novel words", format: fmt.pct, direction: "descriptive" },
        { key: "compression_ratio", label: "Compression", format: fmt.pct, direction: "lower" },
        { key: "runtime_seconds", label: "Runtime", format: fmt.sec, direction: "lower" },
    ];

    const A = () => window.App;

    /* ------------------------------------------------------------------ */
    /* Loading the pickers                                                 */
    /* ------------------------------------------------------------------ */

    async function ensureLoaded() {
        if (state.loaded) return;
        state.loaded = true;
        const { api, el, $, escapeHtml, modelColor, MODEL_TYPE_LABELS } = A();

        // --- Documents ---
        try {
            const data = await api("/experiment/documents");
            state.documents = data.documents;
            data.documents.forEach((doc) => state.selectedDocuments.add(doc.key));
            renderDocuments();
        } catch (error) {
            $("experiment-documents").innerHTML =
                `<div class="callout is-bad">Could not load documents: ${escapeHtml(error.message)}</div>`;
        }

        // --- Models: union over the languages present in the document set ---
        try {
            const languages = Array.from(new Set(state.documents.map((d) => d.language)));
            const perLanguage = await Promise.all(
                languages.map((lang) => api(`/models?lang=${lang}`).then((r) => ({ lang, models: r.models })))
            );
            const merged = new Map();
            perLanguage.forEach(({ lang, models }) => {
                models.forEach((model) => {
                    if (!merged.has(model.key)) merged.set(model.key, { ...model, languages: [] });
                    merged.get(model.key).languages.push(lang);
                });
            });
            state.models = Array.from(merged.values());
            // Paid models are opt-in: a run spends credit once per document.
            state.models.forEach((model) => {
                if (model.type !== "paid") state.selectedModels.add(model.key);
            });
            renderModels();
        } catch (error) {
            $("experiment-models").innerHTML =
                `<div class="callout is-bad">Could not load models: ${escapeHtml(error.message)}</div>`;
        }

        wireControls();
        updateEstimate();
    }

    function renderDocuments() {
        const { el, $, escapeHtml } = A();
        const host = $("experiment-documents");
        host.innerHTML = "";
        state.documents.forEach((doc, index) => {
            const label = el("label", "doc-item");
            label.style.setProperty("--i", index);
            label.innerHTML =
                `<input type="checkbox" ${state.selectedDocuments.has(doc.key) ? "checked" : ""}>` +
                `<span><span class="doc-title">${escapeHtml(doc.title)}` +
                `<span class="doc-flag ${doc.has_reference ? "has-ref" : ""}">` +
                `${doc.has_reference ? "reference" : "no reference"}</span></span>` +
                `<span class="doc-sub">${doc.language.toUpperCase()} · ${doc.word_count} words · ${escapeHtml(doc.category)}</span></span>`;
            label.querySelector("input").addEventListener("change", (event) => {
                if (event.target.checked) state.selectedDocuments.add(doc.key);
                else state.selectedDocuments.delete(doc.key);
                updateEstimate();
            });
            host.appendChild(label);
        });
    }

    function renderModels() {
        const { el, $, escapeHtml, modelColor, MODEL_TYPE_LABELS } = A();
        const host = $("experiment-models");
        host.innerHTML = "";
        state.models.forEach((model, index) => {
            const label = el("label", `doc-item${model.type === "paid" ? " is-cost" : ""}`);
            label.style.setProperty("--i", index);
            label.innerHTML =
                `<input type="checkbox" ${state.selectedModels.has(model.key) ? "checked" : ""}>` +
                `<span><span class="doc-title">` +
                `<span class="model-dot" style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${modelColor(model.key)};margin-right:7px"></span>` +
                `${escapeHtml(model.name)}` +
                (model.type === "paid" ? `<span class="doc-flag">costs credit</span>` : "") +
                `</span>` +
                `<span class="doc-sub">${escapeHtml(MODEL_TYPE_LABELS[model.type] || model.type)} · ` +
                `${model.languages.map((l) => l.toUpperCase()).join(", ")}</span></span>`;
            label.querySelector("input").addEventListener("change", (event) => {
                if (event.target.checked) state.selectedModels.add(model.key);
                else state.selectedModels.delete(model.key);
                updateEstimate();
            });
            host.appendChild(label);
        });
    }

    function renderCustomDocuments() {
        const { el, $, escapeHtml } = A();
        const host = $("custom-doc-list");
        host.innerHTML = "";
        state.customDocuments.forEach((doc, index) => {
            const item = el("li");
            item.innerHTML =
                `<span>${escapeHtml(doc.title)} · ${doc.text.split(/\s+/).length} words` +
                `${doc.reference_summary ? " · reference" : ""}</span>`;
            const remove = el("button", "", "×");
            remove.title = "Remove";
            remove.addEventListener("click", () => {
                state.customDocuments.splice(index, 1);
                renderCustomDocuments();
                updateEstimate();
            });
            item.appendChild(remove);
            host.appendChild(item);
        });
    }

    function selectedDocumentCount() {
        return state.selectedDocuments.size + state.customDocuments.length;
    }

    function updateEstimate() {
        const { $ } = A();
        const docs = selectedDocumentCount();
        const models = state.selectedModels.size;
        const paid = state.models.filter((m) => m.type === "paid" && state.selectedModels.has(m.key)).length;

        // Per-language counts, since a model only runs on its own languages.
        const languages = {};
        state.documents
            .filter((d) => state.selectedDocuments.has(d.key))
            .forEach((d) => (languages[d.language] = (languages[d.language] || 0) + 1));
        state.customDocuments.forEach((d) => {
            const lang = d.language || "en";
            languages[lang] = (languages[lang] || 0) + 1;
        });

        const perLanguage = Object.entries(languages)
            .map(([lang, count]) => {
                const applicable = state.models.filter(
                    (m) => state.selectedModels.has(m.key) && m.languages.includes(lang)
                ).length;
                return { lang, count, applicable, runs: count * applicable };
            });
        const totalRuns = perLanguage.reduce((sum, entry) => sum + entry.runs, 0);
        const smallest = Math.min(...Object.values(languages).concat([Infinity]));

        const detail = perLanguage
            .map((e) => `${e.lang.toUpperCase()}: ${e.count} docs × ${e.applicable} models = ${e.runs} runs`)
            .join("<br>");

        $("run-estimate").innerHTML =
            `<strong>${totalRuns}</strong> model runs over <strong>${docs}</strong> documents.<br>` +
            (detail ? detail + "<br>" : "") +
            (paid ? `<span style="color:var(--warn)">${paid} paid model(s) selected — each run costs credit.</span><br>` : "") +
            (smallest < 5
                ? `<span style="color:var(--warn)">Fewer than 5 documents per language: means and intervals are reported, p-values are not.</span>`
                : `Significance tests will be reported.`);

        $("btn-run-experiment").disabled = totalRuns === 0 || state.running;
    }

    function wireControls() {
        const { $, toast } = A();

        $("btn-add-doc").addEventListener("click", () => {
            const text = $("custom-doc-text").value.trim();
            if (text.length < 50) {
                toast("The document needs at least 50 characters.", "bad");
                return;
            }
            state.customDocuments.push({
                title: $("custom-doc-title").value.trim() || `Custom ${state.customDocuments.length + 1}`,
                text,
                reference_summary: $("custom-doc-ref").value.trim() || null,
            });
            $("custom-doc-title").value = "";
            $("custom-doc-text").value = "";
            $("custom-doc-ref").value = "";
            renderCustomDocuments();
            updateEstimate();
            toast("Document added to the experiment set.", "good");
        });

        $("btn-run-experiment").addEventListener("click", runExperiment);
        $("btn-cancel-experiment").addEventListener("click", () => {
            state.aborted = true;
            A().toast("Stopping after the current model finishes…");
        });
    }

    /* ------------------------------------------------------------------ */
    /* Running                                                             */
    /* ------------------------------------------------------------------ */

    async function runExperiment() {
        const { $, el, escapeHtml, streamEvents, toast, ensureMetricMeta } = A();
        await ensureMetricMeta();

        state.running = true;
        state.aborted = false;
        $("btn-run-experiment").disabled = true;
        $("btn-cancel-experiment").classList.remove("hidden");
        $("exp-progress").classList.remove("hidden");
        $("exp-log").innerHTML = "";
        $("experiment-results").classList.add("hidden");
        $("experiment-results").innerHTML = "";

        const setProgress = (done, total, label) => {
            const percent = total ? (done / total) * 100 : 0;
            $("exp-progress-fill").style.width = `${percent}%`;
            $("exp-progress-pct").textContent = `${Math.round(percent)}%`;
            if (label) $("exp-progress-label").textContent = label;
        };
        setProgress(0, 1, "Planning the run…");

        const log = (docKey, modelName, text, bad) => {
            const row = el("div", bad ? "log-bad" : "");
            row.innerHTML =
                `<span class="log-doc">${escapeHtml(docKey)}</span>` +
                `<span class="log-model">${escapeHtml(modelName)}</span>` +
                `<span>${escapeHtml(text)}</span>`;
            $("exp-log").appendChild(row);
            $("exp-log").scrollTop = $("exp-log").scrollHeight;
        };

        try {
            await streamEvents(
                "/experiment/run",
                {
                    document_keys: Array.from(state.selectedDocuments),
                    model_keys: Array.from(state.selectedModels),
                    ratio: parseFloat($("exp-ratio").value),
                    extra_documents: state.customDocuments,
                },
                (event) => {
                    const data = event.data || {};
                    if (event.stage === "plan") {
                        setProgress(0, data.total_runs, `0 of ${data.total_runs} runs`);
                        log("—", "plan", `${data.documents.length} documents, ${data.total_runs} runs`);
                    } else if (event.stage === "result") {
                        const { completed, total } = data.progress;
                        setProgress(completed, total, `${completed} of ${total} runs`);
                        log(
                            data.document,
                            data.model_name,
                            `quality ${fmt.pct(data.scores.overall_quality)} · ` +
                            `faithful ${fmt.pct(data.scores.factual_consistency)} · ${fmt.sec(data.runtime_seconds)}`
                        );
                    } else if (event.stage === "error") {
                        if (data.progress) setProgress(data.progress.completed, data.progress.total);
                        log(data.document || "—", data.model_name || "error", data.message, true);
                    } else if (event.stage === "aggregate") {
                        renderAggregate(data);
                    } else if (event.stage === "done") {
                        setProgress(1, 1, "Complete");
                        const failures = (data.errors || []).length;
                        toast(
                            `Experiment finished${failures ? ` with ${failures} failed run(s)` : ""}.`,
                            failures ? "warn" : "good"
                        );
                    }
                }
            );
        } catch (error) {
            toast(`Experiment failed: ${error.message}`, "bad");
        } finally {
            state.running = false;
            $("btn-cancel-experiment").classList.add("hidden");
            updateEstimate();
            setTimeout(() => $("exp-progress").classList.add("hidden"), 1200);
        }
    }

    /* ------------------------------------------------------------------ */
    /* Rendering the aggregate                                             */
    /* ------------------------------------------------------------------ */

    function renderAggregate(aggregate) {
        const { $, el, escapeHtml, modelColor, infoTip } = A();
        const host = $("experiment-results");
        host.classList.remove("hidden");
        host.innerHTML = "";

        const languages = Object.entries(aggregate.by_language || {});
        if (!languages.length) {
            host.innerHTML = `<div class="callout is-bad">No results were produced.</div>`;
            return;
        }

        // Export + method note
        const header = el("div", "card");
        header.innerHTML =
            `<div class="card-head"><h3>Results</h3>` +
            `<a class="btn btn-small" style="width:auto;text-decoration:none" href="${A().API}/experiment/last.csv" download>Download CSV</a>` +
            `</div>` +
            `<p class="hint">${escapeHtml(aggregate.note || "")}</p>`;
        host.appendChild(header);

        languages.forEach(([lang, block]) => {
            host.appendChild(
                el("h3", "section-title", `${lang.toUpperCase()} — ${block.n_documents} document(s)`)
            );

            const docsLine = block.documents
                .map((d) => `${escapeHtml(d.title)} (${d.words}w)`)
                .join(" · ");
            const anyMetric = Object.values(block.metrics)[0];
            const tested = anyMetric ? anyMetric.significance_tested : false;
            host.appendChild(
                el(
                    "div",
                    tested ? "callout" : "callout is-warn",
                    `<strong>Document set:</strong> ${docsLine}. ` +
                    `${block.references_available} of ${block.n_documents} have a reference summary. ` +
                    `<strong>Models compared:</strong> ${block.models_compared.length}` +
                    (block.models_excluded.length
                        ? ` (excluded for incomplete coverage: ${block.models_excluded.map(escapeHtml).join(", ")})`
                        : "") +
                    `.<br>${escapeHtml(anyMetric ? anyMetric.caveat : "")}`
                )
            );

            // --- Main table: mean [CI] per model per metric ---
            const table = el("table", "data");
            const metrics = TABLE_METRICS.filter((m) => block.metrics[m.key]);
            table.innerHTML =
                `<thead><tr><th style="text-align:left">Model</th>` +
                metrics
                    .map(
                        (m) =>
                            `<th title="${m.direction === "lower" ? "lower is better" : m.direction === "higher" ? "higher is better" : "descriptive"}">` +
                            `${escapeHtml(m.label)}${m.direction === "lower" ? " ↓" : m.direction === "higher" ? " ↑" : ""}</th>`
                    )
                    .join("") +
                `</tr></thead>`;

            const tbody = el("tbody");
            block.models_compared.forEach((model, index) => {
                const row = el("tr");
                row.style.setProperty("--i", index);
                let cells =
                    `<td><span class="model-cell">` +
                    `<span class="model-dot" style="background:${modelColor(model)}"></span>` +
                    `${escapeHtml(block.model_names[model] || model)}</span></td>`;
                metrics.forEach((metric) => {
                    const stats = block.metrics[metric.key].per_system[model];
                    if (!stats || stats.mean === null) {
                        cells += `<td class="num">—</td>`;
                        return;
                    }
                    const best = isBest(block.metrics[metric.key], model, metric.direction);
                    const ci =
                        stats.n > 1 && stats.ci_low !== stats.ci_high
                            ? `<br><span style="color:var(--text-faint);font-size:0.9em">[${metric.format(stats.ci_low)}, ${metric.format(stats.ci_high)}]</span>`
                            : "";
                    cells += `<td class="num${best ? " best" : ""}">${metric.format(stats.mean)}${ci}</td>`;
                });
                row.innerHTML = cells;
                tbody.appendChild(row);
            });
            table.appendChild(tbody);
            const wrap = el("div", "table-wrap");
            wrap.appendChild(table);
            host.appendChild(wrap);
            host.appendChild(
                el(
                    "div",
                    "section-note",
                    "Cells show the mean across documents with its 95% bootstrap confidence interval. " +
                    "Green marks the best value for metrics that have a direction; descriptive metrics are never marked."
                )
            );

            // --- Confidence-interval charts for the headline metrics ---
            const ciGrid = el("div", "metrics-grid");
            ["overall_quality", "rouge_l_f1", "factual_consistency", "information_density"].forEach(
                (metricKey, index) => {
                    const stats = block.metrics[metricKey];
                    if (!stats) return;
                    const metric = TABLE_METRICS.find((m) => m.key === metricKey);
                    // Error bars need horizontal room: with more than three
                    // systems a narrow card truncates the labels.
                    const wide = block.models_compared.length > 3;
                    const card = el("div", `metric-card${wide ? " is-wide" : ""}`);
                    card.style.setProperty("--i", index);
                    card.innerHTML =
                        `<div class="metric-head"><span class="metric-title">${escapeHtml(metric.label)} — mean and 95% CI</span>` +
                        `${infoTip(metricKey === "rouge_l_f1" ? "rouge" : metricKey)}</div>`;
                    const chartHost = el("div");
                    card.appendChild(chartHost);
                    ciRows(
                        chartHost,
                        block.models_compared
                            .map((model) => {
                                const s = stats.per_system[model];
                                return s && s.mean !== null
                                    ? {
                                        label: block.model_names[model] || model,
                                        color: modelColor(model),
                                        mean: s.mean,
                                        low: s.ci_low,
                                        high: s.ci_high,
                                    }
                                    : null;
                            })
                            .filter(Boolean),
                        { format: metric.format }
                    );
                    ciGrid.appendChild(card);
                }
            );
            host.appendChild(ciGrid);

            // --- Pairwise significance ---
            const pairwiseMetrics = Object.entries(block.metrics).filter(
                ([, stats]) => Object.keys(stats.pairwise || {}).length
            );
            if (pairwiseMetrics.length) {
                host.appendChild(el("h4", "section-title", "Pairwise comparisons"));
                pairwiseMetrics.forEach(([metricKey, stats]) => {
                    const metric = TABLE_METRICS.find((m) => m.key === metricKey) || { label: metricKey, format: fmt.pct };
                    const card = el("div", "card");
                    card.style.marginBottom = "12px";
                    card.innerHTML = `<div class="card-head"><h3>${escapeHtml(metric.label)}</h3></div>`;

                    const table = el("table", "data");
                    const showP = stats.significance_tested;
                    table.innerHTML =
                        `<thead><tr><th style="text-align:left">Comparison</th><th>Difference</th>` +
                        `<th>95% CI of difference</th>` +
                        (showP ? `<th>p (bootstrap)</th><th>p (Holm)</th><th>Wilcoxon p</th><th>Significant</th>` : "") +
                        `<th>Cohen's d</th><th>Effect</th></tr></thead>`;
                    const tbody = el("tbody");
                    Object.entries(stats.pairwise).forEach(([pair, result], index) => {
                        const row = el("tr");
                        row.style.setProperty("--i", index);
                        const nameA = block.model_names[result.system_a] || result.system_a;
                        const nameB = block.model_names[result.system_b] || result.system_b;
                        const ci =
                            result.diff_ci_low !== undefined && result.diff_ci_low !== null
                                ? `[${metric.format(result.diff_ci_low)}, ${metric.format(result.diff_ci_high)}]`
                                : "—";
                        let cells =
                            `<td style="text-align:left">${escapeHtml(nameA)} <span style="color:var(--text-faint)">vs</span> ${escapeHtml(nameB)}</td>` +
                            `<td class="num">${metric.format(result.mean_difference)}</td>` +
                            `<td class="num">${ci}</td>`;
                        if (showP) {
                            const wilcoxon = result.wilcoxon || {};
                            cells +=
                                `<td class="num">${result.bootstrap_p ?? "—"}</td>` +
                                `<td class="num">${result.p_adjusted ?? "—"}</td>` +
                                `<td class="num">${wilcoxon.skipped ? "—" : wilcoxon.p_value}</td>` +
                                `<td class="${result.significant ? "sig-yes" : "sig-no"}">${result.significant ? "yes" : "no"}</td>`;
                        }
                        cells +=
                            `<td class="num">${result.cohens_d ?? "—"}</td>` +
                            `<td>${escapeHtml(result.effect_size || "—")}</td>`;
                        row.innerHTML = cells;
                        tbody.appendChild(row);
                    });
                    table.appendChild(tbody);
                    const tableWrap = el("div", "table-wrap");
                    tableWrap.appendChild(table);
                    card.appendChild(tableWrap);
                    if (!showP) {
                        card.appendChild(
                            el("div", "section-note",
                                "Differences are descriptive: with fewer than five documents no p-value is reported.")
                        );
                    }
                    host.appendChild(card);
                });
            }

            // --- Runtime comparison ---
            if (block.metrics.runtime_seconds) {
                const card = el("div", "metric-card");
                card.innerHTML =
                    `<div class="metric-head"><span class="metric-title">Mean runtime per document</span>` +
                    `<span class="metric-dir">lower = better</span></div>`;
                const chartHost = el("div");
                card.appendChild(chartHost);
                barRows(
                    chartHost,
                    block.models_compared.map((model) => ({
                        label: block.model_names[model] || model,
                        value: block.metrics.runtime_seconds.per_system[model]?.mean ?? 0,
                        color: modelColor(model),
                        striped: true,
                    })),
                    { format: fmt.sec }
                );
                host.appendChild(card);
            }
        });
    }

    /** Whether this model holds the best mean for a metric with a direction. */
    function isBest(metricStats, model, direction) {
        if (direction !== "higher" && direction !== "lower") return false;
        const values = Object.entries(metricStats.per_system)
            .map(([key, stats]) => ({ key, mean: stats.mean }))
            .filter((entry) => entry.mean !== null && entry.mean !== undefined);
        if (values.length < 2) return false;
        const best = direction === "higher"
            ? Math.max(...values.map((v) => v.mean))
            : Math.min(...values.map((v) => v.mean));
        const own = metricStats.per_system[model];
        return own && own.mean === best;
    }

    window.Experiment = { ensureLoaded };
})();
