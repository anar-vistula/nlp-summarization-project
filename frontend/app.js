/* ===========================================================================
   Long Document Summarization — evaluation platform (frontend core)

   Responsibilities: theme, tabs, backend status, sample loading, the three
   single-document run modes, the multi-model comparison, and the methodology
   table. The experiment tab lives in experiment.js and reuses the helpers
   exported on `window.App`.
   =========================================================================== */

(function () {
    "use strict";

    const API = window.location.port === "8000" ? "" : "http://localhost:8000";
    const { fmt, barRows, radar, scatter, countUp } = window.Charts;

    /* ------------------------------------------------------------------ */
    /* Model identity                                                      */
    /* ------------------------------------------------------------------ */

    const MODEL_COLORS = {
        extractive: "#12a06a",
        extractive_textrank: "#12a06a",
        distilbart: "#e0761f",
        abstractive_transformer: "#e0761f",
        "bart-large": "#e04a3f",
        pegasus: "#9b5de5",
        "mt5-xlsum": "#2f80ed",
        "mt5-turkish-finetuned": "#0f9b8e",
        "led-large-book": "#e5397f",
        "flan-t5-large": "#00a7c4",
        "turkish-mbart": "#f4623a",
        "qwen-llm": "#6d5bd0",
        "gpt-4": "#1f6f54",
        "gpt-4o-mini": "#2f6f9f",
        "claude-sonnet": "#c08a4a",
        "claude-haiku": "#8c6fd0",
    };

    const MODEL_TYPE_LABELS = {
        extractive: "Extractive · graph-based",
        hf: "Abstractive · local transformer",
        colab: "Abstractive · Colab GPU",
        paid: "Abstractive · paid API",
    };

    const modelColor = (key) => MODEL_COLORS[key] || "#7a8699";

    /* ------------------------------------------------------------------ */
    /* Small DOM helpers                                                   */
    /* ------------------------------------------------------------------ */

    const $ = (id) => document.getElementById(id);
    const el = (tag, className, html) => {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (html !== undefined) node.innerHTML = html;
        return node;
    };
    const escapeHtml = (text) =>
        String(text ?? "").replace(/[&<>"']/g, (c) =>
            ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
        );

    /** Append a child unless the builder returned null (nothing to show). */
    const appendCard = (parent, child) => { if (child) parent.appendChild(child); };

    function toast(message, kind = "") {
        const stack = $("toast-stack");
        const node = el("div", `toast ${kind ? "is-" + kind : ""}`, escapeHtml(message));
        stack.appendChild(node);
        setTimeout(() => {
            node.classList.add("is-leaving");
            setTimeout(() => node.remove(), 300);
        }, 5200);
    }

    async function api(path, options = {}) {
        const response = await fetch(`${API}${path}`, options);
        if (!response.ok) {
            let detail = `${response.status} ${response.statusText}`;
            try {
                const body = await response.json();
                if (body.detail) detail = body.detail;
            } catch (_) { /* response was not JSON */ }
            throw new Error(detail);
        }
        return response.json();
    }

    /** POST that consumes a Server-Sent Events stream, one callback per event. */
    async function streamEvents(path, body, onEvent) {
        const response = await fetch(`${API}${path}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const chunks = buffer.split("\n\n");
            buffer = chunks.pop();
            for (const chunk of chunks) {
                const line = chunk.trim();
                if (!line.startsWith("data:")) continue;
                try {
                    onEvent(JSON.parse(line.slice(5).trim()));
                } catch (error) {
                    console.error("Bad SSE payload", error, line);
                }
            }
        }
    }

    /* ------------------------------------------------------------------ */
    /* Theme                                                               */
    /* ------------------------------------------------------------------ */

    function initTheme() {
        const stored = (() => {
            try { return localStorage.getItem("theme"); } catch (_) { return null; }
        })();
        const system = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
        document.documentElement.dataset.theme = stored || system;

        $("theme-toggle").addEventListener("click", () => {
            const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
            document.documentElement.dataset.theme = next;
            try { localStorage.setItem("theme", next); } catch (_) { /* private mode */ }
        });
    }

    /* ------------------------------------------------------------------ */
    /* Tabs                                                                */
    /* ------------------------------------------------------------------ */

    function initTabs() {
        const tabs = Array.from(document.querySelectorAll(".tab"));
        const indicator = $("tab-indicator");

        function moveIndicator(tab) {
            indicator.style.width = `${tab.offsetWidth}px`;
            indicator.style.transform = `translateX(${tab.offsetLeft}px)`;
        }

        function activate(name) {
            tabs.forEach((tab) => {
                const active = tab.dataset.tab === name;
                tab.classList.toggle("is-active", active);
                if (active) moveIndicator(tab);
            });
            document.querySelectorAll(".panel").forEach((panel) => {
                panel.classList.toggle("is-active", panel.id === `panel-${name}`);
            });
            if (name === "methodology") loadMethodology();
            if (name === "experiment" && window.Experiment) window.Experiment.ensureLoaded();
        }

        tabs.forEach((tab) => tab.addEventListener("click", () => activate(tab.dataset.tab)));
        document.querySelectorAll("[data-goto]").forEach((button) =>
            button.addEventListener("click", () => activate(button.dataset.goto))
        );
        requestAnimationFrame(() => moveIndicator(tabs[0]));
        window.addEventListener("resize", () => {
            const active = tabs.find((t) => t.classList.contains("is-active"));
            if (active) moveIndicator(active);
        });
        return { activate };
    }

    /* ------------------------------------------------------------------ */
    /* Backend status                                                      */
    /* ------------------------------------------------------------------ */

    async function refreshStatus() {
        const apiChip = $("status-api");
        const setChip = (chip, state, label) => {
            chip.className = `status-chip is-${state}`;
            chip.querySelector(".status-label").textContent = label;
        };
        try {
            await api("/health");
            setChip(apiChip, "ok", "backend online");
        } catch (error) {
            setChip(apiChip, "bad", "backend offline");
            return;
        }
        try {
            const status = await api("/colab/status");
            const gpuChip = $("status-gpu");
            if (status.connected) {
                setChip(gpuChip, "ok", `GPU · ${status.models.length} models`);
            } else {
                setChip(gpuChip, "off", "GPU not connected");
                gpuChip.title = status.reason || "Colab GPU server unavailable";
            }
        } catch (_) {
            setChip($("status-gpu"), "off", "GPU not connected");
        }
    }

    async function refreshModelAvailability() {
        const container = $("model-availability");
        const text = $("input-text").value.trim();
        const lang = text.length > 40 ? undefined : "en";
        try {
            const cross = $("cross-lingual")?.checked ?? false;
        const data = await api(`/models?lang=${lang || "en"}&cross_lingual=${cross}`);
            container.innerHTML = "";
            data.models.forEach((model, index) => {
                const chip = el("span", "avail-chip");
                chip.style.setProperty("--i", index);
                chip.innerHTML =
                    `<span class="model-dot" style="background:${modelColor(model.key)}"></span>` +
                    `${escapeHtml(model.name)}`;
                chip.title = MODEL_TYPE_LABELS[model.type] || model.type;
                container.appendChild(chip);
            });
            $("compare-all-sub").textContent = `${data.models.length} models available (${data.language})`;
        } catch (_) { /* status chip already reports backend problems */ }
    }

    /* ------------------------------------------------------------------ */
    /* Samples and input                                                   */
    /* ------------------------------------------------------------------ */

    async function initSamples() {
        const container = $("sample-chips");
        try {
            const data = await api("/samples");
            container.innerHTML = "";
            Object.entries(data.samples).forEach(([key, sample], index) => {
                const chip = el("button", "chip");
                chip.style.animationDelay = `${index * 50}ms`;
                chip.innerHTML =
                    `${escapeHtml(sample.title)}<span class="chip-lang">${sample.language.toUpperCase()}</span>`;
                chip.title = sample.category;
                chip.addEventListener("click", async () => {
                    document.querySelectorAll("#sample-chips .chip").forEach((c) => c.classList.remove("is-active"));
                    chip.classList.add("is-active");
                    try {
                        const full = await api(`/samples/${key}`);
                        $("input-text").value = full.text;
                        $("reference-summary").value = full.reference_summary || "";
                        updateInputMeter();
                        toast(`Loaded “${sample.title}” (${full.text.split(/\s+/).length} words)`, "good");
                    } catch (error) {
                        toast(`Could not load sample: ${error.message}`, "bad");
                    }
                });
                container.appendChild(chip);
            });
        } catch (error) {
            container.innerHTML = `<span class="hint">Samples unavailable (${escapeHtml(error.message)})</span>`;
        }
    }

    function updateInputMeter() {
        const text = $("input-text").value;
        const words = text.trim() ? text.trim().split(/\s+/).length : 0;
        $("doc-words").textContent = `${words.toLocaleString()} words`;
        $("doc-chars").textContent = `${text.length.toLocaleString()} characters`;

        const reference = $("reference-summary").value.trim();
        const refState = $("ref-state");
        refState.textContent = reference ? `${reference.split(/\s+/).length} words` : "not provided";
        refState.classList.toggle("is-set", Boolean(reference));
    }

    function initInputs() {
        $("input-text").addEventListener("input", updateInputMeter);
        $("reference-summary").addEventListener("input", updateInputMeter);

        const syncSlider = (slider, output, hint) => {
            const value = parseFloat(slider.value);
            slider.style.setProperty("--fill", `${((value - 0.1) / 0.8) * 100}%`);
            output.textContent = value.toFixed(2);
            if (hint) hint.textContent = `Summary length ≈ ${Math.round(value * 100)}% of the source.`;
        };
        const ratio = $("ratio");
        ratio.addEventListener("input", () => syncSlider(ratio, $("ratio-value"), $("ratio-hint")));
        syncSlider(ratio, $("ratio-value"), $("ratio-hint"));

        const expRatio = $("exp-ratio");
        if (expRatio) {
            expRatio.addEventListener("input", () => syncSlider(expRatio, $("exp-ratio-value")));
            syncSlider(expRatio, $("exp-ratio-value"));
        }

        // Click ripple, purely decorative.
        document.addEventListener("pointerdown", (event) => {
            const button = event.target.closest(".btn");
            if (!button) return;
            const rect = button.getBoundingClientRect();
            button.style.setProperty("--rx", `${((event.clientX - rect.left) / rect.width) * 100}%`);
            button.style.setProperty("--ry", `${((event.clientY - rect.top) / rect.height) * 100}%`);
            button.classList.add("is-rippling");
            setTimeout(() => button.classList.remove("is-rippling"), 60);
        });
    }

    function readRequest() {
        const text = $("input-text").value.trim();
        if (text.length < 50) {
            toast("Enter at least 50 characters of text, or load a sample.", "bad");
            return null;
        }
        const reference = $("reference-summary").value.trim();
        return {
            text,
            ratio: parseFloat($("ratio").value),
            reference_summary: reference || null,
            cross_lingual: $("cross-lingual")?.checked ?? false,
        };
    }

    /* ------------------------------------------------------------------ */
    /* Progress UI                                                         */
    /* ------------------------------------------------------------------ */

    const progress = {
        show(stages) {
            $("progress-block").classList.remove("hidden");
            const list = $("stage-list");
            list.innerHTML = "";
            stages.forEach((stage, index) => {
                const item = el("li", "", `<span class="stage-dot"></span><span>${escapeHtml(stage.label)}</span>`);
                item.dataset.stage = stage.key;
                item.style.animationDelay = `${index * 60}ms`;
                list.appendChild(item);
            });
            this.set(0, "Starting…");
        },
        set(percent, label) {
            $("progress-fill").style.width = `${percent}%`;
            $("progress-pct").textContent = `${Math.round(percent)}%`;
            if (label) $("progress-label").textContent = label;
        },
        stage(key, state) {
            const item = document.querySelector(`#stage-list li[data-stage="${key}"]`);
            if (!item) return;
            item.classList.remove("is-running", "is-done");
            if (state) item.classList.add(`is-${state}`);
        },
        hide() {
            setTimeout(() => $("progress-block").classList.add("hidden"), 700);
        },
    };

    function setBusy(busy) {
        ["btn-stream", "btn-extractive", "btn-abstractive", "btn-compare-all"].forEach((id) => {
            $(id).disabled = busy;
        });
    }

    /* ------------------------------------------------------------------ */
    /* Metric presentation helpers                                         */
    /* ------------------------------------------------------------------ */

    let METRIC_META = null;

    async function ensureMetricMeta() {
        if (METRIC_META) return METRIC_META;
        try {
            const data = await api("/metrics/metadata");
            METRIC_META = data.metrics;
        } catch (_) {
            METRIC_META = {};
        }
        return METRIC_META;
    }

    function infoTip(metaKey) {
        const meta = (METRIC_META || {})[metaKey];
        if (!meta) return "";
        const body = `<strong>${escapeHtml(meta.label)}</strong> — ${escapeHtml(meta.measures)}` +
            `<br><br>Source: ${escapeHtml(meta.source)}` +
            `<br>Caveat: ${escapeHtml(meta.caveat)}`;
        return `<span class="info-tip" tabindex="0">i<span class="tip-body">${body}</span></span>`;
    }

    function directionTag(metaKey) {
        const meta = (METRIC_META || {})[metaKey];
        if (!meta) return "";
        const label = meta.direction === "higher" ? "higher = better"
            : meta.direction === "lower" ? "lower = better"
            : "descriptive";
        return `<span class="metric-dir">${label}</span>`;
    }

    /**
     * One metric card comparing all models on a single metric.
     * accessor returns a number (or null) from a metrics object.
     */
    function metricCard(models, { title, metaKey, accessor, format, index, striped, max, note }) {
        const items = models
            .map((model) => ({
                label: model.shortName || model.name,
                value: accessor(model.metrics || {}),
                color: modelColor(model.key),
                striped,
            }))
            .filter((item) => item.value !== null && item.value !== undefined);
        // A card with nothing to plot (e.g. runtime that was never measured)
        // is noise, not information.
        if (!items.length) return null;

        const card = el("div", "metric-card");
        card.style.setProperty("--i", index || 0);
        const meta = (METRIC_META || {})[metaKey] || {};
        card.innerHTML =
            `<div class="metric-head"><span class="metric-title">${escapeHtml(title)}</span>` +
            `${directionTag(metaKey)}${infoTip(metaKey)}</div>` +
            (meta.source && meta.source !== "—" ? `<div class="metric-source">${escapeHtml(meta.source)}</div>` : "");
        const chart = el("div");
        card.appendChild(chart);
        if (note) card.appendChild(el("div", "metric-caveat", escapeHtml(note)));

        barRows(chart, items, { format: format || fmt.pct, max });
        return card;
    }

    /* ------------------------------------------------------------------ */
    /* Result rendering (shared by all single-document modes)              */
    /* ------------------------------------------------------------------ */

    function renderStatStrip(container, stats) {
        container.innerHTML = "";
        stats.forEach((stat, index) => {
            const tile = el("div", "stat-tile");
            tile.style.setProperty("--i", index);
            tile.innerHTML =
                `<span class="stat-label">${escapeHtml(stat.label)}</span>` +
                `<span class="stat-value" data-value>${stat.static ?? ""}</span>` +
                (stat.sub ? `<span class="stat-sub">${escapeHtml(stat.sub)}</span>` : "");
            container.appendChild(tile);
            if (stat.value !== undefined) {
                countUp(tile.querySelector("[data-value]"), stat.value, { format: stat.format || fmt.int });
            }
        });
    }

    function summaryCard(model, index) {
        const card = el("div", "summary-card");
        card.style.setProperty("--i", index);
        card.style.setProperty("--model-color", modelColor(model.key));
        card.dataset.model = model.key;

        const reliability = model.reliability || "unknown";
        card.innerHTML =
            `<div class="summary-head">` +
            `<h3><span class="model-dot"></span>${escapeHtml(model.name)}</h3>` +
            `<span style="display:flex;gap:6px;align-items:center">` +
            (model.language_match
                ? `<span class="match-badge ${model.language_match}" title="${
                    model.language_match === "cross_language"
                        ? "This model was not trained on this language"
                        : model.language_match === "multilingual"
                        ? "Multilingual model"
                        : "Trained on this language"
                  }">${escapeHtml(model.language_match.replace("_", " "))}</span>`
                : "") +
            `<span class="badge ${reliability}">${reliability}</span>` +
            `</span></div>` +
            `<div class="summary-method">${escapeHtml(model.method || "")}` +
            (model.runtime_seconds !== undefined ? ` · ${fmt.sec(model.runtime_seconds)}` : "") +
            `</div>` +
            `<div class="summary-text${model.summary ? "" : " is-loading"}" data-summary>` +
            `${model.summary ? escapeHtml(model.summary) : "Generating…"}</div>` +
            `<div class="summary-meta" data-meta></div>` +
            (model.warning ? `<div class="warning-note">${escapeHtml(model.warning)}</div>` : "");

        if (model.metrics) fillSummaryMeta(card.querySelector("[data-meta]"), model.metrics);
        return card;
    }

    function fillSummaryMeta(container, metrics) {
        const quality = metrics.overall_quality || {};
        const compression = metrics.compression_ratio || {};
        const abstract = metrics.abstractiveness || {};
        const factual = metrics.factual_consistency || {};
        const pills = [
            { text: `quality ${fmt.pct(quality.score)}`, kind: quality.score > 0.45 ? "good" : quality.score > 0.3 ? "" : "warn" },
            { text: `${compression.summary_words ?? "?"} words · ${fmt.pct(compression.ratio)} of source` },
            { text: `${fmt.pct(abstract.novel_1gram_ratio)} novel words` },
            {
                text: `faithful ${fmt.pct(factual.score)}`,
                kind: factual.score >= 0.999 ? "good" : factual.score >= 0.8 ? "warn" : "bad",
            },
        ];
        container.innerHTML = pills
            .map((p) => `<span class="pill${p.kind ? " is-" + p.kind : ""}">${escapeHtml(p.text)}</span>`)
            .join("");
    }

    const RADAR_AXES = [
        { key: "information_density", label: "Density" },
        { key: "topic_coverage", label: "Topics" },
        { key: "source_coverage", label: "Coverage" },
        { key: "factual_consistency", label: "Faithful" },
        { key: "non_redundancy", label: "Non-redund." },
        { key: "lexical_cohesion", label: "Cohesion" },
    ];

    function radarValues(metrics) {
        return {
            information_density: metrics.information_density?.score ?? 0,
            topic_coverage: metrics.topic_coverage?.score ?? 0,
            source_coverage: metrics.coverage_score?.score ?? 0,
            factual_consistency: metrics.factual_consistency?.score ?? 0,
            non_redundancy: 1 - (metrics.redundancy_score?.score ?? 0),
            lexical_cohesion: Math.min(1, (metrics.coherence_score?.score ?? 0) * 3),
        };
    }

    /**
     * Full metrics view for a set of models evaluated on one document.
     * models: [{key, name, summary, method, metrics, reliability, warning}]
     */
    function renderMetrics(container, models, options = {}) {
        container.innerHTML = "";
        const withMetrics = models.filter((m) => m.metrics);
        if (!withMetrics.length) return;

        const hasReference = withMetrics.some((m) => m.metrics.reference_available);

        // --- Reference-based section -------------------------------------
        container.appendChild(el("h3", "section-title", "Agreement with the reference summary"));
        if (hasReference) {
            const grid = el("div", "metrics-grid");
            [
                { title: "ROUGE-1 F1", metaKey: "rouge", path: ["rouge_scores", "rouge1", "f1"] },
                { title: "ROUGE-2 F1", metaKey: "rouge", path: ["rouge_scores", "rouge2", "f1"] },
                { title: "ROUGE-L F1", metaKey: "rouge", path: ["rouge_scores", "rougeL", "f1"] },
                { title: "METEOR", metaKey: "meteor", path: ["meteor_score", "score"] },
            ].forEach((spec, index) => {
                appendCard(grid, metricCard(withMetrics, {
                    title: spec.title,
                    metaKey: spec.metaKey,
                    index,
                    accessor: (m) => spec.path.reduce((node, key) => (node ? node[key] : null), m) ?? null,
                }));
            });
            container.appendChild(grid);
        } else {
            container.appendChild(
                el(
                    "div",
                    "callout is-warn",
                    "<strong>No reference summary supplied.</strong> ROUGE, METEOR and BERTScore " +
                    "compare a summary against a human-written reference, so they are not reported. " +
                    "Scoring against the source document instead would not be ROUGE — that quantity " +
                    "is shown below as the copy rate. Paste a reference summary to enable this section."
                )
            );
        }

        // --- Intrinsic section -------------------------------------------
        container.appendChild(el("h3", "section-title", "Content retention and faithfulness"));
        const intrinsic = el("div", "metrics-grid");
        [
            {
                title: "Information density", metaKey: "information_density",
                accessor: (m) => m.information_density?.score ?? null,
            },
            {
                title: "Topic coverage", metaKey: "topic_coverage",
                accessor: (m) => m.topic_coverage?.score ?? null,
            },
            {
                title: "Source coverage", metaKey: "source_coverage",
                accessor: (m) => m.coverage_score?.score ?? null,
            },
            {
                title: "Factual consistency (proxy)", metaKey: "factual_consistency",
                accessor: (m) => m.factual_consistency?.score ?? null,
            },
            {
                title: "Redundancy", metaKey: "redundancy", striped: true,
                accessor: (m) => m.redundancy_score?.score ?? null,
                note: "Lower is better. Bars show the raw score, so a shorter bar is a better result.",
                max: 1,
            },
            {
                title: "Lexical cohesion", metaKey: "lexical_cohesion",
                accessor: (m) => m.coherence_score?.score ?? null,
            },
        ].forEach((spec, index) => appendCard(intrinsic, metricCard(withMetrics, { ...spec, index })));
        container.appendChild(intrinsic);

        // --- Extraction vs. abstraction ----------------------------------
        container.appendChild(el("h3", "section-title", "Extraction vs. abstraction"));
        container.appendChild(
            el(
                "div",
                "section-note",
                "These measures are descriptive, not better-or-worse: they characterize <em>how</em> each " +
                "system produces its summary. A purely extractive method has zero novel n-grams by " +
                "construction; novel content is where genuine abstraction and hallucination both live."
            )
        );
        const style = el("div", "metrics-grid");
        [
            {
                title: "Novel words (abstractiveness)", metaKey: "abstractiveness",
                accessor: (m) => m.abstractiveness?.novel_1gram_ratio ?? null, max: 1,
            },
            {
                title: "Novel bigrams", metaKey: "abstractiveness",
                accessor: (m) => m.abstractiveness?.novel_2gram_ratio ?? null, max: 1,
            },
            {
                title: "Copy rate (words from source)", metaKey: "extractive_fragments",
                accessor: (m) => m.source_overlap?.unigram_copy_rate ?? null, max: 1,
            },
            {
                title: "Extractive density", metaKey: "extractive_fragments",
                accessor: (m) => m.extractive_fragments?.density ?? null,
                format: (v) => fmt.num(v, 1),
                note: "Mean squared length of copied runs: ~1 means word-level reuse, high means copied sentences.",
            },
            {
                title: "Compression ratio", metaKey: "compression_ratio",
                accessor: (m) => m.compression_ratio?.ratio ?? null, striped: true,
                note: "Lower means more compression. Compare other metrics only at similar compression.",
            },
            {
                title: "Runtime", metaKey: "overall_quality",
                accessor: (m) => m._runtime ?? null,
                format: fmt.sec, striped: true,
                note: "Wall-clock time on this machine, for the efficiency comparison.",
            },
        ].forEach((spec, index) => appendCard(style, metricCard(withMetrics, { ...spec, index })));
        container.appendChild(style);

        // --- Composite + profile -----------------------------------------
        container.appendChild(el("h3", "section-title", "Overall quality"));
        const basis = withMetrics[0].metrics.overall_quality?.basis;
        container.appendChild(
            el(
                "div",
                "callout",
                `<strong>Basis: ${basis === "reference+intrinsic" ? "reference + intrinsic metrics" : "intrinsic metrics only"}.</strong> ` +
                "The composite is a weighted mean; the weights are a stated modelling choice, not an " +
                "empirical result. Hover a metric's <em>i</em> for its definition and caveat."
            )
        );

        const compositeGrid = el("div", "metrics-grid");
        const compositeCard = el("div", "metric-card");
        compositeCard.innerHTML =
            `<div class="metric-head"><span class="metric-title">Composite score</span>` +
            `${directionTag("overall_quality")}${infoTip("overall_quality")}</div>`;
        const compositeChart = el("div");
        compositeCard.appendChild(compositeChart);
        barRows(
            compositeChart,
            withMetrics.map((m) => ({
                label: m.shortName || m.name,
                value: m.metrics.overall_quality?.score ?? 0,
                color: modelColor(m.key),
            })),
            { max: 1 }
        );
        compositeGrid.appendChild(compositeCard);

        if (withMetrics.length >= 2) {
            const radarCard = el("div", "metric-card");
            radarCard.innerHTML =
                `<div class="metric-head"><span class="metric-title">Metric profile</span>` +
                `<span class="metric-dir">normalized</span></div>` +
                `<div class="metric-source">Cohesion is scaled ×3 to be visible on the same axes.</div>`;
            const radarHost = el("div");
            radarCard.appendChild(radarHost);
            radar(
                radarHost,
                RADAR_AXES,
                withMetrics.map((m) => ({
                    name: m.shortName || m.name,
                    color: modelColor(m.key),
                    values: radarValues(m.metrics),
                }))
            );
            compositeGrid.appendChild(radarCard);

            const scatterCard = el("div", "metric-card is-wide");
            scatterCard.innerHTML =
                `<div class="metric-head"><span class="metric-title">Abstractiveness vs. quality</span>` +
                `<span class="metric-dir">descriptive</span></div>`;
            const scatterHost = el("div");
            scatterCard.appendChild(scatterHost);
            scatter(
                scatterHost,
                withMetrics.map((m) => ({
                    x: m.metrics.abstractiveness?.novel_1gram_ratio ?? 0,
                    y: m.metrics.overall_quality?.score ?? 0,
                    label: m.shortName || m.name,
                    color: modelColor(m.key),
                })),
                { xLabel: "novel unigrams", yLabel: "composite quality" }
            );
            compositeGrid.appendChild(scatterCard);
        }
        container.appendChild(compositeGrid);

        // --- Per-model detail --------------------------------------------
        container.appendChild(el("h3", "section-title", "Per-model detail"));
        const accordions = el("div", "accordions");
        withMetrics.forEach((model, index) => accordions.appendChild(modelAccordion(model, index)));
        container.appendChild(accordions);

        if (options.errors && Object.keys(options.errors).length) {
            container.appendChild(el("h3", "section-title", "Models that failed"));
            Object.entries(options.errors).forEach(([key, error]) => {
                container.appendChild(
                    el("div", "callout is-bad",
                        `<strong>${escapeHtml(error.name || key)}:</strong> ${escapeHtml(error.error)}`)
                );
            });
        }
    }

    function modelAccordion(model, index) {
        const metrics = model.metrics;
        const wrapper = el("div", "accordion");
        wrapper.style.setProperty("--i", index);

        const head = el("button", "accordion-head");
        head.innerHTML =
            `<span class="model-dot" style="background:${modelColor(model.key)}"></span>` +
            `<span class="accordion-name">${escapeHtml(model.name)}</span>` +
            `<span class="accordion-score">${fmt.pct(metrics.overall_quality?.score)}</span>` +
            `<svg class="accordion-chevron" width="15" height="15" viewBox="0 0 24 24" fill="none" ` +
            `stroke="currentColor" stroke-width="2"><path d="M6 9l6 6 6-6"/></svg>`;

        const body = el("div", "accordion-body");
        const inner = el("div", "", `<div class="accordion-inner"></div>`);
        const content = inner.firstElementChild;
        body.appendChild(inner);

        // Explanations
        const explanations = metrics.explanations || {};
        const list = el("div", "explain-list");
        Object.entries(explanations).forEach(([key, text]) => {
            list.appendChild(
                el("div", "explain-item",
                    `<h5>${escapeHtml(key.replace(/_/g, " "))}</h5><p>${escapeHtml(text)}</p>`)
            );
        });
        content.appendChild(list);

        // Unsupported items (the faithfulness evidence)
        const factual = metrics.factual_consistency || {};
        const unsupported = [...(factual.unsupported_numbers || []), ...(factual.unsupported_entities || [])];
        if (unsupported.length) {
            const box = el("div", "");
            box.innerHTML =
                `<div class="metric-head"><span class="metric-title">Not found in the source</span>` +
                `<span class="metric-dir">possible hallucination</span></div>`;
            const chips = el("div", "chip-list");
            unsupported.forEach((item) => chips.appendChild(el("span", "term-chip is-missing", escapeHtml(item))));
            box.appendChild(chips);
            content.appendChild(box);
        }

        // Topic coverage detail
        const topics = metrics.topic_coverage || {};
        if ((topics.covered_topics || []).length || (topics.missing_topics || []).length) {
            const box = el("div", "");
            box.innerHTML = `<div class="metric-head"><span class="metric-title">Key phrases</span></div>`;
            const chips = el("div", "chip-list");
            (topics.covered_topics || []).forEach((t) => chips.appendChild(el("span", "term-chip", escapeHtml(t))));
            (topics.missing_topics || []).forEach((t) =>
                chips.appendChild(el("span", "term-chip is-missing", escapeHtml(t))));
            box.appendChild(chips);
            content.appendChild(box);
        }

        // Readability + length
        const readability = metrics.readability || {};
        const length = metrics.length_stats || {};
        const facts = [
            `${length.summary_words ?? "—"} words in ${length.summary_sentences ?? "—"} sentences`,
            `${readability.words_per_sentence ?? "—"} words/sentence`,
            `LIX ${readability.lix ?? "—"}`,
            readability.flesch_reading_ease !== undefined ? `Flesch ${readability.flesch_reading_ease}` : null,
            `type/token ${readability.type_token_ratio ?? "—"}`,
            `distinct-2 ${fmt.pct(metrics.redundancy_score?.distinct_2)}`,
        ].filter(Boolean);
        content.appendChild(
            el("div", "summary-meta", facts.map((f) => `<span class="pill">${escapeHtml(f)}</span>`).join(""))
        );

        head.addEventListener("click", () => wrapper.classList.toggle("is-open"));
        wrapper.append(head, body);
        return wrapper;
    }

    function renderRankings(container, rankings, models) {
        const nameOf = (key) => {
            const model = models.find((m) => m.key === key);
            return model ? model.shortName || model.name : key;
        };
        const labels = {
            by_overall_quality: "Composite quality",
            by_rouge_l: "ROUGE-L F1",
            by_information_density: "Information density",
            by_factual_consistency: "Factual consistency",
            by_source_coverage: "Source coverage",
            by_coherence: "Lexical cohesion",
            by_abstractiveness: "Abstractiveness",
        };
        const rows = Object.entries(rankings || {}).filter(([, value]) => value && value.length);
        if (!rows.length) return;

        const table = el("table", "data");
        table.innerHTML =
            `<thead><tr><th>Criterion</th><th style="text-align:left">Ranking (best first)</th></tr></thead>`;
        const tbody = el("tbody");
        rows.forEach(([key, ranking], index) => {
            const tr = el("tr");
            tr.style.setProperty("--i", index);
            const ordered = ranking
                .map(([model, score], position) =>
                    `${position + 1}. ${escapeHtml(nameOf(model))} <span class="pill">${fmt.pct(score)}</span>`)
                .join(" &nbsp; ");
            tr.innerHTML = `<td>${escapeHtml(labels[key] || key)}</td><td style="text-align:left">${ordered}</td>`;
            tbody.appendChild(tr);
        });
        table.appendChild(tbody);
        const wrap = el("div", "table-wrap");
        wrap.appendChild(table);
        container.appendChild(el("h3", "section-title", "Rankings by criterion"));
        container.appendChild(wrap);
    }

    /* ------------------------------------------------------------------ */
    /* Run modes                                                           */
    /* ------------------------------------------------------------------ */

    async function runStream() {
        const request = readRequest();
        if (!request) return;
        await ensureMetricMeta();
        setBusy(true);

        const results = $("single-results");
        results.classList.remove("hidden");
        $("single-metrics").innerHTML = "";
        const grid = $("summaries-grid");
        grid.innerHTML = "";

        progress.show([
            { key: "preprocessing", label: "Preprocessing and language detection" },
            { key: "extractive", label: "Extractive summarization (TextRank)" },
            { key: "abstractive", label: "Abstractive summarization (transformer)" },
            { key: "metrics", label: "Evaluating both summaries" },
        ]);
        progress.stage("preprocessing", "running");

        const models = [
            { key: "extractive_textrank", name: "Extractive · TextRank", shortName: "TextRank", method: "Graph-based sentence selection" },
            { key: "abstractive_transformer", name: "Abstractive · Transformer", shortName: "Transformer", method: "Neural sequence-to-sequence" },
        ];
        models.forEach((model, index) => grid.appendChild(summaryCard(model, index)));

        const cardFor = (key) => grid.querySelector(`.summary-card[data-model="${key}"]`);

        try {
            await streamEvents("/summarize/stream", request, (event) => {
                const data = event.data || {};
                if (event.stage === "preprocessing") {
                    progress.stage("preprocessing", "done");
                    progress.stage("extractive", "running");
                    progress.set(25, "Extractive summarization…");
                    $("doc-lang").textContent = data.language;
                    $("doc-lang").classList.remove("hidden");
                    refreshModelAvailability(data.language);
                    renderStatStrip($("input-stats"), [
                        { label: "Words", value: data.word_count },
                        { label: "Sentences", value: data.sentence_count },
                        { label: "Chunks", value: data.chunk_count, sub: "for the transformer" },
                        { label: "Language", static: String(data.language).toUpperCase() },
                    ]);
                } else if (event.stage === "extractive") {
                    progress.stage("extractive", "done");
                    progress.stage("abstractive", "running");
                    progress.set(50, "Abstractive summarization (this is the slow one)…");
                    const card = cardFor("extractive_textrank");
                    const text = card.querySelector("[data-summary]");
                    text.textContent = data.summary;
                    text.classList.remove("is-loading");
                    models[0].summary = data.summary;
                    models[0].method = data.method || models[0].method;
                    models[0].runtime_seconds = data.runtime_seconds;
                    models[0].reliability = data.reliability || "high";
                    card.querySelector(".summary-method").textContent =
                        `${data.method || ""} · ${fmt.sec(data.runtime_seconds)}`;
                    const extractiveBadge = card.querySelector(".badge");
                    extractiveBadge.className = `badge ${models[0].reliability}`;
                    extractiveBadge.textContent = models[0].reliability;
                } else if (event.stage === "abstractive") {
                    progress.stage("abstractive", "done");
                    progress.stage("metrics", "running");
                    progress.set(78, "Computing metrics…");
                    const card = cardFor("abstractive_transformer");
                    const text = card.querySelector("[data-summary]");
                    text.textContent = data.summary;
                    text.classList.remove("is-loading");
                    card.querySelector(".summary-method").textContent =
                        `${data.method || ""} · ${fmt.sec(data.runtime_seconds)}`;
                    models[1].runtime_seconds = data.runtime_seconds;
                    card.querySelector(".badge").className = `badge ${data.reliability || "unknown"}`;
                    card.querySelector(".badge").textContent = data.reliability || "unknown";
                    if (data.warning) card.appendChild(el("div", "warning-note", escapeHtml(data.warning)));
                    models[1].summary = data.summary;
                    models[1].method = data.method || models[1].method;
                    models[1].reliability = data.reliability;
                } else if (event.stage === "metrics") {
                    progress.stage("metrics", "done");
                    progress.set(100, "Done");
                    const individual = data.individual_results || {};
                    models.forEach((model) => {
                        model.metrics = individual[model.key];
                        if (model.metrics) model.metrics._runtime = model.runtime_seconds;
                        const card = cardFor(model.key);
                        if (model.metrics && card) fillSummaryMeta(card.querySelector("[data-meta]"), model.metrics);
                    });
                    const host = $("single-metrics");
                    renderMetrics(host, models);
                    renderRankings(host, data.rankings, models);
                } else if (event.stage === "error") {
                    toast(`Summarization failed: ${data.message}`, "bad");
                }
            });
        } catch (error) {
            toast(`Request failed: ${error.message}`, "bad");
        } finally {
            setBusy(false);
            progress.hide();
        }
    }

    async function runSingleMethod(kind) {
        const request = readRequest();
        if (!request) return;
        await ensureMetricMeta();
        setBusy(true);
        progress.show([{ key: "run", label: kind === "extractive" ? "TextRank" : "Transformer" }]);
        progress.stage("run", "running");
        progress.set(40, "Summarizing…");

        try {
            const started = performance.now();
            const data = await api(`/summarize/${kind}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(request),
            });
            const runtime = (performance.now() - started) / 1000;
            progress.stage("run", "done");
            progress.set(100, "Done");

            const model = {
                key: kind === "extractive" ? "extractive_textrank" : "abstractive_transformer",
                name: kind === "extractive" ? "Extractive · TextRank" : "Abstractive · Transformer",
                shortName: kind === "extractive" ? "TextRank" : "Transformer",
                summary: data.summary,
                method: data.method,
                metrics: data.metrics,
                reliability: data.reliability,
                warning: data.warning,
                runtime_seconds: runtime,
            };
            if (model.metrics) model.metrics._runtime = runtime;

            $("single-results").classList.remove("hidden");
            const stats = data.metrics?.length_stats || {};
            renderStatStrip($("input-stats"), [
                { label: "Source words", value: stats.source_words },
                { label: "Summary words", value: stats.summary_words },
                { label: "Reduction", value: data.metrics?.compression_ratio?.reduction_percentage, format: (v) => `${v.toFixed(0)}%` },
                { label: "Language", static: String(data.metrics?.language || "").toUpperCase() },
            ]);
            const grid = $("summaries-grid");
            grid.innerHTML = "";
            grid.appendChild(summaryCard(model, 0));
            renderMetrics($("single-metrics"), [model]);
        } catch (error) {
            toast(`${kind} summarization failed: ${error.message}`, "bad");
        } finally {
            setBusy(false);
            progress.hide();
        }
    }

    async function runCompareAll(tabs) {
        const request = readRequest();
        if (!request) return;
        await ensureMetricMeta();
        setBusy(true);
        tabs.activate("compare");

        const host = $("compare-results");
        host.classList.remove("hidden");
        $("compare-empty").classList.add("hidden");
        host.innerHTML =
            `<div class="callout"><strong>Running every available model…</strong> ` +
            `Local transformers run on CPU; the first run of a model also downloads its weights.</div>` +
            `<div class="metrics-grid">${Array.from({ length: 3 })
                .map(() => `<div class="metric-card"><div class="skeleton-chip" style="width:100%;height:78px;border-radius:9px"></div></div>`)
                .join("")}</div>`;

        try {
            const data = await api("/compare/all", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(request),
            });

            const models = Object.entries(data.results).map(([key, value]) => ({
                key,
                name: value.name,
                shortName: value.name.replace(/\s*\(.*?\)\s*/g, "").trim(),
                summary: value.summary,
                method: value.method,
                metrics: value.metrics,
                reliability: value.reliability,
                warning: value.warning,
                runtime_seconds: value.runtime_seconds,
                language_match: value.language_match,
            }));
            models.forEach((m) => {
                if (m.metrics) m.metrics._runtime = m.runtime_seconds;
            });

            if (data.input_stats?.language) {
                refreshModelAvailability(data.input_stats.language);
            }
            host.innerHTML = "";
            const strip = el("div", "stat-strip");
            host.appendChild(strip);
            renderStatStrip(strip, [
                { label: "Models compared", value: models.length },
                { label: "Source words", value: data.input_stats?.word_count },
                { label: "Sentences", value: data.input_stats?.sentence_count },
                { label: "Language", static: String(data.input_stats?.language || "").toUpperCase() },
                {
                    label: "Reference",
                    static: models[0]?.metrics?.reference_available ? "provided" : "none",
                },
            ]);

            host.appendChild(el("h3", "section-title", "Summaries"));
            const grid = el("div", "summaries-grid");
            models.forEach((model, index) => grid.appendChild(summaryCard(model, index)));
            host.appendChild(grid);

            const metricsHost = el("div");
            host.appendChild(metricsHost);
            renderMetrics(metricsHost, models, { errors: data.errors });
            renderRankings(metricsHost, data.rankings, models);

            const failed = Object.keys(data.errors || {}).length;
            toast(
                `Compared ${models.length} model${models.length === 1 ? "" : "s"}` +
                (failed ? `, ${failed} failed` : ""),
                failed ? "warn" : "good"
            );
        } catch (error) {
            host.innerHTML = `<div class="callout is-bad"><strong>Comparison failed.</strong> ${escapeHtml(error.message)}</div>`;
        } finally {
            setBusy(false);
        }
    }

    /* ------------------------------------------------------------------ */
    /* Methodology tab                                                     */
    /* ------------------------------------------------------------------ */

    let methodologyLoaded = false;

    async function loadMethodology() {
        if (methodologyLoaded) return;
        const host = $("methodology-content");
        await ensureMetricMeta();
        const families = {
            reference: "Reference-based — require a human reference summary",
            intrinsic: "Intrinsic — computed against the source document only",
            composite: "Composite",
        };

        host.innerHTML = "";
        Object.entries(families).forEach(([family, title]) => {
            const entries = Object.entries(METRIC_META || {}).filter(([, meta]) => meta.family === family);
            if (!entries.length) return;
            host.appendChild(el("h3", "section-title", title));
            const table = el("table", "data");
            table.innerHTML =
                `<thead><tr>` +
                `<th style="text-align:left">Metric</th>` +
                `<th style="text-align:left">Measures</th>` +
                `<th>Direction</th><th>Range</th>` +
                `<th style="text-align:left">Source</th>` +
                `<th style="text-align:left">Limitation</th>` +
                `</tr></thead>`;
            const tbody = el("tbody");
            entries.forEach(([, meta], index) => {
                const row = el("tr");
                row.style.setProperty("--i", index);
                row.innerHTML =
                    `<td style="text-align:left"><strong>${escapeHtml(meta.label)}</strong></td>` +
                    `<td style="text-align:left;white-space:normal;max-width:320px">${escapeHtml(meta.measures)}</td>` +
                    `<td>${escapeHtml(meta.direction)}</td>` +
                    `<td class="num">${escapeHtml(meta.range)}</td>` +
                    `<td style="text-align:left">${escapeHtml(meta.source)}</td>` +
                    `<td style="text-align:left;white-space:normal;max-width:300px">${escapeHtml(meta.caveat)}</td>`;
                tbody.appendChild(row);
            });
            table.appendChild(tbody);
            const wrap = el("div", "table-wrap");
            wrap.appendChild(table);
            host.appendChild(wrap);
        });

        host.appendChild(el("h3", "section-title", "Statistical procedure"));
        host.appendChild(
            el(
                "div",
                "callout",
                "<strong>Per-document scores → aggregate claims.</strong> The Experiment tab evaluates every " +
                "model on every selected document, then reports each system's mean with a percentile " +
                "<em>bootstrap 95% confidence interval</em> (10,000 resamples, fixed seed). Systems are " +
                "compared with a <em>paired bootstrap resampling test</em> (Koehn, 2004) — documents are " +
                "resampled with each system's paired scores kept together — cross-checked with a " +
                "<em>Wilcoxon signed-rank test</em>, and corrected for multiple comparisons with " +
                "<em>Holm–Bonferroni</em>. Effect sizes are reported as Cohen's d for paired samples. " +
                "Below five documents no p-value is reported at all: a paired bootstrap over two documents " +
                "returns an arbitrarily small number that reflects the resampling procedure, not evidence."
            )
        );
        host.appendChild(
            el(
                "div",
                "callout is-warn",
                "<strong>Known limitations of this evaluation.</strong> Composite weights are a modelling " +
                "choice. Information density and source coverage reward reusing source vocabulary, which " +
                "systematically favours extractive methods. Lexical cohesion is a proxy for coherence and " +
                "cannot judge logical flow. Factual consistency is a surface check over numbers and proper " +
                "nouns; it cannot detect distorted relations between facts. LIX penalizes agglutinative " +
                "languages such as Turkish. None of these metrics replaces human judgement."
            )
        );
        methodologyLoaded = true;
    }

    /* ------------------------------------------------------------------ */
    /* Boot                                                                */
    /* ------------------------------------------------------------------ */

    function init() {
        initTheme();
        const tabs = initTabs();
        initInputs();
        updateInputMeter();
        initSamples();
        ensureMetricMeta();
        refreshStatus();
        refreshModelAvailability();
        setInterval(refreshStatus, 45000);

        $("btn-stream").addEventListener("click", runStream);
        $("btn-extractive").addEventListener("click", () => runSingleMethod("extractive"));
        $("btn-abstractive").addEventListener("click", () => runSingleMethod("abstractive"));
        $("btn-compare-all").addEventListener("click", () => runCompareAll(tabs));
        $("cross-lingual")?.addEventListener("change", () => refreshModelAvailability());

        window.App = {
            API, api, streamEvents, toast, el, $, escapeHtml,
            modelColor, MODEL_TYPE_LABELS, fmt,
            ensureMetricMeta, infoTip, directionTag,
            tabs,
        };
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
