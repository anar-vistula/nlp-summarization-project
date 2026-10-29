/* ===========================================================================
   Chart primitives — plain SVG and DOM, no external libraries.
   Deliberately dependency-free: the app must render identically offline, on a
   projector, with no CDN reachable.
   All charts read their colors from CSS custom properties, so they follow the
   light/dark theme automatically.
   =========================================================================== */

(function () {
    "use strict";

    const fmt = {
        pct: (v, digits = 1) =>
            v === null || v === undefined || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(digits)}%`,
        num: (v, digits = 2) =>
            v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(digits),
        sec: (v) => (v === null || v === undefined ? "—" : `${Number(v).toFixed(v < 10 ? 2 : 1)}s`),
        int: (v) => (v === null || v === undefined ? "—" : String(Math.round(v))),
    };

    const prefersReducedMotion = () =>
        window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    /** Animate a number from 0 (or from its current value) to `to`. */
    function countUp(el, to, options = {}) {
        const format = options.format || fmt.num;
        const duration = prefersReducedMotion() ? 0 : options.duration ?? 850;
        const from = options.from ?? 0;
        if (to === null || to === undefined || Number.isNaN(to)) {
            el.textContent = "—";
            return;
        }
        if (duration === 0) {
            el.textContent = format(to);
            return;
        }
        const start = performance.now();
        function frame(now) {
            const t = Math.min(1, (now - start) / duration);
            // easeOutCubic
            const eased = 1 - Math.pow(1 - t, 3);
            el.textContent = format(from + (to - from) * eased);
            if (t < 1) requestAnimationFrame(frame);
        }
        requestAnimationFrame(frame);
    }

    /**
     * Horizontal bar rows.
     * items: [{label, value, color, display?, striped?, title?}]
     */
    function barRows(container, items, options = {}) {
        const format = options.format || fmt.pct;
        const values = items.map((i) => i.value).filter((v) => typeof v === "number");
        const max = options.max ?? Math.max(0.0001, ...values);
        container.innerHTML = "";
        const wrap = document.createElement("div");
        wrap.className = "bar-rows";

        items.forEach((item, index) => {
            const row = document.createElement("div");
            row.className = "bar-row";
            if (item.title) row.title = item.title;

            const label = document.createElement("span");
            label.className = "bar-label";
            label.textContent = item.label;

            const track = document.createElement("div");
            track.className = "bar-track";
            const fill = document.createElement("div");
            fill.className = "bar-fill" + (item.striped ? " is-striped" : "");
            fill.style.setProperty("--bar-color", item.color || "var(--accent)");
            fill.style.transitionDelay = `${index * 60}ms`;
            track.appendChild(fill);

            const value = document.createElement("span");
            value.className = "bar-value";

            row.append(label, track, value);
            wrap.appendChild(row);

            const ratio = typeof item.value === "number" ? Math.max(0, Math.min(1, item.value / max)) : 0;
            requestAnimationFrame(() => {
                fill.style.width = `${ratio * 100}%`;
            });
            countUp(value, item.value, { format: item.display ? () => item.display : format });
        });

        container.appendChild(wrap);
    }

    /**
     * Mean with 95% confidence interval, one row per system.
     * items: [{label, color, mean, low, high, n, suffix?}]
     */
    function ciRows(container, items, options = {}) {
        const format = options.format || fmt.pct;
        const lows = items.map((i) => (typeof i.low === "number" ? i.low : i.mean));
        const highs = items.map((i) => (typeof i.high === "number" ? i.high : i.mean));
        const dataMin = Math.min(...lows, ...items.map((i) => i.mean));
        const dataMax = Math.max(...highs, ...items.map((i) => i.mean));
        const pad = Math.max((dataMax - dataMin) * 0.18, Math.abs(dataMax) * 0.05, 0.0001);
        const min = options.min ?? dataMin - pad;
        const max = options.max ?? dataMax + pad;
        const span = max - min || 1;
        const pos = (v) => `${((v - min) / span) * 100}%`;

        container.innerHTML = "";
        const wrap = document.createElement("div");
        wrap.className = "ci-rows";

        items.forEach((item, index) => {
            const row = document.createElement("div");
            row.className = "ci-row";
            row.style.setProperty("--i", index);
            row.style.setProperty("--bar-color", item.color || "var(--accent)");

            const label = document.createElement("span");
            label.className = "bar-label";
            label.textContent = item.label;

            const track = document.createElement("div");
            track.className = "ci-track";
            const axis = document.createElement("div");
            axis.className = "ci-axis";
            track.appendChild(axis);

            const hasInterval = typeof item.low === "number" && typeof item.high === "number" && item.high > item.low;
            if (hasInterval) {
                const bar = document.createElement("div");
                bar.className = "ci-bar";
                bar.style.left = pos(item.low);
                bar.style.width = `calc(${pos(item.high)} - ${pos(item.low)})`;
                track.appendChild(bar);

                [item.low, item.high].forEach((v) => {
                    const whisker = document.createElement("div");
                    whisker.className = "ci-whisker";
                    whisker.style.left = pos(v);
                    track.appendChild(whisker);
                });
            }

            const point = document.createElement("div");
            point.className = "ci-point";
            point.style.left = pos(item.mean);
            point.style.setProperty("--i", index);
            track.appendChild(point);

            const value = document.createElement("span");
            value.className = "ci-value";
            const ciText = hasInterval ? ` [${format(item.low)}, ${format(item.high)}]` : "";
            value.innerHTML = `<strong>${format(item.mean)}</strong>${ciText}`;

            row.append(label, track, value);
            wrap.appendChild(row);
        });

        container.appendChild(wrap);
    }

    /**
     * Radar chart comparing systems across normalized (0–1) axes.
     * axes:   [{key, label}]
     * series: [{name, color, values: {key: number}}]
     */
    function radar(container, axes, series) {
        const size = 300;
        const center = size / 2;
        const radius = center - 34;
        const rings = 4;
        const angleFor = (i) => (Math.PI * 2 * i) / axes.length - Math.PI / 2;
        const point = (i, value) => {
            const r = radius * Math.max(0.02, Math.min(1, value));
            return [center + Math.cos(angleFor(i)) * r, center + Math.sin(angleFor(i)) * r];
        };

        const svgNS = "http://www.w3.org/2000/svg";
        const svg = document.createElementNS(svgNS, "svg");
        svg.setAttribute("viewBox", `0 0 ${size} ${size}`);
        svg.setAttribute("class", "radar-svg");
        svg.setAttribute("role", "img");

        // Grid rings
        for (let ring = 1; ring <= rings; ring++) {
            const polygon = document.createElementNS(svgNS, "polygon");
            const pts = axes.map((_, i) => point(i, ring / rings).join(",")).join(" ");
            polygon.setAttribute("points", pts);
            polygon.setAttribute("class", "radar-grid");
            svg.appendChild(polygon);
        }

        // Axes and labels
        axes.forEach((axis, i) => {
            const line = document.createElementNS(svgNS, "line");
            const [x, y] = point(i, 1);
            line.setAttribute("x1", center);
            line.setAttribute("y1", center);
            line.setAttribute("x2", x);
            line.setAttribute("y2", y);
            line.setAttribute("class", "radar-axis");
            svg.appendChild(line);

            const label = document.createElementNS(svgNS, "text");
            const lx = center + Math.cos(angleFor(i)) * (radius + 18);
            const ly = center + Math.sin(angleFor(i)) * (radius + 18);
            label.setAttribute("x", lx);
            label.setAttribute("y", ly);
            label.setAttribute("class", "radar-label");
            label.setAttribute("text-anchor", lx < center - 8 ? "end" : lx > center + 8 ? "start" : "middle");
            label.setAttribute("dominant-baseline", "middle");
            label.textContent = axis.label;
            svg.appendChild(label);
        });

        // Series polygons
        series.forEach((entry, index) => {
            const polygon = document.createElementNS(svgNS, "polygon");
            const pts = axes
                .map((axis, i) => point(i, entry.values[axis.key] ?? 0).join(","))
                .join(" ");
            polygon.setAttribute("points", pts);
            polygon.setAttribute("class", "radar-series");
            polygon.setAttribute("fill", entry.color);
            polygon.setAttribute("stroke", entry.color);
            polygon.style.animationDelay = `${index * 140}ms`;
            const title = document.createElementNS(svgNS, "title");
            title.textContent = entry.name;
            polygon.appendChild(title);
            svg.appendChild(polygon);
        });

        const wrap = document.createElement("div");
        wrap.className = "radar-wrap";
        wrap.appendChild(svg);

        const legend = document.createElement("div");
        legend.className = "legend";
        series.forEach((entry) => {
            const item = document.createElement("div");
            item.className = "legend-item";
            const swatch = document.createElement("span");
            swatch.className = "legend-swatch";
            swatch.style.background = entry.color;
            const name = document.createElement("span");
            name.textContent = entry.name;
            item.append(swatch, name);
            legend.appendChild(item);
        });
        wrap.appendChild(legend);

        container.innerHTML = "";
        container.appendChild(wrap);
    }

    /**
     * Scatter plot. points: [{x, y, label, color, r?}]
     */
    function scatter(container, points, options = {}) {
        const width = 460;
        const height = 260;
        const margin = { top: 14, right: 18, bottom: 34, left: 44 };
        const xFormat = options.xFormat || fmt.pct;
        const yFormat = options.yFormat || fmt.pct;

        const xs = points.map((p) => p.x);
        const ys = points.map((p) => p.y);
        const xMin = options.xMin ?? Math.min(0, ...xs);
        const xMax = options.xMax ?? Math.max(0.0001, ...xs) * 1.1;
        const yMin = options.yMin ?? Math.min(...ys) * 0.9;
        const yMax = options.yMax ?? Math.max(0.0001, ...ys) * 1.08;

        const sx = (v) => margin.left + ((v - xMin) / (xMax - xMin || 1)) * (width - margin.left - margin.right);
        const sy = (v) => height - margin.bottom - ((v - yMin) / (yMax - yMin || 1)) * (height - margin.top - margin.bottom);

        const svgNS = "http://www.w3.org/2000/svg";
        const svg = document.createElementNS(svgNS, "svg");
        svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
        svg.setAttribute("class", "scatter-svg");

        const addLine = (x1, y1, x2, y2) => {
            const line = document.createElementNS(svgNS, "line");
            line.setAttribute("x1", x1); line.setAttribute("y1", y1);
            line.setAttribute("x2", x2); line.setAttribute("y2", y2);
            line.setAttribute("class", "scatter-axis-line");
            svg.appendChild(line);
        };
        const addText = (x, y, text, anchor = "middle") => {
            const node = document.createElementNS(svgNS, "text");
            node.setAttribute("x", x); node.setAttribute("y", y);
            node.setAttribute("class", "scatter-label");
            node.setAttribute("text-anchor", anchor);
            node.textContent = text;
            svg.appendChild(node);
        };

        // Axes with 3 ticks each
        addLine(margin.left, height - margin.bottom, width - margin.right, height - margin.bottom);
        addLine(margin.left, margin.top, margin.left, height - margin.bottom);
        for (let i = 0; i <= 2; i++) {
            const xv = xMin + ((xMax - xMin) * i) / 2;
            addText(sx(xv), height - margin.bottom + 15, xFormat(xv, 0));
            const yv = yMin + ((yMax - yMin) * i) / 2;
            addText(margin.left - 7, sy(yv) + 3, yFormat(yv, 0), "end");
        }
        addText((width + margin.left) / 2, height - 4, options.xLabel || "");
        const yTitle = document.createElementNS(svgNS, "text");
        yTitle.setAttribute("transform", `rotate(-90) translate(${-(height - margin.bottom + margin.top) / 2}, 11)`);
        yTitle.setAttribute("class", "scatter-label");
        yTitle.setAttribute("text-anchor", "middle");
        yTitle.textContent = options.yLabel || "";
        svg.appendChild(yTitle);

        points.forEach((p, index) => {
            const group = document.createElementNS(svgNS, "g");
            group.setAttribute("class", "scatter-point");
            group.style.animationDelay = `${index * 80}ms`;

            const circle = document.createElementNS(svgNS, "circle");
            circle.setAttribute("cx", sx(p.x));
            circle.setAttribute("cy", sy(p.y));
            circle.setAttribute("r", p.r || 6);
            circle.setAttribute("fill", p.color);
            circle.setAttribute("fill-opacity", "0.85");
            circle.setAttribute("stroke", "var(--surface)");
            circle.setAttribute("stroke-width", "1.5");
            const title = document.createElementNS(svgNS, "title");
            title.textContent = `${p.label}: ${xFormat(p.x)} / ${yFormat(p.y)}`;
            circle.appendChild(title);
            group.appendChild(circle);

            const label = document.createElementNS(svgNS, "text");
            label.setAttribute("x", sx(p.x));
            label.setAttribute("y", sy(p.y) - 10);
            label.setAttribute("class", "scatter-label");
            label.setAttribute("text-anchor", "middle");
            label.textContent = p.label;
            group.appendChild(label);

            svg.appendChild(group);
        });

        container.innerHTML = "";
        container.appendChild(svg);
    }

    window.Charts = { barRows, ciRows, radar, scatter, countUp, fmt, prefersReducedMotion };
})();
