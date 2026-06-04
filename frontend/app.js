/**
 * Frontend application for the Text Summarization System.
 * Uses Server-Sent Events for progressive display of results.
 */

// Use same origin if served from FastAPI, otherwise fallback to localhost
const API_BASE = window.location.port === "8000" ? "" : "http://localhost:8000";

// --- Sample Texts ---
const SAMPLES = {
    climate: {
        text: `Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, such as through variations in the solar cycle, but since the 1800s, human activities have been the main driver of climate change, primarily due to burning fossil fuels like coal, oil and gas. Burning fossil fuels generates greenhouse gas emissions that act like a blanket wrapped around the Earth, trapping the sun's heat and raising temperatures.

Examples of greenhouse gas emissions that are causing climate change include carbon dioxide and methane. These come from using gasoline for driving a car or coal for heating a building, for example. Clearing land and forests can also release carbon dioxide. Landfills for garbage are a major source of methane emissions. Energy, industry, transport, buildings, agriculture and land use are among the main emitters.

The Earth is now about 1.1 degrees Celsius warmer than it was in the late 1800s. The last decade was the warmest on record. Many people think climate change mainly means warmer temperatures. But temperature rise is only the beginning of the story. Because the Earth is a system, where everything is connected, changes in one area can influence changes in all others.

The consequences of climate change now include, among others, intense droughts, water scarcity, severe fires, rising sea levels, flooding, melting polar ice, catastrophic storms and declining biodiversity. People are experiencing climate change in diverse ways. Climate change can affect our health, ability to grow food, housing, safety and work. Some of us are already more vulnerable to climate impacts, such as people living in small island nations and other developing countries.

Conditions like sea-level rise and saltwater intrusion have advanced to the point where whole communities have had to relocate, and protracted droughts are putting people at risk of famine. In the future, the number of people displaced by weather-related events is expected to rise. Every increase in global warming matters. In a series of UN reports, thousands of scientists and government reviewers agreed that limiting global temperature rise to no more than 1.5 degrees Celsius would help us avoid the worst climate impacts and maintain a livable climate.

Yet based on current national climate plans, global warming is projected to reach around 3.2 degrees Celsius by the end of the century. The emissions that cause climate change come from every part of the world and affect everyone, but some countries produce much more than others. The 100 least-emitting countries generate 3 per cent of total emissions. The 10 countries with the largest emissions contribute 68 per cent. Everyone must take climate action, but people and countries creating more of the problem have a greater responsibility to act first.

Many climate change solutions can deliver economic benefits while improving our lives and protecting the environment. We also have global frameworks and agreements to guide progress, such as the Sustainable Development Goals, the UN Framework Convention on Climate Change and the Paris Agreement. Three broad categories of action are: cutting emissions, adapting to climate impacts and financing required adjustments.

Switching energy systems from fossil fuels to renewables like solar or wind will reduce the emissions driving climate change. But we have to start right now. While a growing coalition of countries is committing to net zero emissions by 2050, about half of emissions cuts must be in place by 2030 to keep warming below 1.5 degrees. Fossil fuel production must decline by roughly 6 per cent per year between 2020 and 2030.`,
        reference: `Climate change is caused primarily by human activities since the 1800s, especially burning fossil fuels which release greenhouse gases. The Earth has warmed 1.1°C since the late 1800s, causing droughts, floods, rising seas, and biodiversity loss. Scientists recommend limiting warming to 1.5°C, but current plans project 3.2°C by century's end. Solutions include switching to renewable energy, adapting to impacts, and financing adjustments, with emissions needing to drop 6% annually through 2030.`
    },
    ai: {
        text: `Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.

The term "artificial intelligence" had previously been used to describe machines that mimic and display human cognitive skills that are associated with the human mind, such as learning and problem-solving. This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated.

AI applications include advanced web search engines, recommendation systems, understanding human speech, self-driving cars, automated decision-making and competing at the highest level in strategic game systems. As machines become increasingly capable, tasks considered to require intelligence are often removed from the definition of AI, a phenomenon known as the AI effect.

The field was founded on the assumption that human intelligence can be so precisely described that a machine can be made to simulate it. This raised philosophical arguments about the mind and the ethical consequences of creating artificial beings endowed with human-like intelligence. These issues have previously been explored by myth, fiction and philosophy since antiquity. Computer scientists and philosophers have since suggested that AI may become an existential risk to humanity if its rational capacities are not steered towards beneficial goals.

The general problem of simulating intelligence has been broken down into sub-problems. These consist of particular traits or capabilities that researchers expect an intelligent system to display. The traits described below have received the most attention and cover the majority of AI research.

Reasoning and problem solving: Early researchers developed algorithms that imitated step-by-step reasoning that humans use when they solve puzzles or make logical deductions. By the late 1980s and 1990s, methods were developed for dealing with uncertain or incomplete information, employing concepts from probability and economics.

Knowledge representation: Knowledge representation and knowledge engineering allow AI programs to answer questions intelligently and make deductions about real-world facts. Formal knowledge representations are used in content-based indexing and retrieval, scene interpretation, clinical decision support, knowledge discovery, and other areas.

Planning and decision making: An intelligent agent that can plan makes a representation of the state of the world, makes predictions about how their actions will change it and makes choices that maximize the utility of available choices. In classical planning problems, the agent can assume it is the only system acting in the world, allowing the agent to be certain of the consequences of its actions.

Machine learning: Machine learning is the study of programs that can improve their performance on a given task automatically. It has been a part of AI from the beginning. There are several kinds of machine learning. Unsupervised learning analyzes a stream of data and finds patterns and makes predictions without any other guidance. Supervised learning requires a human to label the input data first and comes in two main varieties: classification and numerical regression. Reinforcement learning is modeled on psychology and involves an agent improving its behavior based on rewards.

Natural language processing: Natural language processing allows programs to read, write and communicate in human languages such as English. Specific problems include speech recognition, speech synthesis, machine translation, information extraction, and question answering. Modern deep learning techniques for NLP include word embedding, transformers, and large language models.

The field of AI has experienced several waves of optimism, followed by disappointment and loss of funding, known as AI winters. The field recovered after each setback and has grown enormously since 2012, driven by advances in deep learning, increases in computing power, and the availability of large datasets. The current wave of AI development has led to breakthroughs in image recognition, natural language processing, game playing, and scientific discovery.`,
        reference: `Artificial intelligence is the field studying intelligent agents that perceive environments and take rational actions. AI applications include search engines, recommendation systems, speech recognition, self-driving cars, and game-playing systems. Key AI sub-problems include reasoning, knowledge representation, planning, machine learning, and natural language processing. The field has experienced cycles of optimism and AI winters but has grown enormously since 2012 due to deep learning advances, increased computing power, and large datasets.`
    },
    space: {
        text: `Space exploration is the use of astronomy and space technology to explore outer space. While the exploration of space is carried out mainly by astronomers with telescopes, its physical exploration is conducted both by uncrewed robotic space probes and human spaceflight. Space exploration, like its classical form astronomy, is one of the main sources for space science.

While the observation of objects in space, known as astronomy, predates reliable recorded history, it was the development of large and relatively efficient rockets during the mid-twentieth century that allowed physical space exploration to become a reality. The Soviet Union launched the first artificial satellite, Sputnik 1, in 1957, and the first human in space, Yuri Gagarin, in 1961. The United States landed the first humans on the Moon during the Apollo 11 mission in 1969.

After the first 20 years of exploration, focus shifted from one-off flights to renewable hardware, such as the Space Shuttle program, and from competition to cooperation as with the International Space Station. In the 2000s, China initiated a successful crewed spaceflight program while the European Union and Japan have also planned future crewed space missions.

In the 2010s, private spaceflight companies began to develop and launch their own rockets and spacecraft. SpaceX, founded by Elon Musk, developed the Falcon 9 rocket and Dragon spacecraft, which became the first commercial vehicles to deliver cargo and crew to the International Space Station. Blue Origin, founded by Jeff Bezos, developed the New Shepard suborbital vehicle. These companies have significantly reduced the cost of access to space through innovations like reusable rocket technology.

Mars exploration has been a major focus of space agencies worldwide. NASA's Mars rovers, including Spirit, Opportunity, Curiosity, and Perseverance, have explored the Martian surface and searched for signs of past microbial life. The Perseverance rover, which landed in February 2021, carries instruments designed to produce oxygen from the Martian atmosphere and collect rock samples for future return to Earth.

The James Webb Space Telescope, launched in December 2021, represents the next generation of space observatories. It is designed to observe the most distant events and objects in the universe, such as the formation of the first galaxies. The telescope operates primarily in the infrared spectrum and has already produced groundbreaking observations of exoplanet atmospheres, distant galaxies, and stellar nurseries.

Future plans for space exploration include NASA's Artemis program, which aims to return humans to the Moon and establish a sustainable presence there. The program plans to land the first woman and first person of color on the Moon. Beyond the Moon, both NASA and SpaceX have expressed ambitions to send humans to Mars within the coming decades.

The scientific benefits of space exploration are numerous. Satellite technology has revolutionized communications, weather forecasting, navigation, and Earth observation. Space-based telescopes have transformed our understanding of the universe. Research conducted in microgravity environments has led to advances in medicine, materials science, and fundamental physics. Space exploration also drives technological innovation that finds applications in everyday life, from memory foam to water purification systems.

International cooperation in space has served as a model for peaceful collaboration between nations. The International Space Station, involving five space agencies and fifteen countries, has been continuously occupied since November 2000 and has hosted over 250 visitors from 20 countries. This cooperation demonstrates that complex scientific and engineering challenges can be addressed through international partnership.`,
        reference: `Space exploration uses astronomy and space technology to explore outer space through telescopes, robotic probes, and human spaceflight. Beginning with Sputnik in 1957 and the Moon landing in 1969, focus shifted to reusable hardware and international cooperation. Private companies like SpaceX have reduced costs through reusable rockets. Mars exploration continues with rovers searching for past life, while the James Webb Telescope observes distant galaxies. Future plans include NASA's Artemis Moon program and eventual Mars missions. Space exploration has yielded benefits in communications, medicine, and international cooperation.`
    },
    health: {
        text: `Public health is the science of protecting and improving the health of people and their communities. This work is achieved by promoting healthy lifestyles, researching disease and injury prevention, and detecting, preventing and responding to infectious diseases. Overall, public health is concerned with protecting the health of entire populations. These populations can be as small as a local neighborhood, or as big as an entire country or region of the world.

Public health professionals try to prevent problems from happening or recurring through implementing educational programs, recommending policies, administering services and conducting research. In contrast to clinical professionals like doctors and nurses, who focus primarily on treating individuals after they become sick or injured, public health professionals focus on preventing illness and injury before they occur.

The concept of public health has evolved significantly over centuries. In ancient civilizations, public health measures included basic sanitation, quarantine of the sick, and regulation of food and water supplies. The modern public health movement began in the 19th century with the recognition that unsanitary conditions and contaminated water supplies were major causes of disease. John Snow's investigation of cholera outbreaks in London in 1854 is often cited as a founding event of modern epidemiology.

The 20th century saw enormous advances in public health. Vaccination programs eliminated smallpox and dramatically reduced the incidence of polio, measles, and other infectious diseases. Improvements in sanitation, nutrition, and living conditions contributed to significant increases in life expectancy. The discovery of antibiotics transformed the treatment of bacterial infections. Public health campaigns against smoking led to significant reductions in lung cancer and cardiovascular disease.

Epidemiology is the study and analysis of the distribution, patterns and determinants of health and disease conditions in a defined population. It is a cornerstone of public health, and shapes policy decisions and evidence-based practice by identifying risk factors for disease and targets for preventive healthcare. Epidemiologists help with study design, collection, and statistical analysis of data, amend interpretation, and dissemination of results.

The COVID-19 pandemic, which began in late 2019, highlighted both the importance and challenges of public health systems worldwide. The pandemic demonstrated the need for robust surveillance systems, rapid response capabilities, effective communication strategies, and international cooperation. It also exposed significant health disparities, with disadvantaged communities experiencing disproportionately higher rates of infection and death.

Mental health has emerged as a critical public health concern in recent decades. Depression, anxiety, and substance abuse disorders affect hundreds of millions of people worldwide and contribute significantly to disability and premature death. The stigma associated with mental illness often prevents people from seeking help. Public health approaches to mental health include promoting awareness, reducing stigma, improving access to treatment, and addressing social determinants of mental health.

Non-communicable diseases, including cardiovascular disease, cancer, diabetes, and chronic respiratory diseases, now account for the majority of deaths worldwide. These diseases share common risk factors including tobacco use, unhealthy diet, physical inactivity, and harmful use of alcohol. Public health strategies to address non-communicable diseases focus on prevention through lifestyle modification, early detection through screening programs, and ensuring access to affordable treatment.

Health equity is a fundamental principle of public health. Health disparities exist along lines of race, ethnicity, socioeconomic status, gender, geography, and other factors. Achieving health equity requires addressing the social determinants of health, including poverty, education, housing, employment, and access to healthcare. Public health professionals work to identify and eliminate barriers to health and to ensure that all people have the opportunity to achieve their full health potential.

The future of public health will be shaped by emerging challenges including climate change, antimicrobial resistance, aging populations, and the increasing burden of non-communicable diseases. Advances in genomics, data science, and digital health technologies offer new tools for disease prevention and health promotion. However, addressing these challenges will require sustained investment in public health infrastructure, workforce development, and research.`,
        reference: `Public health protects population health through prevention, education, policy, and research, unlike clinical medicine which treats individuals. Evolving from ancient sanitation to modern epidemiology, 20th-century advances included vaccination, antibiotics, and anti-smoking campaigns. COVID-19 exposed system weaknesses and health disparities. Current challenges include mental health, non-communicable diseases, and health equity across socioeconomic lines. Future public health must address climate change, antimicrobial resistance, and aging populations while leveraging genomics and digital health technologies.`
    }
};

function loadSample(key) {
    const sample = SAMPLES[key];
    if (sample) {
        document.getElementById("input-text").value = sample.text;
        document.getElementById("reference-summary").value = sample.reference;
    }
}

function collapseInput() {
    const section = document.getElementById("input-section");
    const collapsed = document.getElementById("input-collapsed");
    section.classList.add("collapsing");
    setTimeout(() => {
        section.classList.add("hidden");
        section.classList.remove("collapsing");
        collapsed.classList.remove("hidden");
        collapsed.classList.add("fade-in");
    }, 400);
}

function expandInput() {
    const section = document.getElementById("input-section");
    const collapsed = document.getElementById("input-collapsed");
    collapsed.classList.add("hidden");
    collapsed.classList.remove("fade-in");
    section.classList.remove("hidden");
    section.classList.add("expanding");
    setTimeout(() => {
        section.classList.remove("expanding");
    }, 400);
}

// Ratio slider
const ratioSlider = document.getElementById("ratio");
const ratioValue = document.getElementById("ratio-value");
ratioSlider.addEventListener("input", () => {
    ratioValue.textContent = ratioSlider.value;
});

// --- Progress / Loading State ---

function showLoading() {
    document.getElementById("loading").classList.remove("hidden");
    document.getElementById("results").classList.add("hidden");
    updateProgress("Preparing...", 0);
}

function hideLoading() {
    document.getElementById("loading").classList.add("hidden");
}

function showResults() {
    document.getElementById("results").classList.remove("hidden");
}

function updateProgress(message, percent) {
    const el = document.getElementById("loading");
    el.querySelector("p").textContent = message;
    const bar = el.querySelector(".progress-bar-fill");
    if (bar) bar.style.width = `${percent}%`;
}

// --- Main Summarize (Streaming) ---

async function summarize() {
    const text = document.getElementById("input-text").value.trim();
    const reference = document.getElementById("reference-summary").value.trim();
    const ratio = parseFloat(ratioSlider.value);

    if (text.length < 50) {
        alert("Please enter at least 50 characters of text.");
        return;
    }

    collapseInput();
    showLoading();
    resetResults();

    try {
        const response = await fetch(`${API_BASE}/summarize/stream`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                text,
                ratio,
                reference_summary: reference || null,
            }),
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Request failed");
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n\n");
            buffer = lines.pop(); // keep incomplete chunk

            for (const line of lines) {
                if (line.startsWith("data: ")) {
                    const payload = JSON.parse(line.slice(6));
                    handleStreamEvent(payload);
                }
            }
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        hideLoading();
    }
}

let streamData = {};

function resetResults() {
    streamData = {};
    document.getElementById("input-stats").innerHTML = "";
    document.getElementById("extractive-summary").textContent = "";
    document.getElementById("extractive-meta").innerHTML = "";
    document.getElementById("abstractive-summary").textContent = "";
    document.getElementById("abstractive-meta").innerHTML = "";
    document.getElementById("rouge-table").innerHTML = "";
    document.getElementById("density-metrics").innerHTML = "";
    document.getElementById("coherence-metrics").innerHTML = "";
    document.getElementById("redundancy-metrics").innerHTML = "";
    document.getElementById("coverage-metrics").innerHTML = "";
    document.getElementById("overall-quality").innerHTML = "";
    document.getElementById("rankings").innerHTML = "";
    document.getElementById("topic-coverage").innerHTML = "";
    document.getElementById("explanations").innerHTML = "";
    // Hide metrics section until ready
    document.getElementById("metrics-section").classList.add("hidden");
    // Hide multi-model section, show 2-card grid
    document.getElementById("multi-model-section").classList.add("hidden");
    document.getElementById("summaries-grid").classList.remove("hidden");
    document.getElementById("model-accordions").innerHTML = "";
    document.getElementById("models-list").innerHTML = "";
    document.getElementById("performance-content").innerHTML = "";
    document.getElementById("multi-model-errors").innerHTML = "";
    // Reset scroll-reveal states
    document.querySelectorAll(".scroll-reveal").forEach(el => {
        el.classList.remove("revealed");
    });
    // Reset card visibility
    document.getElementById("extractive-card").style.display = "";
    document.getElementById("abstractive-card").style.display = "";
}

function handleStreamEvent(payload) {
    switch (payload.stage) {
        case "preprocessing":
            updateProgress("Text preprocessed. Generating extractive summary...", 20);
            showResults();
            renderInputStats(payload.data);
            break;

        case "extractive":
            updateProgress("Extractive summary ready. Generating abstractive summary...", 45);
            streamData.extractive = payload.data;
            renderExtractiveSummary(payload.data);
            break;

        case "abstractive":
            updateProgress("Abstractive summary ready. Computing metrics...", 70);
            streamData.abstractive = payload.data;
            renderAbstractiveSummary(payload.data);
            break;

        case "metrics":
            updateProgress("Metrics computed!", 100);
            streamData.metrics = payload.data;
            renderMetrics(payload.data);
            break;

        case "done":
            hideLoading();
            break;

        case "error":
            hideLoading();
            alert(`Error: ${payload.data.message}`);
            break;
    }
}

// --- Non-streaming single method calls (kept for individual buttons) ---

async function summarizeExtractiveOnly() {
    const text = document.getElementById("input-text").value.trim();
    const ratio = parseFloat(ratioSlider.value);
    if (text.length < 50) { alert("Please enter at least 50 characters."); return; }

    collapseInput();
    showLoading();
    updateProgress("Generating extractive summary...", 30);
    resetResults();

    try {
        const response = await fetch(`${API_BASE}/summarize/extractive`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text, ratio }),
        });
        if (!response.ok) { const err = await response.json(); throw new Error(err.detail); }
        const data = await response.json();
        showResults();
        document.getElementById("abstractive-card").style.display = "none";
        renderExtractiveSummary(data);
        renderSingleMetrics(data.metrics, "extractive");
    } catch (error) { alert(`Error: ${error.message}`); }
    finally { hideLoading(); }
}

async function summarizeAbstractiveOnly() {
    const text = document.getElementById("input-text").value.trim();
    const ratio = parseFloat(ratioSlider.value);
    if (text.length < 50) { alert("Please enter at least 50 characters."); return; }

    collapseInput();
    showLoading();
    updateProgress("Loading model & generating abstractive summary...", 30);
    resetResults();

    try {
        const response = await fetch(`${API_BASE}/summarize/abstractive`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text, ratio }),
        });
        if (!response.ok) { const err = await response.json(); throw new Error(err.detail); }
        const data = await response.json();
        showResults();
        document.getElementById("extractive-card").style.display = "none";
        renderAbstractiveSummary(data);
        renderSingleMetrics(data.metrics, "abstractive");
    } catch (error) { alert(`Error: ${error.message}`); }
    finally { hideLoading(); }
}

// --- Compare All Models ---

async function compareAllModels() {
    const text = document.getElementById("input-text").value.trim();
    const reference = document.getElementById("reference-summary").value.trim();
    const ratio = parseFloat(ratioSlider.value);

    if (text.length < 50) { alert("Please enter at least 50 characters."); return; }

    collapseInput();
    showLoading();
    updateProgress("Running all available models...", 10);

    try {
        const response = await fetch(`${API_BASE}/compare/all`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text, ratio, reference_summary: reference || null }),
        });

        if (!response.ok) { const err = await response.json(); throw new Error(err.detail); }
        const data = await response.json();

        updateProgress("All models complete!", 100);
        showResults();

        // Hide the 2-card grid, show multi-model section
        document.getElementById("summaries-grid").classList.add("hidden");
        document.getElementById("multi-model-section").classList.remove("hidden");

        renderMultiModelResults(data);

        // Show metrics section if we have rankings
        if (data.rankings && Object.keys(data.rankings).length > 0) {
            document.getElementById("metrics-section").classList.remove("hidden");
            renderMultiModelMetrics(data);
            initScrollReveal();
        }
    } catch (error) { alert(`Error: ${error.message}`); }
    finally { hideLoading(); }
}

function renderMultiModelResults(data) {
    const errorContainer = document.getElementById("multi-model-errors");

    // Input stats
    const langNames = {en: "English", pl: "Polish", tr: "Turkish", az: "Azerbaijani"};
    const langDisplay = langNames[data.input_stats.language] || data.input_stats.language;
    document.getElementById("input-stats").innerHTML = `
        <div class="stat-item"><span class="stat-label">Language</span><span class="stat-value">${langDisplay}</span></div>
        <div class="stat-item"><span class="stat-label">Words</span><span class="stat-value">${data.input_stats.word_count}</span></div>
        <div class="stat-item"><span class="stat-label">Sentences</span><span class="stat-value">${data.input_stats.sentence_count}</span></div>
        <div class="stat-item"><span class="stat-label">Models Used</span><span class="stat-value">${data.models_used}</span></div>
    `;

    const modelColors = {
        extractive: "#27ae60",
        distilbart: "#e67e22",
        "bart-large": "#e74c3c",
        pegasus: "#9b59b6",
        "mt5-xlsum": "#3498db",
        "led-large-book": "#e91e63",
        "flan-t5-large": "#00bcd4",
        "turkish-mbart": "#ff5722",
        "gpt-4": "#1a1a2e",
        "gpt-4o-mini": "#2c3e50",
        "claude-sonnet": "#d4a574",
        "claude-haiku": "#8e7cc3",
    };

    const modelTypes = {
        extractive: "Extractive (Graph-based)",
        distilbart: "Abstractive (Local HuggingFace)",
        "bart-large": "Abstractive (Local HuggingFace)",
        pegasus: "Abstractive (Local HuggingFace)",
        "mt5-xlsum": "Abstractive (Multilingual HuggingFace)",
        "led-large-book": "Abstractive (GPU — Colab)",
        "flan-t5-large": "Abstractive (GPU — Colab)",
        "turkish-mbart": "Abstractive (GPU — Turkish)",
        "gpt-4": "Abstractive (Paid API)",
        "gpt-4o-mini": "Abstractive (Paid API)",
        "claude-sonnet": "Abstractive (Paid API)",
        "claude-haiku": "Abstractive (Paid API)",
    };

    // --- Models List ---
    let listHtml = "";
    for (const [key, result] of Object.entries(data.results)) {
        const color = modelColors[key] || "#555";
        const type = modelTypes[key] || "Unknown";
        listHtml += `
            <div class="model-list-item fade-in">
                <div class="model-list-dot" style="background: ${color};"></div>
                <div class="model-list-info">
                    <span class="model-list-name">${result.name}</span>
                    <span class="model-list-type">${type}</span>
                </div>
                <span class="model-list-status">✓</span>
            </div>
        `;
    }
    for (const [key, err] of Object.entries(data.errors)) {
        listHtml += `
            <div class="model-list-item model-list-error fade-in">
                <div class="model-list-dot" style="background: #e74c3c;"></div>
                <div class="model-list-info">
                    <span class="model-list-name">${err.name}</span>
                    <span class="model-list-type">Failed</span>
                </div>
                <span class="model-list-status">✗</span>
            </div>
        `;
    }
    document.getElementById("models-list").innerHTML = listHtml;

    // --- Accordion Results ---
    let accordionHtml = "";
    let i = 0;
    for (const [key, result] of Object.entries(data.results)) {
        const color = modelColors[key] || "#555";
        const quality = result.metrics?.overall_quality?.score || 0;
        const reliability = result.reliability || "unknown";
        const reliabilityBadge = reliability === "high"
            ? '<span class="reliability-badge high">HIGH</span>'
            : reliability === "medium"
            ? '<span class="reliability-badge medium">MEDIUM</span>'
            : '<span class="reliability-badge low">LOW</span>';

        const rougeF1 = result.metrics?.rouge_scores?.rougeL?.f1 || 0;
        const coverage = result.metrics?.coverage_score?.score || 0;
        const coherence = result.metrics?.coherence_score?.score || 0;

        // Build detailed metrics inside accordion
        let metricsHtml = "";
        if (result.metrics) {
            const m = result.metrics;
            const rouge = m.rouge_scores || {};
            const pct = (v) => ((v || 0) * 100).toFixed(1) + "%";

            metricsHtml = `
                <div class="accordion-detailed-metrics">
                    <h4>ROUGE Scores</h4>
                    <table class="accordion-rouge-table">
                        <thead><tr><th>Metric</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
                        <tbody>
                            <tr><td>ROUGE-1</td><td>${pct(rouge.rouge1?.precision)}</td><td>${pct(rouge.rouge1?.recall)}</td><td><strong>${pct(rouge.rouge1?.f1)}</strong></td></tr>
                            <tr><td>ROUGE-2</td><td>${pct(rouge.rouge2?.precision)}</td><td>${pct(rouge.rouge2?.recall)}</td><td><strong>${pct(rouge.rouge2?.f1)}</strong></td></tr>
                            <tr><td>ROUGE-L</td><td>${pct(rouge.rougeL?.precision)}</td><td>${pct(rouge.rougeL?.recall)}</td><td><strong>${pct(rouge.rougeL?.f1)}</strong></td></tr>
                        </tbody>
                    </table>

                    <div class="accordion-metrics-grid">
                        <div class="accordion-metric-item">
                            <span class="accordion-metric-label">Information Density</span>
                            <div class="accordion-metric-bar-track">
                                <div class="accordion-metric-bar-fill" style="width: ${((m.information_density?.score || 0) * 100)}%; background: ${color};"></div>
                            </div>
                            <span class="accordion-metric-value">${pct(m.information_density?.score)}</span>
                        </div>
                        <div class="accordion-metric-item">
                            <span class="accordion-metric-label">Coherence</span>
                            <div class="accordion-metric-bar-track">
                                <div class="accordion-metric-bar-fill" style="width: ${((m.coherence_score?.score || 0) * 100)}%; background: ${color};"></div>
                            </div>
                            <span class="accordion-metric-value">${pct(m.coherence_score?.score)}</span>
                        </div>
                        <div class="accordion-metric-item">
                            <span class="accordion-metric-label">Redundancy</span>
                            <div class="accordion-metric-bar-track">
                                <div class="accordion-metric-bar-fill" style="width: ${((m.redundancy_score?.score || 0) * 100)}%; background: #e74c3c;"></div>
                            </div>
                            <span class="accordion-metric-value">${pct(m.redundancy_score?.score)} <small>(lower=better)</small></span>
                        </div>
                        <div class="accordion-metric-item">
                            <span class="accordion-metric-label">Coverage</span>
                            <div class="accordion-metric-bar-track">
                                <div class="accordion-metric-bar-fill" style="width: ${((m.coverage_score?.score || 0) * 100)}%; background: ${color};"></div>
                            </div>
                            <span class="accordion-metric-value">${pct(m.coverage_score?.score)}</span>
                        </div>
                    </div>

                    ${m.topic_coverage ? `
                    <h4>Topic Coverage (${m.topic_coverage.covered_topics?.length || 0}/${m.topic_coverage.total_topics || 0})</h4>
                    <div class="accordion-topics">
                        <div class="topic-tags">
                            ${(m.topic_coverage.covered_topics || []).map(t => `<span class="topic-tag covered">✓ ${t}</span>`).join("")}
                            ${(m.topic_coverage.missing_topics || []).map(t => `<span class="topic-tag missing">✗ ${t}</span>`).join("")}
                        </div>
                    </div>
                    ` : ""}

                    <h4>Overall Quality: ${(quality * 100).toFixed(1)}%</h4>
                    ${m.overall_quality?.component_scores ? `
                    <div class="accordion-components">
                        ${Object.entries(m.overall_quality.component_scores).map(([k, v]) => `
                            <div class="accordion-component-row">
                                <span>${k.replace(/_/g, " ")}</span>
                                <strong>${((v || 0) * 100).toFixed(1)}%</strong>
                            </div>
                        `).join("")}
                    </div>
                    ` : ""}

                    ${m.explanations ? `
                    <details class="accordion-explanations">
                        <summary>📊 Why These Scores?</summary>
                        <div class="accordion-explanations-content">
                            ${Object.entries(m.explanations).map(([k, v]) => `
                                <div class="accordion-explanation-item">
                                    <strong>${k.replace(/_/g, " ")}</strong>
                                    <p>${v}</p>
                                </div>
                            `).join("")}
                        </div>
                    </details>
                    ` : ""}
                </div>
            `;
        }

        accordionHtml += `
            <div class="accordion-item fade-in" style="animation-delay: ${i * 0.08}s;">
                <button class="accordion-header" onclick="toggleAccordion(this)" style="border-left-color: ${color};">
                    <div class="accordion-title">
                        <span class="accordion-model-name" style="color: ${color};">${result.name}</span>
                        <div class="accordion-badges">
                            ${reliabilityBadge}
                            <span class="quality-badge" style="background: ${color};">${(quality * 100).toFixed(0)}%</span>
                        </div>
                    </div>
                    <span class="accordion-arrow">▼</span>
                </button>
                <div class="accordion-body">
                    <div class="accordion-summary">${result.summary}</div>
                    <div class="accordion-metrics">
                        <div class="acc-metric"><span>ROUGE-L F1</span><strong>${(rougeF1 * 100).toFixed(1)}%</strong></div>
                        <div class="acc-metric"><span>Coverage</span><strong>${(coverage * 100).toFixed(1)}%</strong></div>
                        <div class="acc-metric"><span>Coherence</span><strong>${(coherence * 100).toFixed(1)}%</strong></div>
                        <div class="acc-metric"><span>Compression</span><strong>${(result.compression_ratio * 100).toFixed(1)}%</strong></div>
                    </div>
                    ${metricsHtml}
                    ${result.warning ? `<div class="summary-warning">${result.warning}</div>` : ""}
                </div>
            </div>
        `;
        i++;
    }
    document.getElementById("model-accordions").innerHTML = accordionHtml;

    // --- Performance Summary ---
    renderPerformanceSummary(data);

    // Errors
    if (Object.keys(data.errors).length > 0) {
        let errHtml = '<div class="errors-card"><h4>⚠️ Failed Models</h4>';
        for (const [key, err] of Object.entries(data.errors)) {
            errHtml += `<p><strong>${err.name}:</strong> ${err.error}</p>`;
        }
        errHtml += "</div>";
        errorContainer.innerHTML = errHtml;
    } else {
        errorContainer.innerHTML = "";
    }
}

function toggleAccordion(btn) {
    const item = btn.parentElement;
    item.classList.toggle("open");
}

function renderPerformanceSummary(data) {
    const results = data.results;
    const keys = Object.keys(results);
    if (keys.length === 0) return;

    // Find best and worst by overall quality
    let best = { key: "", score: -1 };
    let worst = { key: "", score: 2 };
    const summaryLengths = {};
    const topicsCovered = {};

    for (const [key, result] of Object.entries(results)) {
        const quality = result.metrics?.overall_quality?.score || 0;
        if (quality > best.score) { best = { key, score: quality, name: result.name }; }
        if (quality < worst.score) { worst = { key, score: quality, name: result.name }; }
        summaryLengths[key] = result.summary.split(" ").length;
        topicsCovered[key] = result.metrics?.topic_coverage?.covered_topics?.length || 0;
    }

    const totalTopics = results[keys[0]]?.metrics?.topic_coverage?.total_topics || 0;

    // Find topics that ALL models missed
    const allMissing = results[keys[0]]?.metrics?.topic_coverage?.missing_topics || [];
    const universallyMissed = allMissing.filter(topic =>
        keys.every(k => results[k]?.metrics?.topic_coverage?.missing_topics?.includes(topic))
    );

    // Find topics only the best model caught
    const bestTopics = results[best.key]?.metrics?.topic_coverage?.covered_topics || [];
    const uniqueToBest = bestTopics.filter(topic =>
        keys.filter(k => k !== best.key).every(k =>
            results[k]?.metrics?.topic_coverage?.missing_topics?.includes(topic)
        )
    );

    let html = `
        <div class="perf-grid">
            <div class="perf-card best">
                <div class="perf-icon">🏆</div>
                <div class="perf-label">Best Model</div>
                <div class="perf-value">${best.name}</div>
                <div class="perf-detail">Quality: ${(best.score * 100).toFixed(1)}%</div>
            </div>
            <div class="perf-card worst">
                <div class="perf-icon">📉</div>
                <div class="perf-label">Weakest Model</div>
                <div class="perf-value">${worst.name}</div>
                <div class="perf-detail">Quality: ${(worst.score * 100).toFixed(1)}%</div>
            </div>
            <div class="perf-card topics">
                <div class="perf-icon">📊</div>
                <div class="perf-label">Topic Coverage Range</div>
                <div class="perf-value">${Math.min(...Object.values(topicsCovered))} – ${Math.max(...Object.values(topicsCovered))} / ${totalTopics}</div>
                <div class="perf-detail">topics covered across models</div>
            </div>
        </div>
    `;

    if (universallyMissed.length > 0) {
        html += `
            <div class="perf-finding">
                <h4>❌ Topics ALL models missed:</h4>
                <div class="topic-tags">
                    ${universallyMissed.map(t => `<span class="topic-tag missing">✗ ${t}</span>`).join("")}
                </div>
            </div>
        `;
    }

    if (uniqueToBest.length > 0) {
        html += `
            <div class="perf-finding">
                <h4>⭐ Topics only ${best.name} captured:</h4>
                <div class="topic-tags">
                    ${uniqueToBest.map(t => `<span class="topic-tag covered">✓ ${t}</span>`).join("")}
                </div>
            </div>
        `;
    }

    html += `
        <div class="perf-finding">
            <h4>📏 Summary Lengths:</h4>
            <div class="length-comparison">
                ${Object.entries(results).map(([key, r]) => `
                    <div class="length-item">
                        <span>${r.name.split("(")[0].trim()}</span>
                        <div class="length-bar-track">
                            <div class="length-bar-fill" style="width: ${Math.min((summaryLengths[key] / Math.max(...Object.values(summaryLengths))) * 100, 100)}%; background: ${key === best.key ? '#27ae60' : '#4a90d9'};"></div>
                        </div>
                        <span class="length-words">${summaryLengths[key]} words</span>
                    </div>
                `).join("")}
            </div>
        </div>
    `;

    document.getElementById("performance-content").innerHTML = html;
}

function renderMultiModelMetrics(data) {
    const results = data.results;
    const keys = Object.keys(results);

    // Show metrics section
    const metricsSection = document.getElementById("metrics-section");
    metricsSection.classList.remove("hidden");

    // --- ROUGE Table for all models ---
    if (keys.length >= 2 && results[keys[0]]?.metrics?.rouge_scores) {
        renderMultiModelRouge(results);
    }

    // --- Quality metrics comparison bars ---
    renderMultiModelQualityBars(results);

    // --- Overall Quality scores ---
    renderMultiModelOverallQuality(results);

    // --- Rankings with analysis ---
    if (data.rankings) {
        renderMultiModelRankings(data.rankings, results);
    }

    // --- Topic coverage comparison ---
    renderMultiModelTopicCoverage(results);

    // --- Detailed analysis: why each model ranked where it did ---
    renderMultiModelAnalysis(results, data.rankings);

    initScrollReveal();
}

function renderMultiModelQualityBars(results) {
    const keys = Object.keys(results);
    const modelColors = {
        extractive: "#27ae60", distilbart: "#e67e22", "bart-large": "#e74c3c",
        pegasus: "#9b59b6", "mt5-xlsum": "#3498db", "led-large-book": "#e91e63",
        "flan-t5-large": "#00bcd4", "turkish-mbart": "#ff5722", "gpt-4": "#1a1a2e", "gpt-4o-mini": "#2c3e50",
        "claude-sonnet": "#d4a574", "claude-haiku": "#8e7cc3",
    };

    const metricsMap = {
        information_density: { label: "Information Density", target: "density-metrics" },
        coherence_score: { label: "Coherence", target: "coherence-metrics" },
        redundancy_score: { label: "Redundancy (lower = better)", target: "redundancy-metrics" },
        coverage_score: { label: "Coverage", target: "coverage-metrics" },
    };

    for (const [metric, config] of Object.entries(metricsMap)) {
        let html = "";
        for (const key of keys) {
            const score = results[key]?.metrics?.[metric]?.score || 0;
            const color = modelColors[key] || "#555";
            const name = results[key].name.split("(")[0].trim();
            html += `
                <div class="score-bar">
                    <span class="score-bar-label">${name}</span>
                    <div class="score-bar-track">
                        <div class="score-bar-fill" style="background: ${color}; width: ${(score * 100)}%;"></div>
                    </div>
                    <span class="score-bar-value">${(score * 100).toFixed(1)}%</span>
                </div>
            `;
        }
        document.getElementById(config.target).innerHTML = html;
    }
}

function renderMultiModelOverallQuality(results) {
    const keys = Object.keys(results);
    const modelColors = {
        extractive: "#27ae60", distilbart: "#e67e22", "bart-large": "#e74c3c",
        pegasus: "#9b59b6", "mt5-xlsum": "#3498db", "led-large-book": "#e91e63",
        "flan-t5-large": "#00bcd4", "turkish-mbart": "#ff5722", "gpt-4": "#1a1a2e", "gpt-4o-mini": "#2c3e50",
        "claude-sonnet": "#d4a574", "claude-haiku": "#8e7cc3",
    };

    let cardsHtml = '<div class="quality-scores multi-quality-scores">';
    for (const key of keys) {
        const quality = results[key]?.metrics?.overall_quality?.score || 0;
        const color = modelColors[key] || "#555";
        const name = results[key].name.split("(")[0].trim();
        cardsHtml += `
            <div class="quality-score-card" style="background: linear-gradient(135deg, ${color}22, ${color}11); border: 1px solid ${color}44;">
                <div class="quality-number">${(quality * 100).toFixed(1)}%</div>
                <div class="quality-label">${name}</div>
            </div>
        `;
    }
    cardsHtml += '</div>';

    // Component scores comparison table
    const firstMetrics = results[keys[0]]?.metrics?.overall_quality?.component_scores;
    if (firstMetrics) {
        cardsHtml += '<h4 style="margin: 1.5rem 0 0.5rem;">Component Scores</h4>';
        cardsHtml += '<table class="component-comparison-table"><thead><tr><th>Component</th>';
        for (const key of keys) {
            cardsHtml += `<th>${results[key].name.split("(")[0].trim()}</th>`;
        }
        cardsHtml += '</tr></thead><tbody>';
        for (const component of Object.keys(firstMetrics)) {
            cardsHtml += `<tr><td><strong>${component.replace(/_/g, " ")}</strong></td>`;
            for (const key of keys) {
                const val = results[key]?.metrics?.overall_quality?.component_scores?.[component] || 0;
                cardsHtml += `<td>${(val * 100).toFixed(1)}%</td>`;
            }
            cardsHtml += '</tr>';
        }
        cardsHtml += '</tbody></table>';
    }

    document.getElementById("overall-quality").innerHTML = cardsHtml;
}

function renderMultiModelRankings(rankings, results) {
    const keys = Object.keys(results);
    const modelColors = {
        extractive: "#27ae60", distilbart: "#e67e22", "bart-large": "#e74c3c",
        pegasus: "#9b59b6", "mt5-xlsum": "#3498db", "led-large-book": "#e91e63",
        "flan-t5-large": "#00bcd4", "turkish-mbart": "#ff5722", "gpt-4": "#1a1a2e", "gpt-4o-mini": "#2c3e50",
        "claude-sonnet": "#d4a574", "claude-haiku": "#8e7cc3",
    };

    let html = '<div class="multi-rankings">';

    // Rankings table
    html += '<table class="rankings-table"><thead><tr><th>Criterion</th>';
    // Get ranking positions: 1st, 2nd, 3rd...
    const maxModels = Math.min(keys.length, 5);
    for (let p = 1; p <= maxModels; p++) {
        html += `<th>${p === 1 ? "🥇" : p === 2 ? "🥈" : p === 3 ? "🥉" : `#${p}`}</th>`;
    }
    html += '</tr></thead><tbody>';

    for (const [criterion, ranking] of Object.entries(rankings)) {
        const label = criterion.replace(/by_/g, "").replace(/_/g, " ");
        html += `<tr><td><strong>${label.charAt(0).toUpperCase() + label.slice(1)}</strong></td>`;
        for (let p = 0; p < maxModels && p < ranking.length; p++) {
            const [modelKey, score] = ranking[p];
            const name = results[modelKey]?.name?.split("(")[0]?.trim() || modelKey;
            const color = modelColors[modelKey] || "#555";
            html += `<td><span style="color: ${color}; font-weight: 600;">${name}</span><br><small>${(score * 100).toFixed(1)}%</small></td>`;
        }
        html += '</tr>';
    }
    html += '</tbody></table>';

    // Bar chart comparison across all criteria
    html += '<div class="multi-bar-chart">';
    for (const [criterion, ranking] of Object.entries(rankings)) {
        const label = criterion.replace(/by_/g, "").replace(/_/g, " ");
        html += `<div class="multi-bar-group"><div class="bar-label">${label.charAt(0).toUpperCase() + label.slice(1)}</div>`;
        for (const [modelKey, score] of ranking) {
            const name = results[modelKey]?.name?.split("(")[0]?.trim() || modelKey;
            const color = modelColors[modelKey] || "#555";
            html += `
                <div class="bar-row">
                    <span class="bar-row-name">${name}</span>
                    <div class="bar-track">
                        <div class="bar-fill" style="width: ${(score * 100)}%; background: ${color};"></div>
                    </div>
                    <span class="bar-value">${(score * 100).toFixed(1)}%</span>
                </div>
            `;
        }
        html += '</div>';
    }
    html += '</div></div>';

    document.getElementById("rankings").innerHTML = html;
}

function renderMultiModelTopicCoverage(results) {
    const keys = Object.keys(results);
    const modelColors = {
        extractive: "#27ae60", distilbart: "#e67e22", "bart-large": "#e74c3c",
        pegasus: "#9b59b6", "mt5-xlsum": "#3498db", "led-large-book": "#e91e63",
        "flan-t5-large": "#00bcd4", "turkish-mbart": "#ff5722", "gpt-4": "#1a1a2e", "gpt-4o-mini": "#2c3e50",
        "claude-sonnet": "#d4a574", "claude-haiku": "#8e7cc3",
    };

    const totalTopics = results[keys[0]]?.metrics?.topic_coverage?.total_topics || 0;

    let html = '<div class="multi-topic-coverage">';
    html += '<div class="topic-summary-row multi-topic-summary">';
    for (const key of keys) {
        const tc = results[key]?.metrics?.topic_coverage;
        if (!tc) continue;
        const color = modelColors[key] || "#555";
        const name = results[key].name.split("(")[0].trim();
        html += `
            <div class="topic-score" style="border-color: ${color};">
                <span class="topic-score-number">${tc.covered_topics?.length || 0}</span>/<span>${totalTopics}</span>
                <span class="topic-score-label">${name}</span>
            </div>
        `;
    }
    html += '</div>';

    // Per-model topic details (expanded by default)
    for (const key of keys) {
        const tc = results[key]?.metrics?.topic_coverage;
        if (!tc) continue;
        const name = results[key].name.split("(")[0].trim();
        const color = modelColors[key] || "#555";
        html += `
            <details class="topic-model-detail" open>
                <summary style="color: ${color}; font-weight: 600;">${name} — ${tc.covered_topics?.length || 0}/${totalTopics} topics</summary>
                <div class="topic-tags">
                    ${(tc.covered_topics || []).map(t => `<span class="topic-tag covered">✓ ${t}</span>`).join("")}
                    ${(tc.missing_topics || []).map(t => `<span class="topic-tag missing">✗ ${t}</span>`).join("")}
                </div>
            </details>
        `;
    }
    html += '</div>';

    document.getElementById("topic-coverage").innerHTML = html;
}

function renderMultiModelAnalysis(results, rankings) {
    const keys = Object.keys(results);
    if (keys.length < 2) return;

    const modelColors = {
        extractive: "#27ae60", distilbart: "#e67e22", "bart-large": "#e74c3c",
        pegasus: "#9b59b6", "mt5-xlsum": "#3498db", "led-large-book": "#e91e63",
        "flan-t5-large": "#00bcd4", "turkish-mbart": "#ff5722", "gpt-4": "#1a1a2e", "gpt-4o-mini": "#2c3e50",
        "claude-sonnet": "#d4a574", "claude-haiku": "#8e7cc3",
    };

    // Sort models by overall quality
    const sorted = keys.map(k => ({
        key: k,
        name: results[k].name,
        quality: results[k]?.metrics?.overall_quality?.score || 0,
        rouge: results[k]?.metrics?.rouge_scores?.rougeL?.f1 || 0,
        density: results[k]?.metrics?.information_density?.score || 0,
        coherence: results[k]?.metrics?.coherence_score?.score || 0,
        coverage: results[k]?.metrics?.coverage_score?.score || 0,
        redundancy: results[k]?.metrics?.redundancy_score?.score || 0,
        topics: results[k]?.metrics?.topic_coverage?.covered_topics?.length || 0,
        totalTopics: results[k]?.metrics?.topic_coverage?.total_topics || 0,
    })).sort((a, b) => b.quality - a.quality);

    const best = sorted[0];
    const worst = sorted[sorted.length - 1];

    let html = '<div class="model-analysis">';

    // Winner announcement
    html += `
        <div class="analysis-winner">
            <h4>🏆 Best Performing Model: <span style="color: ${modelColors[best.key] || '#555'};">${best.name}</span></h4>
            <p>Scored <strong>${(best.quality * 100).toFixed(1)}%</strong> overall quality, beating the second place by 
            <strong>${((best.quality - (sorted[1]?.quality || 0)) * 100).toFixed(1)}pp</strong>.</p>
        </div>
    `;

    // Strengths/weaknesses for each model
    html += '<h4 style="margin-top: 1.5rem;">Model Analysis</h4>';
    for (const model of sorted) {
        const color = modelColors[model.key] || "#555";
        const strengths = [];
        const weaknesses = [];

        // Determine strengths relative to average
        const avgRouge = sorted.reduce((s, m) => s + m.rouge, 0) / sorted.length;
        const avgDensity = sorted.reduce((s, m) => s + m.density, 0) / sorted.length;
        const avgCoherence = sorted.reduce((s, m) => s + m.coherence, 0) / sorted.length;
        const avgCoverage = sorted.reduce((s, m) => s + m.coverage, 0) / sorted.length;
        const avgRedundancy = sorted.reduce((s, m) => s + m.redundancy, 0) / sorted.length;

        if (model.rouge > avgRouge * 1.1) strengths.push("High ROUGE scores (good word overlap with reference)");
        if (model.density > avgDensity * 1.1) strengths.push("High information density (retains key terms)");
        if (model.coherence > avgCoherence * 1.1) strengths.push("Good coherence (smooth sentence transitions)");
        if (model.coverage > avgCoverage * 1.1) strengths.push("Strong source coverage");
        if (model.redundancy < avgRedundancy * 0.9) strengths.push("Low redundancy (no repetition)");
        if (model.topics >= model.totalTopics * 0.8) strengths.push(`Excellent topic coverage (${model.topics}/${model.totalTopics})`);

        if (model.rouge < avgRouge * 0.9) weaknesses.push("Below-average ROUGE scores");
        if (model.density < avgDensity * 0.9) weaknesses.push("Low information density (may miss key terms)");
        if (model.coherence < avgCoherence * 0.9) weaknesses.push("Low coherence (disjointed sentences)");
        if (model.coverage < avgCoverage * 0.9) weaknesses.push("Poor source coverage");
        if (model.redundancy > avgRedundancy * 1.1) weaknesses.push("High redundancy (repetitive content)");
        if (model.topics < model.totalTopics * 0.5) weaknesses.push(`Low topic coverage (${model.topics}/${model.totalTopics})`);

        if (strengths.length === 0) strengths.push("Average performance across metrics");
        if (weaknesses.length === 0) weaknesses.push("No significant weaknesses");

        html += `
            <div class="analysis-model-card" style="border-left: 4px solid ${color};">
                <div class="analysis-model-header">
                    <span style="color: ${color}; font-weight: 700;">${model.name}</span>
                    <span class="analysis-quality-badge" style="background: ${color};">${(model.quality * 100).toFixed(1)}%</span>
                </div>
                <div class="analysis-strengths-weaknesses">
                    <div class="analysis-strengths">
                        <strong>✅ Strengths:</strong>
                        <ul>${strengths.map(s => `<li>${s}</li>`).join("")}</ul>
                    </div>
                    <div class="analysis-weaknesses">
                        <strong>⚠️ Weaknesses:</strong>
                        <ul>${weaknesses.map(w => `<li>${w}</li>`).join("")}</ul>
                    </div>
                </div>
            </div>
        `;
    }

    html += '</div>';

    document.getElementById("explanations").innerHTML = html;
}

function renderMultiModelRouge(results) {
    const pct = (v) => (v * 100).toFixed(1) + "%";
    let headerHtml = "<th>Metric</th>";
    let rows = { rouge1: "<td><strong>ROUGE-1 F1</strong></td>", rouge2: "<td><strong>ROUGE-2 F1</strong></td>", rougeL: "<td><strong>ROUGE-L F1</strong></td>" };

    for (const [key, result] of Object.entries(results)) {
        if (!result.metrics?.rouge_scores) continue;
        const rouge = result.metrics.rouge_scores;
        headerHtml += `<th>${result.name.split("(")[0].trim()}</th>`;
        rows.rouge1 += `<td><span class="rouge-num" data-target="${(rouge.rouge1.f1 * 100).toFixed(1)}" data-suffix="%">0%</span></td>`;
        rows.rouge2 += `<td><span class="rouge-num" data-target="${(rouge.rouge2.f1 * 100).toFixed(1)}" data-suffix="%">0%</span></td>`;
        rows.rougeL += `<td><span class="rouge-num" data-target="${(rouge.rougeL.f1 * 100).toFixed(1)}" data-suffix="%">0%</span></td>`;
    }

    const html = `
        <table class="fade-in rouge-table">
            <thead><tr>${headerHtml}</tr></thead>
            <tbody>
                <tr>${rows.rouge1}</tr>
                <tr>${rows.rouge2}</tr>
                <tr>${rows.rougeL}</tr>
            </tbody>
        </table>
    `;
    document.getElementById("rouge-table").innerHTML = html;

    setTimeout(() => {
        document.querySelectorAll("#rouge-table .rouge-num").forEach((el, i) => {
            setTimeout(() => animateNumber(el), i * 50);
        });
    }, 300);
}

// --- Render Functions ---

function renderInputStats(data) {
    const langNames = {en: "English", pl: "Polish", tr: "Turkish", az: "Azerbaijani", de: "German", fr: "French", es: "Spanish", ru: "Russian"};
    const langDisplay = langNames[data.language] || data.language || "Unknown";
    const statsHtml = `
        <div class="stat-item"><span class="stat-label">Language</span><span class="stat-value">${langDisplay}</span></div>
        <div class="stat-item"><span class="stat-label">Words</span><span class="stat-value" data-target="${data.word_count}">0</span></div>
        <div class="stat-item"><span class="stat-label">Sentences</span><span class="stat-value" data-target="${data.sentence_count}">0</span></div>
        <div class="stat-item"><span class="stat-label">Chunks</span><span class="stat-value" data-target="${data.chunk_count}">0</span></div>
    `;
    document.getElementById("input-stats").innerHTML = statsHtml;
    document.querySelectorAll("#input-stats .stat-value[data-target]").forEach(animateNumber);
}

function renderExtractiveSummary(data) {
    const el = document.getElementById("extractive-summary");
    typewriterEffect(el, data.summary);
    document.getElementById("extractive-meta").innerHTML = `
        Sentences: ${data.num_sentences_selected} / ${data.num_sentences_total} |
        Compression: ${(data.compression_ratio * 100).toFixed(1)}%
    `;
    document.getElementById("extractive-card").classList.add("fade-in");
}

function renderAbstractiveSummary(data) {
    const el = document.getElementById("abstractive-summary");
    typewriterEffect(el, data.summary);
    let metaHtml = `Model: ${data.model_id} | Compression: ${(data.compression_ratio * 100).toFixed(1)}%`;
    if (data.warning) {
        metaHtml += `<div class="summary-warning">${data.warning}</div>`;
    }
    document.getElementById("abstractive-meta").innerHTML = metaHtml;
    document.getElementById("abstractive-card").classList.add("fade-in");
}

function renderMetrics(comparison) {
    const extMetrics = comparison.individual_results.extractive_textrank;
    const absMetrics = comparison.individual_results.abstractive_transformer;

    // Show the metrics section
    const metricsSection = document.getElementById("metrics-section");
    metricsSection.classList.remove("hidden");

    renderRougeTable(extMetrics, absMetrics);
    renderQualityMetrics(extMetrics, absMetrics);
    renderOverallQuality(extMetrics, absMetrics);
    renderRankings(comparison.rankings);
    renderTopicCoverage(extMetrics, absMetrics);
    renderExplanations(extMetrics, absMetrics);

    // Activate scroll-reveal observer for metric cards
    initScrollReveal();
}

function renderRougeTable(extMetrics, absMetrics) {
    const extRouge = extMetrics.rouge_scores;
    const absRouge = absMetrics.rouge_scores;

    const pct = (v) => (v * 100).toFixed(1) + "%";

    const html = `
        <table class="fade-in rouge-table">
            <thead>
                <tr>
                    <th></th>
                    <th colspan="3" class="ext-header">Extractive</th>
                    <th colspan="3" class="abs-header">Abstractive</th>
                </tr>
                <tr>
                    <th>Metric</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>ROUGE-1</strong></td>
                    <td class="ext-cell"><span class="rouge-num" data-target="${(extRouge.rouge1.precision * 100).toFixed(1)}" data-suffix="%">0%</span></td>
                    <td class="ext-cell"><span class="rouge-num" data-target="${(extRouge.rouge1.recall * 100).toFixed(1)}" data-suffix="%">0%</span></td>
                    <td class="ext-cell ext-f1"><strong><span class="rouge-num" data-target="${(extRouge.rouge1.f1 * 100).toFixed(1)}" data-suffix="%">0%</span></strong></td>
                    <td class="abs-cell"><span class="rouge-num" data-target="${(absRouge.rouge1.precision * 100).toFixed(1)}" data-suffix="%">0%</span></td>
                    <td class="abs-cell"><span class="rouge-num" data-target="${(absRouge.rouge1.recall * 100).toFixed(1)}" data-suffix="%">0%</span></td>
                    <td class="abs-cell abs-f1"><strong><span class="rouge-num" data-target="${(absRouge.rouge1.f1 * 100).toFixed(1)}" data-suffix="%">0%</span></strong></td>
                </tr>
                <tr>
                    <td><strong>ROUGE-2</strong></td>
                    <td class="ext-cell"><span class="rouge-num" data-target="${(extRouge.rouge2.precision * 100).toFixed(1)}" data-suffix="%">0%</span></td>
                    <td class="ext-cell"><span class="rouge-num" data-target="${(extRouge.rouge2.recall * 100).toFixed(1)}" data-suffix="%">0%</span></td>
                    <td class="ext-cell ext-f1"><strong><span class="rouge-num" data-target="${(extRouge.rouge2.f1 * 100).toFixed(1)}" data-suffix="%">0%</span></strong></td>
                    <td class="abs-cell"><span class="rouge-num" data-target="${(absRouge.rouge2.precision * 100).toFixed(1)}" data-suffix="%">0%</span></td>
                    <td class="abs-cell"><span class="rouge-num" data-target="${(absRouge.rouge2.recall * 100).toFixed(1)}" data-suffix="%">0%</span></td>
                    <td class="abs-cell abs-f1"><strong><span class="rouge-num" data-target="${(absRouge.rouge2.f1 * 100).toFixed(1)}" data-suffix="%">0%</span></strong></td>
                </tr>
                <tr>
                    <td><strong>ROUGE-L</strong></td>
                    <td class="ext-cell"><span class="rouge-num" data-target="${(extRouge.rougeL.precision * 100).toFixed(1)}" data-suffix="%">0%</span></td>
                    <td class="ext-cell"><span class="rouge-num" data-target="${(extRouge.rougeL.recall * 100).toFixed(1)}" data-suffix="%">0%</span></td>
                    <td class="ext-cell ext-f1"><strong><span class="rouge-num" data-target="${(extRouge.rougeL.f1 * 100).toFixed(1)}" data-suffix="%">0%</span></strong></td>
                    <td class="abs-cell"><span class="rouge-num" data-target="${(absRouge.rougeL.precision * 100).toFixed(1)}" data-suffix="%">0%</span></td>
                    <td class="abs-cell"><span class="rouge-num" data-target="${(absRouge.rougeL.recall * 100).toFixed(1)}" data-suffix="%">0%</span></td>
                    <td class="abs-cell abs-f1"><strong><span class="rouge-num" data-target="${(absRouge.rougeL.f1 * 100).toFixed(1)}" data-suffix="%">0%</span></strong></td>
                </tr>
            </tbody>
        </table>
    `;
    const container = document.getElementById("rouge-table");
    container.innerHTML = html;

    // Animate numbers — only if card is already visible
    const card = container.closest(".scroll-reveal");
    if (!card || card.classList.contains("revealed")) {
        setTimeout(() => {
            container.querySelectorAll(".rouge-num").forEach((el, i) => {
                setTimeout(() => animateNumber(el), i * 50);
            });
        }, 200);
    }
}

function renderQualityMetrics(extMetrics, absMetrics) {
    renderAnimatedScoreComparison("density-metrics",
        "Extractive", extMetrics.information_density.score,
        "Abstractive", absMetrics.information_density.score
    );
    renderAnimatedScoreComparison("coherence-metrics",
        "Extractive", extMetrics.coherence_score.score,
        "Abstractive", absMetrics.coherence_score.score
    );

    const redHtml = `
        ${createAnimatedScoreBar("Extractive", extMetrics.redundancy_score.score, "extractive-color")}
        ${createAnimatedScoreBar("Abstractive", absMetrics.redundancy_score.score, "abstractive-color")}
        <p style="font-size: 0.8rem; color: #666; margin-top: 0.5rem;">Lower is better (less redundancy)</p>
    `;
    const redEl = document.getElementById("redundancy-metrics");
    redEl.innerHTML = redHtml;
    const redCard = redEl.closest(".scroll-reveal");
    if (!redCard || redCard.classList.contains("revealed")) {
        animateAllBars(redEl);
    }

    renderAnimatedScoreComparison("coverage-metrics",
        "Extractive", extMetrics.coverage_score.score,
        "Abstractive", absMetrics.coverage_score.score
    );
}

function renderAnimatedScoreComparison(elementId, label1, score1, label2, score2) {
    const html = `
        ${createAnimatedScoreBar(label1, score1, "extractive-color")}
        ${createAnimatedScoreBar(label2, score2, "abstractive-color")}
    `;
    const el = document.getElementById(elementId);
    el.innerHTML = html;
    // Only animate immediately if parent is already revealed or not a scroll-reveal
    const card = el.closest(".scroll-reveal");
    if (!card || card.classList.contains("revealed")) {
        animateAllBars(el);
    }
}

function createAnimatedScoreBar(label, score, colorClass) {
    const percentage = Math.min(score * 100, 100);
    return `
        <div class="score-bar">
            <span class="score-bar-label">${label}</span>
            <div class="score-bar-track">
                <div class="score-bar-fill ${colorClass}" data-width="${percentage}" style="width: 0%"></div>
            </div>
            <span class="score-bar-value" data-target="${percentage}" data-suffix="%">0%</span>
        </div>
    `;
}

function animateAllBars(container) {
    setTimeout(() => {
        container.querySelectorAll(".score-bar-fill").forEach(bar => {
            bar.style.width = bar.dataset.width + "%";
        });
        container.querySelectorAll(".score-bar-value").forEach(animateNumber);
    }, 100);
}

function renderOverallQuality(extMetrics, absMetrics) {
    const extQuality = extMetrics.overall_quality;
    const absQuality = absMetrics.overall_quality;

    let html = `
        <div class="quality-scores">
            <div class="quality-score-card extractive-bg">
                <div class="quality-number" data-target="${(extQuality.score * 100).toFixed(1)}" data-suffix="%">0%</div>
                <div class="quality-label">Extractive</div>
            </div>
            <div class="quality-score-card abstractive-bg">
                <div class="quality-number" data-target="${(absQuality.score * 100).toFixed(1)}" data-suffix="%">0%</div>
                <div class="quality-label">Abstractive</div>
            </div>
        </div>
        <h4 style="margin: 1rem 0 0.5rem;">Component Scores</h4>
        <div class="component-scores">
    `;

    if (extQuality.component_scores) {
        for (const [key, value] of Object.entries(extQuality.component_scores)) {
            const absValue = absQuality.component_scores[key] || 0;
            html += `
                <div class="component-row">
                    <span class="component-name">${key.replace(/_/g, " ")}</span>
                    <span class="component-ext" data-target="${(value * 100).toFixed(1)}" data-suffix="%">0%</span>
                    <span class="component-vs">vs</span>
                    <span class="component-abs" data-target="${(absValue * 100).toFixed(1)}" data-suffix="%">0%</span>
                </div>
            `;
        }
    }

    html += `</div>`;
    const el = document.getElementById("overall-quality");
    el.innerHTML = html;

    // Only animate if already visible (scroll observer handles the rest)
    const card = el.closest(".scroll-reveal");
    if (!card || card.classList.contains("revealed")) {
        setTimeout(() => {
            el.querySelectorAll("[data-target]").forEach(animateNumber);
        }, 200);
    }
}

function renderRankings(rankings) {
    // Extract scores for each method across all criteria
    const criteria = [];
    const extScores = [];
    const absScores = [];

    for (const [criterion, ranking] of Object.entries(rankings)) {
        const label = criterion.replace(/by_/g, "").replace(/_/g, " ");
        criteria.push(label.charAt(0).toUpperCase() + label.slice(1));
        // Find scores for each method
        let extScore = 0, absScore = 0;
        ranking.forEach(([method, score]) => {
            if (method.includes("extractive")) extScore = score;
            if (method.includes("abstractive")) absScore = score;
        });
        extScores.push(extScore);
        absScores.push(absScore);
    }

    const html = `
        <div class="rankings-visual">
            <div class="radar-chart-container">
                <canvas id="radar-chart" width="320" height="320"></canvas>
            </div>
            <div class="bar-chart-container">
                <div class="bar-chart-legend">
                    <span class="legend-item"><span class="legend-dot extractive-color"></span> Extractive</span>
                    <span class="legend-item"><span class="legend-dot abstractive-color"></span> Abstractive</span>
                </div>
                <div class="bar-chart" id="bar-chart"></div>
            </div>
        </div>
    `;
    document.getElementById("rankings").innerHTML = html;

    // Draw radar chart
    setTimeout(() => drawRadarChart(criteria, extScores, absScores), 200);

    // Draw bar chart
    renderBarChart(criteria, extScores, absScores);
}

function drawRadarChart(labels, extScores, absScores) {
    const canvas = document.getElementById("radar-chart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    const radius = Math.min(cx, cy) - 40;
    const n = labels.length;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw grid rings
    for (let ring = 1; ring <= 5; ring++) {
        const r = (radius / 5) * ring;
        ctx.beginPath();
        for (let i = 0; i <= n; i++) {
            const angle = (Math.PI * 2 * i) / n - Math.PI / 2;
            const x = cx + r * Math.cos(angle);
            const y = cy + r * Math.sin(angle);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = "#e0e0e0";
        ctx.lineWidth = 1;
        ctx.stroke();
    }

    // Draw axes and labels
    ctx.font = "11px -apple-system, sans-serif";
    ctx.fillStyle = "#555";
    ctx.textAlign = "center";
    for (let i = 0; i < n; i++) {
        const angle = (Math.PI * 2 * i) / n - Math.PI / 2;
        const x = cx + radius * Math.cos(angle);
        const y = cy + radius * Math.sin(angle);
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(x, y);
        ctx.strokeStyle = "#ddd";
        ctx.stroke();
        // Label
        const lx = cx + (radius + 25) * Math.cos(angle);
        const ly = cy + (radius + 25) * Math.sin(angle);
        ctx.fillText(labels[i], lx, ly + 4);
    }

    // Animate polygons growing from center
    let progress = 0;
    const duration = 60; // frames
    function animateFrame() {
        progress++;
        const t = Math.min(progress / duration, 1);
        // Ease out cubic
        const eased = 1 - Math.pow(1 - t, 3);

        // Clear only the polygon area (redraw grid would flicker, so redraw all)
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Redraw grid
        for (let ring = 1; ring <= 5; ring++) {
            const r = (radius / 5) * ring;
            ctx.beginPath();
            for (let i = 0; i <= n; i++) {
                const angle = (Math.PI * 2 * i) / n - Math.PI / 2;
                const x = cx + r * Math.cos(angle);
                const y = cy + r * Math.sin(angle);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.strokeStyle = "#e0e0e0";
            ctx.lineWidth = 1;
            ctx.stroke();
        }

        // Redraw axes and labels
        ctx.font = "11px -apple-system, sans-serif";
        ctx.fillStyle = "#555";
        ctx.textAlign = "center";
        for (let i = 0; i < n; i++) {
            const angle = (Math.PI * 2 * i) / n - Math.PI / 2;
            const x = cx + radius * Math.cos(angle);
            const y = cy + radius * Math.sin(angle);
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.lineTo(x, y);
            ctx.strokeStyle = "#ddd";
            ctx.stroke();
            const lx = cx + (radius + 25) * Math.cos(angle);
            const ly = cy + (radius + 25) * Math.sin(angle);
            ctx.fillText(labels[i], lx, ly + 4);
        }

        // Draw animated polygons
        const animExt = extScores.map(s => s * eased);
        const animAbs = absScores.map(s => s * eased);
        drawPolygon(ctx, cx, cy, radius, n, animExt, "rgba(39, 174, 96, 0.25)", "rgba(39, 174, 96, 0.9)");
        drawPolygon(ctx, cx, cy, radius, n, animAbs, "rgba(142, 68, 173, 0.2)", "rgba(142, 68, 173, 0.9)");

        if (t < 1) requestAnimationFrame(animateFrame);
    }
    requestAnimationFrame(animateFrame);
}

function drawPolygon(ctx, cx, cy, radius, n, scores, fillColor, strokeColor) {
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
        const angle = (Math.PI * 2 * i) / n - Math.PI / 2;
        const r = radius * scores[i];
        const x = cx + r * Math.cos(angle);
        const y = cy + r * Math.sin(angle);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fillStyle = fillColor;
    ctx.fill();
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw dots
    for (let i = 0; i < n; i++) {
        const angle = (Math.PI * 2 * i) / n - Math.PI / 2;
        const r = radius * scores[i];
        const x = cx + r * Math.cos(angle);
        const y = cy + r * Math.sin(angle);
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fillStyle = strokeColor;
        ctx.fill();
    }
}

function renderBarChart(labels, extScores, absScores) {
    const container = document.getElementById("bar-chart");
    if (!container) return;
    let html = "";
    labels.forEach((label, i) => {
        const extPct = (extScores[i] * 100).toFixed(1);
        const absPct = (absScores[i] * 100).toFixed(1);
        html += `
            <div class="bar-group bar-group-animated" style="animation-delay: ${i * 0.12}s">
                <div class="bar-label">${label}</div>
                <div class="bar-pair">
                    <div class="bar-row">
                        <div class="bar-track">
                            <div class="bar-fill extractive-color" data-width="${extPct}" style="width: 0%; transition-delay: ${i * 0.12 + 0.3}s"></div>
                        </div>
                        <span class="bar-value" data-target="${extPct}" data-suffix="%">0%</span>
                    </div>
                    <div class="bar-row">
                        <div class="bar-track">
                            <div class="bar-fill abstractive-color" data-width="${absPct}" style="width: 0%; transition-delay: ${i * 0.12 + 0.4}s"></div>
                        </div>
                        <span class="bar-value" data-target="${absPct}" data-suffix="%">0%</span>
                    </div>
                </div>
            </div>
        `;
    });
    container.innerHTML = html;

    // Animate bars and numbers with stagger
    setTimeout(() => {
        container.querySelectorAll(".bar-fill").forEach(bar => {
            bar.style.width = bar.dataset.width + "%";
        });
        container.querySelectorAll(".bar-value").forEach((el, i) => {
            setTimeout(() => animateNumber(el), i * 80);
        });
    }, 300);
}

function renderTopicCoverage(extMetrics, absMetrics) {
    const extTopics = extMetrics.topic_coverage;
    const absTopics = absMetrics.topic_coverage;

    let html = `
        <div class="topic-summary-row">
            <div class="topic-score extractive-topic-score">
                <span class="topic-score-number" data-target="${extTopics.covered_topics.length}">0</span>/<span>${extTopics.total_topics}</span>
                <span class="topic-score-label">Extractive</span>
            </div>
            <div class="topic-score abstractive-topic-score">
                <span class="topic-score-number" data-target="${absTopics.covered_topics.length}">0</span>/<span>${absTopics.total_topics}</span>
                <span class="topic-score-label">Abstractive</span>
            </div>
        </div>

        <h4 style="margin-top: 1.2rem;">Extractive</h4>
        <div class="topic-tags-section">
            <div class="topic-group">
                <span class="topic-group-label covered-label">✓ Covered (${extTopics.covered_topics.length})</span>
                <div class="topic-tags">
                    ${extTopics.covered_topics.map((t, i) => `<span class="topic-tag covered fade-in" style="animation-delay: ${i * 0.05}s">✓ ${t}</span>`).join("")}
                </div>
            </div>
            <div class="topic-group">
                <span class="topic-group-label missing-label">✗ Missing (${extTopics.missing_topics.length})</span>
                <div class="topic-tags">
                    ${extTopics.missing_topics.map((t, i) => `<span class="topic-tag missing fade-in" style="animation-delay: ${(extTopics.covered_topics.length + i) * 0.05}s">✗ ${t}</span>`).join("")}
                </div>
            </div>
        </div>

        <h4 style="margin-top: 1.2rem;">Abstractive</h4>
        <div class="topic-tags-section">
            <div class="topic-group">
                <span class="topic-group-label covered-label">✓ Covered (${absTopics.covered_topics.length})</span>
                <div class="topic-tags">
                    ${absTopics.covered_topics.map((t, i) => `<span class="topic-tag covered fade-in" style="animation-delay: ${i * 0.05}s">✓ ${t}</span>`).join("")}
                </div>
            </div>
            <div class="topic-group">
                <span class="topic-group-label missing-label">✗ Missing (${absTopics.missing_topics.length})</span>
                <div class="topic-tags">
                    ${absTopics.missing_topics.map((t, i) => `<span class="topic-tag missing fade-in" style="animation-delay: ${(absTopics.covered_topics.length + i) * 0.05}s">✗ ${t}</span>`).join("")}
                </div>
            </div>
        </div>
    `;
    const el = document.getElementById("topic-coverage");
    el.innerHTML = html;
    const card = el.closest(".scroll-reveal");
    if (!card || card.classList.contains("revealed")) {
        el.querySelectorAll("[data-target]").forEach(animateNumber);
    }
}

function renderExplanations(extMetrics, absMetrics) {
    const container = document.getElementById("explanations");
    if (!container) return;

    const extExp = extMetrics.explanations || {};
    const absExp = absMetrics.explanations || {};

    const sections = [
        { key: "rouge", title: "ROUGE Scores" },
        { key: "compression_ratio", title: "Compression Ratio" },
        { key: "information_density", title: "Information Density" },
        { key: "redundancy", title: "Redundancy" },
        { key: "coherence", title: "Coherence" },
        { key: "coverage", title: "Coverage" },
        { key: "topic_coverage", title: "Topic Coverage" },
        { key: "overall_quality", title: "Overall Quality" },
    ];

    let html = "";
    sections.forEach((section, i) => {
        const extText = extExp[section.key] || "—";
        const absText = absExp[section.key] || "—";
        html += `
            <div class="explanation-item fade-in" style="animation-delay: ${i * 0.1}s">
                <h4>${section.title}</h4>
                <div class="explanation-row">
                    <div class="explanation-method extractive-border">
                        <span class="explanation-badge extractive-badge">Extractive</span>
                        <p>${extText}</p>
                    </div>
                    <div class="explanation-method abstractive-border">
                        <span class="explanation-badge abstractive-badge">Abstractive</span>
                        <p>${absText}</p>
                    </div>
                </div>
            </div>
        `;
    });

    container.innerHTML = html;
}

function renderSingleMetrics(metrics, type) {
    const rouge = metrics.rouge_scores;
    document.getElementById("rouge-table").innerHTML = `
        <table class="fade-in">
            <thead><tr><th>Metric</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
            <tbody>
                <tr><td>ROUGE-1</td><td>${rouge.rouge1.precision}</td><td>${rouge.rouge1.recall}</td><td><strong>${rouge.rouge1.f1}</strong></td></tr>
                <tr><td>ROUGE-2</td><td>${rouge.rouge2.precision}</td><td>${rouge.rouge2.recall}</td><td><strong>${rouge.rouge2.f1}</strong></td></tr>
                <tr><td>ROUGE-L</td><td>${rouge.rougeL.precision}</td><td>${rouge.rougeL.recall}</td><td><strong>${rouge.rougeL.f1}</strong></td></tr>
            </tbody>
        </table>
    `;

    const colorClass = type === "extractive" ? "extractive-color" : "abstractive-color";
    const densityEl = document.getElementById("density-metrics");
    densityEl.innerHTML = createAnimatedScoreBar("Score", metrics.information_density.score, colorClass);
    animateAllBars(densityEl);

    const cohEl = document.getElementById("coherence-metrics");
    cohEl.innerHTML = createAnimatedScoreBar("Score", metrics.coherence_score.score, colorClass);
    animateAllBars(cohEl);

    const redEl = document.getElementById("redundancy-metrics");
    redEl.innerHTML = createAnimatedScoreBar("Score", metrics.redundancy_score.score, colorClass);
    animateAllBars(redEl);

    const covEl = document.getElementById("coverage-metrics");
    covEl.innerHTML = createAnimatedScoreBar("Score", metrics.coverage_score.score, colorClass);
    animateAllBars(covEl);

    const qualEl = document.getElementById("overall-quality");
    qualEl.innerHTML = `
        <div class="quality-scores">
            <div class="quality-score-card ${type === 'extractive' ? 'extractive-bg' : 'abstractive-bg'}">
                <div class="quality-number" data-target="${(metrics.overall_quality.score * 100).toFixed(1)}" data-suffix="%">0%</div>
                <div class="quality-label">${type}</div>
            </div>
        </div>
    `;
    setTimeout(() => qualEl.querySelectorAll("[data-target]").forEach(animateNumber), 200);

    document.getElementById("rankings").innerHTML = "<p>Compare both methods to see rankings.</p>";

    const topicEl = document.getElementById("topic-coverage");
    topicEl.innerHTML = `
        <div class="topic-tags">
            ${metrics.topic_coverage.covered_topics.map((t, i) => `<span class="topic-tag covered fade-in" style="animation-delay: ${i * 0.05}s">${t}</span>`).join("")}
            ${metrics.topic_coverage.missing_topics.map((t, i) => `<span class="topic-tag missing fade-in" style="animation-delay: ${(metrics.topic_coverage.covered_topics.length + i) * 0.05}s">${t}</span>`).join("")}
        </div>
    `;
}

// --- Animation Utilities ---

function animateNumber(el) {
    const target = parseFloat(el.dataset.target);
    const suffix = el.dataset.suffix || "";
    const duration = 1000;
    const startTime = performance.now();
    const isFloat = String(target).includes(".");

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        // Ease out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = target * eased;

        if (isFloat) {
            el.textContent = current.toFixed(1) + suffix;
        } else {
            el.textContent = Math.round(current) + suffix;
        }

        if (progress < 1) {
            requestAnimationFrame(update);
        } else {
            if (isFloat) {
                el.textContent = target.toFixed(1) + suffix;
            } else {
                el.textContent = Math.round(target) + suffix;
            }
        }
    }

    requestAnimationFrame(update);
}

function typewriterEffect(el, text, speed = 8) {
    el.textContent = "";
    let i = 0;
    const words = text.split(" ");

    function typeNext() {
        if (i < words.length) {
            el.textContent += (i > 0 ? " " : "") + words[i];
            i++;
            setTimeout(typeNext, speed);
        }
    }

    typeNext();
}

// --- Scroll Reveal (IntersectionObserver) ---

let scrollObserver = null;

function initScrollReveal() {
    // Disconnect previous observer if any
    if (scrollObserver) {
        scrollObserver.disconnect();
    }

    const elements = document.querySelectorAll(".scroll-reveal:not(.revealed)");

    scrollObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add("revealed");
                scrollObserver.unobserve(entry.target);

                // Trigger bar animations inside this card once revealed
                setTimeout(() => {
                    const bars = entry.target.querySelectorAll(".score-bar-fill[data-width]");
                    bars.forEach(bar => {
                        bar.style.width = bar.dataset.width + "%";
                    });

                    // Trigger number animations inside this card
                    entry.target.querySelectorAll("[data-target]").forEach(animateNumber);
                }, 150);
            }
        });
    }, {
        threshold: 0.15,
        rootMargin: "0px 0px -50px 0px"
    });

    elements.forEach(el => scrollObserver.observe(el));
}
