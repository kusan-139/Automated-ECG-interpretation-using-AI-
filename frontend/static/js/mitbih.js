// static/js/mitbih.js
// MIT-BIH frontend logic (2-lead CSV upload, plotting, API call)

const MITBIH_API_BASE = "";   // same origin

// DOM elements
const mitFileInput      = document.getElementById("mitbih-file");
const mitAnalyzeBtn     = document.getElementById("mitbih-analyze-btn");
const mitPredictionDiv  = document.getElementById("mitbih-prediction");
const mitWaveformCanvas = document.getElementById("mitbih-waveform");
const mitProbsCanvas    = document.getElementById("mitbih-probabilities");
const mitLegendDiv      = document.getElementById("mitbih-class-legend");

// Lead info + navigation
const mitLeadPrevBtn    = document.getElementById("mit-lead-prev");
const mitLeadNextBtn    = document.getElementById("mit-lead-next");
const mitLeadNameSpan   = document.getElementById("mit-lead-name");
const mitLeadIndexSpan  = document.getElementById("mit-lead-index");
const mitLeadColorDot   = document.getElementById("mit-lead-color-dot");

// Zoom & pan sliders
const mitZoomInBtn      = document.getElementById("mit-zoom-in-btn");
const mitZoomOutBtn     = document.getElementById("mit-zoom-out-btn");
const mitHorizSlider    = document.getElementById("mit-horizontal-slider");
const mitVertSlider     = document.getElementById("mit-vertical-slider");

// Charts
let mitWaveformChart = null;
let mitProbsChart    = null;

// ---------- multi-lead data & state ----------

// mitAllLeads[leadIndex] = [samples...]
let mitAllLeads    = null;
let mitCurrentLead = 0;

const MIT_NUM_LEADS   = 2;
const MIT_LEAD_NAMES  = ["MLII", "V1"];
const MIT_LEAD_COLORS = ["#1f77b4", "#d62728"]; // blue & red

// vertical scale base range
const MIT_BASE_Y_RANGE = 0.7;

// zoom in time axis (1 = full, >1 = zoomed-in)
let mitZoomLevel = 1;
const MIT_ZOOM_STEP = 0.5;

// horizontal pan: 0 (left) .. 1 (right)
let mitPanX = 0.0;

// vertical pan: -1 .. 1 (moves y window up/down)
let mitYOffset = 0.0;

// ---------- helper: update lead text + dot ----------

function updateMitLeadInfo(idx) {
  const leadNumber = idx + 1;
  const name = MIT_LEAD_NAMES[idx] || `Lead ${leadNumber}`;
  mitLeadNameSpan.textContent = `Lead ${leadNumber} – ${name}`;
  mitLeadIndexSpan.textContent = `(index ${idx})`;
  mitLeadColorDot.style.backgroundColor =
    MIT_LEAD_COLORS[idx] || "#1f77b4";
}

// ---------- parse 2-lead CSV ----------

function parseMit2LeadCSV(text) {
  const lines = text
    .split(/\r?\n/)
    .map(l => l.trim())
    .filter(l => l.length > 0);

  if (!lines.length) {
    throw new Error("Empty CSV file.");
  }

  const firstParts = lines[0].split(/[,;\s]+/).filter(p => p !== "");
  const numCols = firstParts.length;
  if (numCols < MIT_NUM_LEADS) {
    throw new Error(
      `Expected at least ${MIT_NUM_LEADS} columns for 2-lead MIT-BIH ECG, found ${numCols}.`
    );
  }

  const leads = [];
  for (let c = 0; c < MIT_NUM_LEADS; c++) leads.push([]);

  for (const line of lines) {
    const parts = line.split(/[,;\s]+/).filter(p => p !== "");
    if (parts.length < MIT_NUM_LEADS) continue;
    for (let c = 0; c < MIT_NUM_LEADS; c++) {
      const v = parseFloat(parts[c]);
      if (!Number.isNaN(v)) leads[c].push(v);
    }
  }

  return leads;
}

// ---------- waveform chart setup ----------

function initMitWaveformChart() {
  if (!mitAllLeads) return;

  mitCurrentLead = 0;
  mitZoomLevel   = 1;
  mitPanX        = 0;
  mitYOffset     = 0;
  mitHorizSlider.value = 0;
  mitVertSlider.value  = 0;

  const data   = mitAllLeads[mitCurrentLead];
  const labels = data.map((_, i) => i);

  const ctx = mitWaveformCanvas.getContext("2d");

  mitWaveformChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: `Lead ${mitCurrentLead + 1} (${MIT_LEAD_NAMES[mitCurrentLead] || "ECG"})`,
          data,
          borderColor: MIT_LEAD_COLORS[mitCurrentLead] || "#1f77b4",
          pointRadius: 0,
          borderWidth: 1.4,
          tension: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { display: false },
        y: {
          min: -MIT_BASE_Y_RANGE,
          max:  MIT_BASE_Y_RANGE,
          ticks: { stepSize: 0.1 },
        },
      },
      plugins: {
        legend: {
          display: true,
          labels: { boxWidth: 18 },
        },
      },
    },
  });

  updateMitLeadInfo(mitCurrentLead);
  applyMitView();
}

function updateMitLeadChart(idx) {
  if (!mitAllLeads || !mitWaveformChart) return;
  if (idx < 0 || idx >= mitAllLeads.length) return;

  mitCurrentLead = idx;
  mitZoomLevel   = 1;
  mitPanX        = 0;
  mitYOffset     = 0;
  mitHorizSlider.value = 0;
  mitVertSlider.value  = 0;

  const fullData = mitAllLeads[mitCurrentLead];

  const ds = mitWaveformChart.data.datasets[0];
  ds.data  = fullData;
  ds.label = `Lead ${mitCurrentLead + 1} (${MIT_LEAD_NAMES[mitCurrentLead] || "ECG"})`;
  ds.borderColor = MIT_LEAD_COLORS[mitCurrentLead] || "#1f77b4";

  mitWaveformChart.data.labels = fullData.map((_, i) => i);

  updateMitLeadInfo(mitCurrentLead);
  applyMitView();
}

// ---------- zoom + pan apply ----------

function applyMitView() {
  if (!mitWaveformChart || !mitAllLeads) return;

  const full = mitAllLeads[mitCurrentLead];
  const N = full.length || 1;

  // horizontal zoom & pan
  const factor   = Math.max(1, mitZoomLevel);
  const visible  = Math.max(20, Math.floor(N / factor));
  const maxStart = Math.max(0, N - visible);

  // mitPanX in [0,1]: 0 left, 1 right
  const start = Math.floor(maxStart * mitPanX);
  const end   = Math.min(N, start + visible);
  const windowData = full.slice(start, end);

  mitWaveformChart.data.labels = windowData.map((_, i) => i);
  mitWaveformChart.data.datasets[0].data = windowData;

  // vertical pan based on offset
  const baseMin = -MIT_BASE_Y_RANGE;
  const baseMax =  MIT_BASE_Y_RANGE;
  const span    = baseMax - baseMin;
  const offset  = mitYOffset * (span * 0.5); // move up/down half span

  mitWaveformChart.options.scales.y.min = baseMin + offset;
  mitWaveformChart.options.scales.y.max = baseMax + offset;

  mitWaveformChart.update("none");
}

// ---------- zoom buttons ----------

mitZoomInBtn.addEventListener("click", () => {
  mitZoomLevel += MIT_ZOOM_STEP;
  if (mitZoomLevel > 8) mitZoomLevel = 8;
  applyMitView();
});

mitZoomOutBtn.addEventListener("click", () => {
  mitZoomLevel -= MIT_ZOOM_STEP;
  if (mitZoomLevel < 1) mitZoomLevel = 1;
  applyMitView();
});

// ---------- sliders (pan) ----------

// horizontal: 0..100 -> 0..1
mitHorizSlider.addEventListener("input", () => {
  const v = parseInt(mitHorizSlider.value, 10) || 0;
  mitPanX = Math.min(1, Math.max(0, v / 100));
  applyMitView();
});

// vertical: -50..50 -> -1..1
mitVertSlider.addEventListener("input", () => {
  const v = parseInt(mitVertSlider.value, 10) || 0;
  mitYOffset = Math.min(1, Math.max(-1, v / 50));
  applyMitView();
});

// ---------- arrow navigation ----------

mitLeadPrevBtn.addEventListener("click", () => {
  if (!mitAllLeads) return;
  const next = (mitCurrentLead - 1 + MIT_NUM_LEADS) % MIT_NUM_LEADS;
  updateMitLeadChart(next);
});

mitLeadNextBtn.addEventListener("click", () => {
  if (!mitAllLeads) return;
  const next = (mitCurrentLead + 1) % MIT_NUM_LEADS;
  updateMitLeadChart(next);
});

// ---------- Analyze (upload + backend) ----------

mitAnalyzeBtn.addEventListener("click", async () => {
  if (!mitFileInput.files.length) {
    alert("Please select a CSV file first.");
    return;
  }

  const file = mitFileInput.files[0];

  // 1) Local preview (2 leads)
  const reader = new FileReader();
  reader.onload = () => {
    try {
      mitAllLeads = parseMit2LeadCSV(reader.result);

      if (!mitWaveformChart) {
        initMitWaveformChart();
      } else {
        updateMitLeadChart(0);
      }
    } catch (err) {
      console.error(err);
      mitPredictionDiv.textContent =
        "Error parsing CSV: " + (err.message || String(err));
    }
  };
  reader.readAsText(file);

  // 2) Send to backend for prediction
  const formData = new FormData();
  formData.append("file", file);

  mitPredictionDiv.textContent =
    "Running MIT-BIH model (research-only)...";

  try {
    const res = await fetch(`${MITBIH_API_BASE}/api/predict_ecg_mitbih`, {
      method: "POST",
      body: formData,
    });
    const data = await res.json();

    if (!res.ok) {
      mitPredictionDiv.textContent = `Error: ${data.error || "Unknown error"}`;
      if (mitLegendDiv) mitLegendDiv.innerHTML = "";
      if (mitProbsChart) mitProbsChart.destroy();
      return;
    }

    mitPredictionDiv.textContent =
      `Predicted ${data.prediction_id} : ${data.prediction_name} (RESEARCH-ONLY)`;

    const probsData   = data.probabilities || [];
    const fullLabels  = probsData.map(p => p.label);
    const shortLabels = fullLabels.map(l => l.split(":")[0].trim());
    const probs       = probsData.map(p => p.prob);

    drawMitProbabilitiesChart(mitProbsCanvas, shortLabels, probs);
    updateMitClassLegend(fullLabels);

  } catch (err) {
    console.error(err);
    mitPredictionDiv.textContent = "Request failed.";
    if (mitLegendDiv) mitLegendDiv.innerHTML = "";
    if (mitProbsChart) mitProbsChart.destroy();
  }
});

// ---------- probabilities + legend ----------

function drawMitProbabilitiesChart(canvas, labels, probs) {
  if (mitProbsChart) mitProbsChart.destroy();
  mitProbsChart = new Chart(canvas.getContext("2d"), {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Probability",
          data: probs,
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        y: { min: 0, max: 1 },
      },
    },
  });
}

function updateMitClassLegend(fullLabels) {
  if (!mitLegendDiv) return;

  if (!fullLabels || !fullLabels.length) {
    mitLegendDiv.innerHTML = "";
    return;
  }

  const items = fullLabels.map(l => `<li>${l}</li>`).join("");

  mitLegendDiv.innerHTML = `
    <strong>Class Legend</strong>
    <ul>${items}</ul>
  `;
}
