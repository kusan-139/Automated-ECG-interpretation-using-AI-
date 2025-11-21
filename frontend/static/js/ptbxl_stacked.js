// frontend/static/js/ptbxl_stacked.js

const API_BASE = "http://localhost:5000";

const fileInput = document.getElementById("ptbxl-file");
const analyzeBtn = document.getElementById("ptbxl-analyze-btn");
const predictionDiv = document.getElementById("ptbxl-prediction");
const waveformCanvas = document.getElementById("ptbxl-waveform");
const probsCanvas = document.getElementById("ptbxl-probabilities");
const legendDiv = document.getElementById("ptbxl-class-legend");   // <--- NEW

const btnPrev = document.getElementById("lead-prev");
const btnNext = document.getElementById("lead-next");
const leadNameSpan = document.getElementById("lead-name");
const leadIndexLabel = document.getElementById("lead-index-label");
const leadColorDot = document.getElementById("lead-color-dot");

const zoomInBtn = document.getElementById("zoom-in-btn");
const zoomOutBtn = document.getElementById("zoom-out-btn");

// new sliders
const xSlider = document.getElementById("ptbxl-x-slider"); // left-right pan
const ySlider = document.getElementById("ptbxl-y-slider"); // up-down pan

let waveformChart = null;
let probsChart = null;

// allLeads[leadIndex] = [samples...]
let allLeads = null;
let currentLead = 0;
const NUM_LEADS = 12;

// vertical scale (fixed)
const BASE_Y_RANGE = 0.7;

// zoom + pan state
let zoomLevel = 1;       // 1 = full width, >1 = zoom in
const ZOOM_STEP = 0.5;

let xCenter = 0.5;       // 0..1 (0 = far left, 1 = far right)
let yCenterOffset = 0.0; // -1..1 (shift up/down one range)

// lead names + colors
const LEAD_NAMES = [
  "I",
  "II",
  "III",
  "aVR",
  "aVL",
  "aVF",
  "V1",
  "V2",
  "V3",
  "V4",
  "V5",
  "V6",
];

const LEAD_COLORS = [
  "#1f77b4",
  "#ff7f0e",
  "#2ca02c",
  "#d62728",
  "#9467bd",
  "#8c564b",
  "#e377c2",
  "#7f7f7f",
  "#bcbd22",
  "#17becf",
  "#ff1493",
  "#00b894",
];

// ----------------- helpers -----------------

function updateLeadInfo(idx) {
  const leadNumber = idx + 1;
  const name = LEAD_NAMES[idx] || `Lead ${leadNumber}`;
  leadNameSpan.textContent = `Lead ${leadNumber} – ${name}`;
  leadIndexLabel.textContent = `(index ${idx})`;
  leadColorDot.style.backgroundColor = LEAD_COLORS[idx] || "#1f77b4";
}

function parse12LeadCSV(text) {
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);

  if (!lines.length) throw new Error("Empty CSV file.");

  const firstParts = lines[0]
    .split(/[,;\s]+/)
    .filter((p) => p !== "");
  const numCols = firstParts.length;
  if (numCols < NUM_LEADS) {
    throw new Error(
      `Expected at least 12 columns for 12-lead ECG, found ${numCols}.`
    );
  }

  const leads = [];
  for (let c = 0; c < NUM_LEADS; c++) leads.push([]);

  for (const line of lines) {
    const parts = line.split(/[,;\s]+/).filter((p) => p !== "");
    if (parts.length < NUM_LEADS) continue;
    for (let c = 0; c < NUM_LEADS; c++) {
      const v = parseFloat(parts[c]);
      if (!Number.isNaN(v)) leads[c].push(v);
    }
  }

  return leads;
}

// ----------------- chart init / update -----------------

function initWaveformChart() {
  if (!allLeads) return;
  currentLead = 0;
  zoomLevel = 1;
  xCenter = 0.5;
  yCenterOffset = 0.0;
  if (xSlider) xSlider.value = 50;
  if (ySlider) ySlider.value = 50;

  const data = allLeads[currentLead];
  const labels = data.map((_, i) => i);

  const ctx = waveformCanvas.getContext("2d");

  waveformChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: `Lead ${currentLead + 1} (${LEAD_NAMES[currentLead] || "ECG"})`,
          data,
          borderColor: LEAD_COLORS[currentLead] || "#1f77b4",
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
          min: -BASE_Y_RANGE,
          max: BASE_Y_RANGE,
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

  updateLeadInfo(currentLead);
  applyZoomPan();
}

function updateLeadChart(idx) {
  if (!allLeads || !waveformChart) return;
  if (idx < 0 || idx >= allLeads.length) return;

  currentLead = idx;
  zoomLevel = 1;
  xCenter = 0.5;
  yCenterOffset = 0.0;
  if (xSlider) xSlider.value = 50;
  if (ySlider) ySlider.value = 50;

  const fullData = allLeads[currentLead];

  const ds = waveformChart.data.datasets[0];
  ds.data = fullData;
  ds.label = `Lead ${currentLead + 1} (${LEAD_NAMES[currentLead] || "ECG"})`;
  ds.borderColor = LEAD_COLORS[currentLead] || "#1f77b4";

  waveformChart.data.labels = fullData.map((_, i) => i);

  updateLeadInfo(currentLead);
  applyZoomPan();
}

// ----------------- zoom + pan (width only) -----------------

function applyZoomPan() {
  if (!waveformChart || !allLeads) return;

  const full = allLeads[currentLead];
  const N = full.length || 1;

  // horizontal zoom
  const factor = Math.max(1, zoomLevel);      // 1 = full, 2 = half width, etc.
  const visible = Math.max(50, Math.floor(N / factor));

  // center index from slider (0..1)
  const centerIndex = Math.round((N - 1) * xCenter);

  let start = centerIndex - Math.floor(visible / 2);
  if (start < 0) start = 0;
  let end = start + visible;
  if (end > N) {
    end = N;
    start = Math.max(0, end - visible);
  }

  const windowData = full.slice(start, end);

  waveformChart.data.labels = windowData.map((_, i) => i);
  waveformChart.data.datasets[0].data = windowData;

  // vertical pan: shift window up/down without changing height
  const offset = yCenterOffset * BASE_Y_RANGE; // -0.7..+0.7-ish
  waveformChart.options.scales.y.min = -BASE_Y_RANGE + offset;
  waveformChart.options.scales.y.max = BASE_Y_RANGE + offset;

  waveformChart.update("none"); // fast, no animation
}

// ----------------- controls -----------------

// zoom buttons
zoomInBtn.addEventListener("click", () => {
  zoomLevel += ZOOM_STEP;
  if (zoomLevel > 8) zoomLevel = 8;
  applyZoomPan();
});

zoomOutBtn.addEventListener("click", () => {
  zoomLevel -= ZOOM_STEP;
  if (zoomLevel < 1) zoomLevel = 1;
  applyZoomPan();
});

// arrows
btnPrev.addEventListener("click", () => {
  if (!allLeads) return;
  const next = (currentLead - 1 + NUM_LEADS) % NUM_LEADS;
  updateLeadChart(next);
});

btnNext.addEventListener("click", () => {
  if (!allLeads) return;
  const next = (currentLead + 1) % NUM_LEADS;
  updateLeadChart(next);
});

// sliders
if (xSlider) {
  xSlider.addEventListener("input", () => {
    xCenter = xSlider.value / 100; // 0..1
    applyZoomPan();
  });
}

if (ySlider) {
  ySlider.addEventListener("input", () => {
    // 0..100 -> -1..+1 (center at 50)
    yCenterOffset = (ySlider.value - 50) / 50;
    applyZoomPan();
  });
}

// ----------------- analyze button (upload + backend) -----------------

analyzeBtn.addEventListener("click", async () => {
  if (!fileInput.files.length) {
    alert("Please select a CSV file first.");
    return;
  }

  const file = fileInput.files[0];

  // 1) parse CSV locally for waveform
  const reader = new FileReader();
  reader.onload = () => {
    try {
      allLeads = parse12LeadCSV(reader.result);

      if (!waveformChart) {
        initWaveformChart();
      } else {
        updateLeadChart(0);
      }
    } catch (err) {
      console.error(err);
      predictionDiv.textContent =
        "Error parsing CSV: " + (err.message || String(err));
    }
  };
  reader.readAsText(file);

  // 2) send to backend for prediction
  const formData = new FormData();
  formData.append("file", file);

  predictionDiv.textContent =
    "Running PTB-XL model (research-only). Please wait…";

  try {
    const res = await fetch(`${API_BASE}/api/predict_ecg_ptbxl`, {
      method: "POST",
      body: formData,
    });
    const data = await res.json();

    if (!res.ok) {
      predictionDiv.textContent = `Error: ${
        data.error || "Unknown error from server."
      }`;
      if (legendDiv) legendDiv.innerHTML = "";
      return;
    }

    // Use id + name together: "Predicted class_0 : Normal (RESEARCH-ONLY)"
    if (data.prediction_id && data.prediction_name) {
      predictionDiv.textContent =
        `Predicted ${data.prediction_id} : ${data.prediction_name} (RESEARCH-ONLY)`;
    } else {
      predictionDiv.textContent =
        `Predicted ${data.prediction_name || data.prediction} (RESEARCH-ONLY)`;
    }

    if (Array.isArray(data.probabilities) && data.probabilities.length > 0) {
      // full label from backend, e.g. "class_0 : Normal"
      const fullLabels = data.probabilities.map((p) => p.label);
      // x-axis labels: only "class_0", "class_1", ...
      const shortLabels = fullLabels.map((l) => l.split(":")[0].trim());
      const probs = data.probabilities.map((p) => p.prob);

      // update legend under chart
      updateClassLegend(fullLabels);

      if (probsChart) probsChart.destroy();
      probsChart = new Chart(probsCanvas.getContext("2d"), {
        type: "bar",
        data: {
          labels: shortLabels,
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
            x: {
              ticks: {
                maxRotation: 0,
                minRotation: 0,   // keep labels straight
              },
            },
            y: { min: 0, max: 1 },
          },
        },
      });
    } else {
      predictionDiv.textContent += " | No probability data returned.";
      if (legendDiv) legendDiv.innerHTML = "";
    }
  } catch (err) {
    console.error(err);
    predictionDiv.textContent = "Request failed.";
    if (legendDiv) legendDiv.innerHTML = "";
  }
});

// ----------------- legend helper -----------------

function updateClassLegend(fullLabels) {
  if (!legendDiv) return;

  if (!fullLabels || !fullLabels.length) {
    legendDiv.innerHTML = "";
    return;
  }

  const items = fullLabels
    .map((l) => `<li>${l}</li>`)
    .join("");

  legendDiv.innerHTML = `
    <strong>Class Legend</strong>
    <ul>${items}</ul>
  `;
}
