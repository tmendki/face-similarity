// Local dev:  http://localhost:8000
// HF Space:   https://YOUR_HF_USERNAME-face-similarity-api.hf.space
const API_BASE = "https://tanishmendki-face-similarity-api.hf.space";

const state = { file1: null, file2: null, loading: false };
let progressTimers = [];

// ── DOM refs ──
const zone1       = document.getElementById("zone1");
const zone2       = document.getElementById("zone2");
const input1      = document.getElementById("input1");
const input2      = document.getElementById("input2");
const placeholder1= document.getElementById("placeholder1");
const placeholder2= document.getElementById("placeholder2");
const preview1    = document.getElementById("preview1");
const preview2    = document.getElementById("preview2");
const previewImg1 = document.getElementById("previewImg1");
const previewImg2 = document.getElementById("previewImg2");
const remove1     = document.getElementById("remove1");
const remove2     = document.getElementById("remove2");
const error1      = document.getElementById("error1");
const error2      = document.getElementById("error2");
const compareBtn  = document.getElementById("compareBtn");
const apiError    = document.getElementById("apiError");
const progressWrap= document.getElementById("progressWrap");
const progressFill= document.getElementById("progressFill");
const progressLabel=document.getElementById("progressLabel");
const progressPct = document.getElementById("progressPct");
const results     = document.getElementById("results");
const scoreNumber = document.getElementById("scoreNumber");
const scoreLabel  = document.getElementById("scoreLabel");
const simBar      = document.getElementById("simBar");
const rawCosineVal= document.getElementById("rawCosineVal");
const breakdownRows=document.getElementById("breakdownRows");
const resetBtn    = document.getElementById("resetBtn");

// ── File handling ──

function setFile(index, file) {
  const allowed = ["image/jpeg", "image/png", "image/webp"];
  clearZoneError(index);

  if (!allowed.includes(file.type)) {
    showZoneError(index, "Invalid file type — use JPEG, PNG, or WebP.");
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showZoneError(index, "File exceeds the 10 MB limit.");
    return;
  }

  if (index === 1) state.file1 = file;
  else             state.file2 = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    const img = index === 1 ? previewImg1 : previewImg2;
    const ph  = index === 1 ? placeholder1 : placeholder2;
    const pv  = index === 1 ? preview1 : preview2;
    img.src = e.target.result;
    ph.hidden = true;
    pv.hidden = false;
  };
  reader.readAsDataURL(file);
  syncCompareBtn();
}

function clearFile(index) {
  if (index === 1) {
    state.file1 = null;
    input1.value = "";
    placeholder1.hidden = false;
    preview1.hidden = true;
    previewImg1.src = "";
  } else {
    state.file2 = null;
    input2.value = "";
    placeholder2.hidden = false;
    preview2.hidden = true;
    previewImg2.src = "";
  }
  clearZoneError(index);
  syncCompareBtn();
}

function showZoneError(index, msg) {
  const el = index === 1 ? error1 : error2;
  const zone = index === 1 ? zone1 : zone2;
  el.textContent = msg;
  el.hidden = false;
  zone.classList.add("has-error");
}

function clearZoneError(index) {
  const el = index === 1 ? error1 : error2;
  const zone = index === 1 ? zone1 : zone2;
  el.hidden = true;
  zone.classList.remove("has-error");
}

function syncCompareBtn() {
  compareBtn.disabled = !(state.file1 && state.file2) || state.loading;
}

// ── Zone wiring ──

function wireZone(zone, input, index) {
  zone.addEventListener("click", (e) => {
    if (!e.target.closest(".remove-btn")) input.click();
  });
  input.addEventListener("change", () => {
    if (input.files[0]) setFile(index, input.files[0]);
  });
  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("drag-over");
  });
  zone.addEventListener("dragleave", (e) => {
    if (!zone.contains(e.relatedTarget)) zone.classList.remove("drag-over");
  });
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) setFile(index, file);
  });
}

wireZone(zone1, input1, 1);
wireZone(zone2, input2, 2);
remove1.addEventListener("click", (e) => { e.stopPropagation(); clearFile(1); });
remove2.addEventListener("click", (e) => { e.stopPropagation(); clearFile(2); });

// ── Progress indicator ──

const PROGRESS_STEPS = [
  { at:    0, pct:  8, label: "Detecting faces…" },
  { at: 1800, pct: 45, label: "Computing embeddings…" },
  { at: 4500, pct: 78, label: "Measuring similarity…" },
];

function startProgress() {
  progressTimers.forEach(clearTimeout);
  progressTimers = [];

  progressFill.style.transition = "none";
  progressFill.style.width = "0%";
  progressPct.textContent = "0%";
  progressLabel.textContent = PROGRESS_STEPS[0].label;
  progressWrap.hidden = false;

  // Schedule each step
  PROGRESS_STEPS.forEach((step) => {
    const t = setTimeout(() => {
      setProgressPct(step.pct, step.label, step.at === 0 ? "none" : "width 2.5s ease-out");
    }, step.at);
    progressTimers.push(t);
  });
}

function setProgressPct(pct, label, transition = "width 0.4s ease") {
  progressFill.style.transition = transition;
  progressFill.style.width = pct + "%";
  progressPct.textContent = Math.round(pct) + "%";
  if (label) progressLabel.textContent = label;
}

function completeProgress() {
  progressTimers.forEach(clearTimeout);
  progressTimers = [];
  setProgressPct(100, "Done");
  setTimeout(() => { progressWrap.hidden = true; }, 700);
}

function cancelProgress() {
  progressTimers.forEach(clearTimeout);
  progressTimers = [];
  progressWrap.hidden = true;
}

// ── Compare ──

compareBtn.addEventListener("click", runComparison);

async function runComparison() {
  if (!state.file1 || !state.file2 || state.loading) return;

  state.loading = true;
  compareBtn.disabled = true;
  compareBtn.textContent = "Analyzing…";
  apiError.hidden = true;
  clearZoneError(1);
  clearZoneError(2);
  startProgress();

  const form = new FormData();
  form.append("image1", state.file1);
  form.append("image2", state.file2);

  try {
    const res = await fetch(`${API_BASE}/compare`, { method: "POST", body: form });
    const data = await res.json();

    if (!res.ok) {
      cancelProgress();
      handleApiError(data);
    } else {
      completeProgress();
      setTimeout(() => showResults(data), 300);
    }
  } catch {
    cancelProgress();
    apiError.textContent = "Could not reach the server. Is the backend running on port 8000?";
    apiError.hidden = false;
  } finally {
    state.loading = false;
    compareBtn.disabled = false;
    compareBtn.textContent = "Compare Faces";
    syncCompareBtn();
  }
}

function handleApiError(data) {
  const msg = data.message || "An unknown error occurred.";
  const code = data.error || "";

  if (code === "no_face_detected") {
    if (/image 1/i.test(msg)) {
      showZoneError(1, "No face detected — try a clearer, front-facing photo.");
    } else if (/image 2/i.test(msg)) {
      showZoneError(2, "No face detected — try a clearer, front-facing photo.");
    } else {
      showZoneError(1, "No face detected.");
      showZoneError(2, "No face detected.");
    }
  } else {
    apiError.textContent = msg;
    apiError.hidden = false;
  }
}

// ── Results ──

function showResults(data) {
  const { score, raw_cosine, label, breakdown } = data;

  // Reset animated elements before (re-)showing
  scoreNumber.textContent = "0.0";
  scoreLabel.textContent = label;
  rawCosineVal.textContent = raw_cosine.toFixed(4);
  simBar.style.setProperty("--pct", "0%");

  // Restart fade-in animation
  results.style.animation = "none";
  results.hidden = false;
  results.offsetHeight; // force reflow
  results.style.animation = "";

  // Scroll results into view
  results.scrollIntoView({ behavior: "smooth", block: "nearest" });

  // Animate score count-up
  animateCount(scoreNumber, 0, score, 850);

  // Animate bar (tiny rAF delay so transition fires)
  requestAnimationFrame(() => requestAnimationFrame(() => {
    simBar.style.setProperty("--pct", score + "%");
  }));

  // Render breakdown rows
  renderBreakdown(breakdown);
}

function animateCount(el, from, to, ms) {
  const start = performance.now();
  function tick(now) {
    const t = Math.min((now - start) / ms, 1);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = (from + (to - from) * eased).toFixed(1);
    if (t < 1) requestAnimationFrame(tick);
    else el.textContent = to.toFixed(1);
  }
  requestAnimationFrame(tick);
}

function renderBreakdown(breakdown) {
  breakdownRows.innerHTML = Object.entries(breakdown).map(([label, score]) => `
    <div class="bd-row">
      <span class="bd-label">${label}</span>
      <div class="bd-bar"><div class="bd-fill" data-pct="${score}"></div></div>
      <span class="bd-score">${score.toFixed(1)}</span>
    </div>
  `).join("");

  // Animate bars in next frame
  requestAnimationFrame(() => requestAnimationFrame(() => {
    breakdownRows.querySelectorAll(".bd-fill").forEach((el) => {
      el.style.width = el.dataset.pct + "%";
    });
  }));
}

// ── Reset ──

resetBtn.addEventListener("click", resetAll);

function resetAll() {
  clearFile(1);
  clearFile(2);
  apiError.hidden = true;
  results.hidden = true;
  simBar.style.setProperty("--pct", "0%");
  document.getElementById("uploadGrid").scrollIntoView({ behavior: "smooth", block: "center" });
}
