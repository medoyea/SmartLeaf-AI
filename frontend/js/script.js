/**
 * SmartLeaf AI — Frontend JavaScript
 * ====================================
 * Handles: image upload, drag & drop, camera capture,
 * API calls, results rendering, model comparison, dashboard.
 */

'use strict';

// ── Configuration ────────────────────────────────────────────────────────────
const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ? 'http://localhost:5000'
  : '';

// ── Demo Classes (shown in tips card) ───────────────────────────────────────
const DEMO_CLASSES = [
  'Apple Scab', 'Apple Black Rot', 'Cedar Rust', 'Apple Healthy',
  'Tomato Early Blight', 'Tomato Late Blight', 'Mosaic Virus', 'Tomato Healthy',
  'Corn Rust', 'Corn Healthy', 'Grape Black Rot', 'Grape Healthy',
  'Potato Early Blight', 'Potato Late Blight', 'Potato Healthy',
];

// ── State ─────────────────────────────────────────────────────────────────────
const state = {
  selectedFile: null,
  cameraStream: null,
  lastResult: null,
};

// ── DOM References ───────────────────────────────────────────────────────────
const el = {
  dropzone: document.getElementById('dropzone'),
  dzDefault: document.getElementById('dzDefault'),
  dzPreview: document.getElementById('dzPreview'),
  previewImg: document.getElementById('previewImg'),
  fileInput: document.getElementById('fileInput'),
  analyzeBtn: document.getElementById('analyzeBtn'),
  cameraBtn: document.getElementById('cameraBtn'),
  cameraVideo: document.getElementById('cameraVideo'),
  cameraCanvas: document.getElementById('cameraCanvas'),
  captureBtn: document.getElementById('captureBtn'),
  loadingOverlay: document.getElementById('loadingOverlay'),
  loadingText: document.getElementById('loadingText'),
  results: document.getElementById('results'),
  scanArea: document.getElementById('scanArea'),
  newScanBtn: document.getElementById('newScanBtn'),
  statusDot: document.getElementById('statusDot'),
  statusLabel: document.getElementById('statusLabel'),
  classTags: document.getElementById('classTags'),
  historyTableBody: document.getElementById('historyTableBody'),
  refreshStatsBtn: document.getElementById('refreshStatsBtn'),
};

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initClassTags();
  checkAPIHealth();
  fetchComparisonData();
  fetchStats();
  attachEventListeners();
  setInterval(fetchStats, 30000); // Refresh every 30s
});

function initClassTags() {
  if (!el.classTags) return;
  DEMO_CLASSES.forEach(cls => {
    const tag = document.createElement('span');
    tag.className = 'sc-tag';
    tag.textContent = cls;
    el.classTags.appendChild(tag);
  });
}

// ── API Health Check ─────────────────────────────────────────────────────────
async function checkAPIHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    const data = await res.json();
    if (data.status === 'online') {
      el.statusDot.className = 'status-dot online';
      el.statusLabel.textContent = data.demo_mode ? 'Demo Mode' : 'Models Ready';
    } else {
      setOffline();
    }
  } catch {
    setOffline();
  }
}
function setOffline() {
  el.statusDot.className = 'status-dot offline';
  el.statusLabel.textContent = 'Offline';
}

// ── Event Listeners ──────────────────────────────────────────────────────────
function attachEventListeners() {
  // File input
  el.fileInput.addEventListener('change', e => handleFileSelect(e.target.files[0]));

  // Dropzone click
  el.dropzone.addEventListener('click', () => {
    if (el.dzPreview.style.display === 'none') {
      el.fileInput.click();
    } else {
      el.fileInput.click(); // Allow re-selection
    }
  });

  // Drag & drop
  el.dropzone.addEventListener('dragover', e => { e.preventDefault(); el.dropzone.classList.add('drag-over'); });
  el.dropzone.addEventListener('dragleave', () => el.dropzone.classList.remove('drag-over'));
  el.dropzone.addEventListener('drop', e => {
    e.preventDefault();
    el.dropzone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file);
  });

  // Analyze button
  el.analyzeBtn.addEventListener('click', runAnalysis);

  // New scan button
  el.newScanBtn.addEventListener('click', resetToScan);

  // Camera
  el.cameraBtn.addEventListener('click', toggleCamera);
  el.captureBtn.addEventListener('click', capturePhoto);

  // Treatment tabs
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });

  // Refresh stats
  if (el.refreshStatsBtn) el.refreshStatsBtn.addEventListener('click', fetchStats);

  // Nav links smooth scroll
  document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', e => {
      e.preventDefault();
      const section = link.dataset.section;
      scrollToSection(section);
      document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
      link.classList.add('active');
    });
  });
}

// ── File Handling ────────────────────────────────────────────────────────────
function handleFileSelect(file) {
  if (!file) return;
  if (!file.type.startsWith('image/')) {
    showToast('Please select an image file (JPG, PNG, WEBP)', 'error');
    return;
  }
  if (file.size > 16 * 1024 * 1024) {
    showToast('Image is too large. Maximum size is 16MB.', 'error');
    return;
  }

  state.selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    el.previewImg.src = e.target.result;
    el.dzDefault.style.display = 'none';
    el.dzPreview.style.display = 'block';
    el.analyzeBtn.disabled = false;
  };
  reader.readAsDataURL(file);
}

// ── Camera ───────────────────────────────────────────────────────────────────
async function toggleCamera() {
  if (state.cameraStream) {
    stopCamera();
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
    });
    state.cameraStream = stream;
    el.cameraVideo.srcObject = stream;
    el.cameraVideo.style.display = 'block';
    el.captureBtn.style.display = 'block';
    el.cameraBtn.textContent = '✕ Close Camera';
  } catch (e) {
    showToast('Camera access denied or not available.', 'error');
  }
}

function stopCamera() {
  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach(t => t.stop());
    state.cameraStream = null;
  }
  el.cameraVideo.style.display = 'none';
  el.captureBtn.style.display = 'none';
  el.cameraBtn.innerHTML = `
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
      <circle cx="12" cy="13" r="4"/>
    </svg>
    Open Camera
  `;
}

function capturePhoto() {
  const canvas = el.cameraCanvas;
  const video = el.cameraVideo;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);

  canvas.toBlob(blob => {
    const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
    handleFileSelect(file);
    stopCamera();
  }, 'image/jpeg', 0.92);
}

// ── Analysis ─────────────────────────────────────────────────────────────────
async function runAnalysis() {
  if (!state.selectedFile) return;

  // Show loading overlay
  showLoading();

  try {
    const formData = new FormData();
    formData.append('image', state.selectedFile);

    // Animate loading steps
    const steps = ['ls1', 'ls2', 'ls3', 'ls4'];
    const delays = [0, 800, 1600, 2400];
    steps.forEach((id, i) => {
      setTimeout(() => {
        document.getElementById(id)?.classList.add('active');
        if (i > 0) document.getElementById(steps[i - 1])?.classList.add('done');
      }, delays[i]);
    });

    const response = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error || `HTTP ${response.status}`);
    }

    const data = await response.json();
    state.lastResult = data;

    // Mark all done
    steps.forEach(id => {
      document.getElementById(id)?.classList.remove('active');
      document.getElementById(id)?.classList.add('done');
    });

    await delay(400);
    hideLoading();
    renderResults(data);
    scrollToSection('results');

  } catch (err) {
    hideLoading();
    console.error('Analysis error:', err);

    // Demo fallback: generate client-side demo result
    const demoData = generateDemoResult();
    state.lastResult = demoData;
    renderResults(demoData);
    showToast(`Demo mode: ${err.message}`, 'warn');
    scrollToSection('results');
  }
}

// ── Demo Fallback ─────────────────────────────────────────────────────────────
function generateDemoResult() {
  const demoClasses = [
    { key: 'Tomato___Early_blight', name: 'Tomato Early Blight', plant: 'Tomato', healthy: false },
    { key: 'Apple___Apple_scab', name: 'Apple Scab', plant: 'Apple', healthy: false },
    { key: 'Tomato___healthy', name: 'Healthy Tomato', plant: 'Tomato', healthy: true },
    { key: 'Potato___Late_blight', name: 'Potato Late Blight', plant: 'Potato', healthy: false },
    { key: 'Grape___Black_rot', name: 'Grape Black Rot', plant: 'Grape', healthy: false },
  ];
  const chosen = demoClasses[Math.floor(Math.random() * demoClasses.length)];
  const cnnConf = 0.75 + Math.random() * 0.22;
  const svmConf = 0.70 + Math.random() * 0.20;
  const sevScore = chosen.healthy ? Math.floor(Math.random() * 8) : 20 + Math.floor(Math.random() * 60);
  const sevLevel = sevScore < 8 ? 'healthy' : sevScore < 25 ? 'mild' : sevScore < 55 ? 'moderate' : 'severe';

  return {
    success: true,
    inference_time_ms: 28 + Math.floor(Math.random() * 40),
    prediction: {
      cnn: { class: chosen.key, confidence: parseFloat((cnnConf * 100).toFixed(1)), top_predictions: [] },
      svm: { class: chosen.key, confidence: parseFloat((svmConf * 100).toFixed(1)), agreement: Math.random() > 0.15 },
      consensus: chosen.key,
      is_healthy: chosen.healthy,
    },
    severity: { level: sevLevel, score: sevScore, affected_percentage: (sevScore / 2.5).toFixed(1), color_analysis: { greenness_index: 8.2 } },
    treatment: getDemoTreatment(chosen),
    scan_id: Math.random().toString(16).slice(2, 10),
  };
}

function getDemoTreatment(cls) {
  const base = {
    common_name: cls.name,
    plant: cls.plant,
    description: cls.healthy
      ? `Your ${cls.plant} plant appears healthy! Continue your excellent care routine to maintain vibrant growth.`
      : `${cls.name} is a fungal disease affecting ${cls.plant} plants. Early intervention prevents significant crop loss.`,
    causes: cls.healthy ? [] : ['Fungal pathogen activity', 'Warm, humid conditions', 'Poor air circulation', 'Infected plant debris'],
    symptoms: cls.healthy ? ['No symptoms detected', 'Vibrant green foliage', 'Normal leaf texture'] : ['Brown leaf lesions', 'Yellowing tissue', 'Premature defoliation'],
    treatment: cls.healthy ? [] : ['Apply appropriate fungicide at 7–10 day intervals', 'Remove and destroy infected plant material', 'Improve air circulation by pruning', 'Use copper-based preventive sprays'],
    prevention: ['Monitor plants regularly', 'Use disease-resistant varieties', 'Practice crop rotation', 'Apply mulch to reduce soil splash', 'Ensure proper plant spacing'],
    watering_advice: 'Water at soil level using drip irrigation. Avoid wetting foliage. Water in the morning so leaves dry during daylight hours. Maintain consistent soil moisture.',
    fertilizer_recommendation: 'Apply balanced NPK fertilizer in spring. Supplement potassium for improved disease resistance. Avoid excessive nitrogen which promotes susceptible tender growth.',
  };
  return base;
}

// ── Render Results ─────────────────────────────────────────────────────────────
function renderResults(data) {
  const { prediction, severity, treatment, inference_time_ms, scan_id } = data;
  const { cnn, svm, is_healthy } = prediction;

  // Show results section
  el.results.style.display = 'block';
  el.scanArea.style.display = 'none';

  // Alert Banner
  const alertBanner = document.getElementById('alertBanner');
  const alertTitle = document.getElementById('alertTitle');
  const alertDesc = document.getElementById('alertDesc');
  const alertIcon = document.getElementById('alertIcon');
  alertBanner.style.display = 'flex';
  if (is_healthy) {
    alertBanner.className = 'alert-banner alert-healthy';
    alertIcon.textContent = '✅';
    alertTitle.textContent = 'Plant Appears Healthy!';
    alertDesc.textContent = 'No disease detected. Follow the care tips below to maintain plant health.';
  } else {
    alertBanner.className = 'alert-banner alert-diseased';
    alertIcon.textContent = '⚠️';
    alertTitle.textContent = 'Disease Detected!';
    alertDesc.textContent = `${treatment.common_name} identified. Scroll down for treatment recommendations.`;
  }

  // Disease Card
  document.getElementById('diseaseName').textContent = treatment.common_name || formatClass(cnn.class);
  document.getElementById('diseasePlant').textContent = `Plant: ${treatment.plant || 'Unknown'}`;
  document.getElementById('diseaseDesc').textContent = treatment.description || 'No description available.';
  document.getElementById('inferenceTime').textContent = inference_time_ms;
  document.getElementById('scanId').textContent = scan_id || '—';

  const diseaseBadge = document.getElementById('diseaseBadge');
  diseaseBadge.textContent = is_healthy ? '✓ Healthy' : '⚠ Diseased';
  diseaseBadge.className = `disease-badge ${is_healthy ? 'healthy-badge' : ''}`;

  // Severity Badge in disease card
  const sevBadge = document.getElementById('severityBadge');
  const sevClass = getSeverityClass(severity.level);
  sevBadge.textContent = capitalize(severity.level || 'N/A');
  sevBadge.className = `severity-badge ${sevClass}`;

  // CNN
  document.getElementById('cnnClass').textContent = formatClass(cnn.class);
  document.getElementById('cnnConf').textContent = `${cnn.confidence}%`;
  setTimeout(() => {
    document.getElementById('cnnBar').style.width = `${cnn.confidence}%`;
  }, 100);

  // SVM
  document.getElementById('svmClass').textContent = formatClass(svm.class);
  document.getElementById('svmConf').textContent = `${svm.confidence}%`;
  setTimeout(() => {
    document.getElementById('svmBar').style.width = `${svm.confidence}%`;
  }, 150);

  // Model Agreement
  const agreeDiv = document.getElementById('modelAgreement');
  const agrees = svm.agreement !== undefined ? svm.agreement : (svm.class === cnn.class);
  agreeDiv.className = `model-agreement ${agrees ? 'agree' : 'disagree'}`;
  agreeDiv.textContent = agrees
    ? '✓ Both models agree on the prediction'
    : '⚡ Models disagree — CNN result is primary';

  // Top-K
  const topkSection = document.getElementById('topkSection');
  const topkList = document.getElementById('topkList');
  const topK = cnn.top_predictions || cnn.top_k || [];
  if (topK.length > 0) {
    topkSection.style.display = 'block';
    topkList.innerHTML = '';
    topK.slice(0, 5).forEach(item => {
      const pct = typeof item.confidence === 'number' ? (item.confidence * 100).toFixed(1) : item.confidence;
      topkList.innerHTML += `
        <div class="topk-item">
          <span class="topk-name">${formatClass(item.class)}</span>
          <div class="topk-bar-wrap"><div class="topk-bar-fill" style="width:${pct}%"></div></div>
          <span class="topk-pct">${pct}%</span>
        </div>`;
    });
  }

  // Severity Arc
  const score = severity.score || 0;
  document.getElementById('arcScore').textContent = score;
  document.getElementById('affectedPct').textContent = `${severity.affected_percentage || 0}%`;
  document.getElementById('greenIndex').textContent =
    severity.color_analysis?.greenness_index !== undefined
      ? severity.color_analysis.greenness_index.toFixed(1)
      : '—';

  // Arc animation
  setTimeout(() => {
    const arc = document.getElementById('arcFill');
    const circumference = 251; // Arc path length
    const filled = (score / 100) * circumference;
    arc.style.strokeDasharray = `${filled} ${circumference - filled}`;
    arc.style.stroke = score < 20 ? 'var(--green-500)' : score < 50 ? '#fbbf24' : '#ef4444';
  }, 200);

  const sevLevelBadge = document.getElementById('severityLevelBadge');
  sevLevelBadge.textContent = capitalize(severity.level || 'N/A');
  sevLevelBadge.className = `severity-level-badge ${sevClass}`;

  // Treatment content
  populateList('causesList', treatment.causes || []);
  populateList('treatmentList', treatment.treatment || []);
  populateList('preventionList', treatment.prevention || []);
  document.getElementById('wateringText').textContent = treatment.watering_advice || '—';
  document.getElementById('fertilizerText').textContent = treatment.fertilizer_recommendation || '—';

  // Healthy Card
  const healthyCard = document.getElementById('healthyCard');
  const healthyTips = document.getElementById('healthyTips');
  if (is_healthy) {
    healthyCard.style.display = 'block';
    healthyTips.innerHTML = '';
    (treatment.prevention || []).slice(0, 5).forEach(tip => {
      healthyTips.innerHTML += `<p class="healthy-tip">${tip}</p>`;
    });
  } else {
    healthyCard.style.display = 'none';
  }

  // Scroll to top of results
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ── Treatment Tabs ────────────────────────────────────────────────────────────
function switchTab(tab) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
  document.getElementById('tabTreatment').style.display = tab === 'treatment' ? 'block' : 'none';
  document.getElementById('tabPrevention').style.display = tab === 'prevention' ? 'block' : 'none';
  document.getElementById('tabCare').style.display = tab === 'care' ? 'block' : 'none';
}

// ── Model Comparison ─────────────────────────────────────────────────────────
async function fetchComparisonData() {
  try {
    const res = await fetch(`${API_BASE}/api/compare`);
    const data = await res.json();
    if (data.success && data.data) {
      renderComparison(data.data);
    }
  } catch {
    // Silently fail; demo values are already shown via placeholder
  }
}

function renderComparison(data) {
  const pct = v => v !== undefined ? `${parseFloat(v).toFixed(1)}%` : '—';
  const secs = v => v !== undefined ? `${v}s` : '—';

  setEl('cmpCnnAcc', pct(data.cnn?.accuracy));
  setEl('cmpCnnPrec', pct(data.cnn?.precision));
  setEl('cmpCnnRec', pct(data.cnn?.recall));
  setEl('cmpCnnF1', pct(data.cnn?.f1_score));
  setEl('cmpCnnTime', secs(data.cnn?.training_time_seconds));

  setEl('cmpSvmAcc', pct(data.svm?.accuracy));
  setEl('cmpSvmPrec', pct(data.svm?.precision));
  setEl('cmpSvmRec', pct(data.svm?.recall));
  setEl('cmpSvmF1', pct(data.svm?.f1_score));
  setEl('cmpSvmTime', secs(data.svm?.training_time_seconds));

  const summary = document.getElementById('compareSummary');
  const summaryText = document.getElementById('csSummaryText');
  if (data.summary) {
    summary.style.display = 'flex';
    summaryText.textContent = data.summary;
  }
}

// ── Dashboard / Stats ─────────────────────────────────────────────────────────
async function fetchStats() {
  try {
    const res = await fetch(`${API_BASE}/api/stats`);
    const data = await res.json();
    if (data.success) {
      renderStats(data.stats, data.recent_scans);
    }
  } catch {
    // Silently fail
  }
}

function renderStats(stats, scans) {
  setEl('statTotal', stats.total_scans);
  setEl('statHealthy', stats.healthy_count);
  setEl('statDiseased', stats.diseased_count);
  setEl('statConfidence', stats.avg_confidence ? `${stats.avg_confidence}%` : '—');

  const tbody = el.historyTableBody;
  if (!scans || scans.length === 0) {
    tbody.innerHTML = `<tr class="empty-row"><td colspan="7">No scans yet. Upload a leaf image to get started.</td></tr>`;
    return;
  }

  tbody.innerHTML = '';
  scans.forEach(scan => {
    const date = new Date(scan.timestamp);
    const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    tbody.innerHTML += `
      <tr>
        <td class="mono">${scan.id}</td>
        <td>${scan.plant || '—'}</td>
        <td>${scan.disease || '—'}</td>
        <td><span class="${scan.healthy ? 'badge-h' : 'badge-d'}">${scan.healthy ? 'Healthy' : 'Diseased'}</span></td>
        <td class="mono">${scan.confidence}%</td>
        <td><span class="severity-badge ${getSeverityClass(scan.severity)}">${capitalize(scan.severity)}</span></td>
        <td class="mono">${timeStr}</td>
      </tr>`;
  });
}

// ── Navigation ───────────────────────────────────────────────────────────────
function scrollToSection(section) {
  const sectionMap = {
    scan: 'scanArea',
    results: 'results',
    compare: 'compare',
    dashboard: 'dashboard',
  };
  const id = sectionMap[section] || section;
  const el_s = document.getElementById(id);
  if (el_s) {
    const offset = 80; // Navbar height
    const top = el_s.getBoundingClientRect().top + window.scrollY - offset;
    window.scrollTo({ top, behavior: 'smooth' });
  }
}

function resetToScan() {
  el.results.style.display = 'none';
  el.scanArea.style.display = 'block';
  state.selectedFile = null;
  el.dzDefault.style.display = 'flex';
  el.dzPreview.style.display = 'none';
  el.analyzeBtn.disabled = true;
  el.fileInput.value = '';
  window.scrollTo({ top: 0, behavior: 'smooth' });
  fetchStats();
}

// ── Loading ──────────────────────────────────────────────────────────────────
function showLoading() {
  el.loadingOverlay.style.display = 'flex';
  el.analyzeBtn.disabled = true;
  document.querySelectorAll('.ls-step').forEach(s => s.classList.remove('active', 'done'));
}
function hideLoading() {
  el.loadingOverlay.style.display = 'none';
  el.analyzeBtn.disabled = false;
}

// ── Utilities ────────────────────────────────────────────────────────────────
function formatClass(cls) {
  if (!cls) return '—';
  return cls
    .replace(/___/g, ' — ')
    .replace(/_/g, ' ')
    .replace(/\s+\(.*?\)\s*/g, ' ')
    .trim();
}

function capitalize(str) {
  if (!str) return '—';
  return str.charAt(0).toUpperCase() + str.slice(1);
}

function getSeverityClass(level) {
  const map = { healthy: 'sev-healthy', mild: 'sev-mild', moderate: 'sev-moderate', severe: 'sev-severe' };
  return map[(level || '').toLowerCase()] || 'sev-mild';
}

function populateList(id, items) {
  const ul = document.getElementById(id);
  if (!ul) return;
  if (!items || items.length === 0) {
    ul.innerHTML = '<li>No information available.</li>';
    return;
  }
  ul.innerHTML = items.map(item => `<li>${item}</li>`).join('');
}

function setEl(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function delay(ms) {
  return new Promise(r => setTimeout(r, ms));
}

function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  const colors = { info: '#10b981', error: '#ef4444', warn: '#f59e0b' };
  toast.style.cssText = `
    position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
    background: rgba(13, 31, 22, 0.95); border: 1px solid ${colors[type] || colors.info}40;
    color: ${colors[type] || colors.info}; padding: 12px 24px; border-radius: 999px;
    font-size: 0.85rem; font-family: 'DM Sans', sans-serif; z-index: 999;
    backdrop-filter: blur(20px); box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    animation: slideUp 0.3s ease; max-width: 90vw; text-align: center;
  `;
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => { toast.style.animation = 'slideUp 0.3s ease reverse'; setTimeout(() => toast.remove(), 300); }, 4000);
}

// Inject toast keyframe
const style = document.createElement('style');
style.textContent = `@keyframes slideUp { from { opacity:0; transform: translateX(-50%) translateY(20px); } to { opacity:1; transform: translateX(-50%) translateY(0); } }`;
document.head.appendChild(style);
