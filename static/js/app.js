/**
 * EmotiNet - Frontend Application Script
 * Handles real-time inference, Chart.js visualizations, Speech-to-Text,
 * batch analysis, local history, and responsive themes.
 */

// State Management
const state = {
  theme: localStorage.getItem('emotinet_theme') || 'dark',
  currentTab: 'single',
  radarChart: null,
  history: JSON.parse(localStorage.getItem('emotinet_history') || '[]'),
  isRecording: false,
  recognition: null
};

// Emotion metadata mapping for styling
const EMOTIONS_CONFIG = {
  joy: { label: "Joy", emoji: "😊", color: "#F59E0B", rgb: "245, 158, 11" },
  sadness: { label: "Sadness", emoji: "😢", color: "#3B82F6", rgb: "59, 130, 246" },
  love: { label: "Love", emoji: "💖", color: "#EC4899", rgb: "236, 72, 153" },
  anger: { label: "Anger", emoji: "🔥", color: "#EF4444", rgb: "239, 68, 68" },
  fear: { label: "Fear", emoji: "⚡", color: "#8B5CF6", rgb: "139, 92, 246" },
  surprise: { label: "Surprise", emoji: "😲", color: "#06B6D4", rgb: "6, 182, 212" }
};

// DOM Elements
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initTabs();
  initTextCounters();
  initQuickPrompts();
  initKeyboardShortcuts();
  initSpeechRecognition();
  initHistoryUI();
  initCodeCopyButtons();
  
  // Attach single prediction event
  const analyzeBtn = document.getElementById('analyze-btn');
  if (analyzeBtn) {
    analyzeBtn.addEventListener('click', handleSinglePrediction);
  }

  // Clear button
  const clearBtn = document.getElementById('clear-btn');
  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      const input = document.getElementById('text-input');
      input.value = '';
      updateTextCounters();
      input.focus();
    });
  }

  // Paste button
  const pasteBtn = document.getElementById('paste-btn');
  if (pasteBtn) {
    pasteBtn.addEventListener('click', async () => {
      try {
        const text = await navigator.clipboard.readText();
        const input = document.getElementById('text-input');
        input.value = text;
        updateTextCounters();
        showToast('Text pasted from clipboard!', 'info');
      } catch (err) {
        showToast('Clipboard access denied.', 'error');
      }
    });
  }

  // Batch analyze button
  const batchAnalyzeBtn = document.getElementById('batch-analyze-btn');
  if (batchAnalyzeBtn) {
    batchAnalyzeBtn.addEventListener('click', handleBatchPrediction);
  }

  // Batch file upload
  const fileUpload = document.getElementById('batch-file-input');
  if (fileUpload) {
    fileUpload.addEventListener('change', handleFileUpload);
  }

  // Export batch buttons
  const exportCsvBtn = document.getElementById('export-csv-btn');
  if (exportCsvBtn) {
    exportCsvBtn.addEventListener('click', exportBatchToCSV);
  }

  const exportJsonBtn = document.getElementById('export-json-btn');
  if (exportJsonBtn) {
    exportJsonBtn.addEventListener('click', exportBatchToJSON);
  }

  // Check API health on load
  checkServerHealth();
});

/* ==========================================================================
   Theme Management
   ========================================================================== */
function initTheme() {
  document.documentElement.setAttribute('data-theme', state.theme);
  const themeToggle = document.getElementById('theme-toggle');
  updateThemeIcon();

  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      state.theme = state.theme === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', state.theme);
      localStorage.setItem('emotinet_theme', state.theme);
      updateThemeIcon();
      
      // Re-render chart with new theme grid colors
      if (state.radarChart && state.lastRadarData) {
        renderRadarChart(state.lastRadarData);
      }
    });
  }
}

function updateThemeIcon() {
  const sunIcon = document.getElementById('sun-icon');
  const moonIcon = document.getElementById('moon-icon');
  if (!sunIcon || !moonIcon) return;

  if (state.theme === 'light') {
    sunIcon.classList.add('hidden');
    moonIcon.classList.remove('hidden');
  } else {
    sunIcon.classList.remove('hidden');
    moonIcon.classList.add('hidden');
  }
}

/* ==========================================================================
   Tab Navigation
   ========================================================================== */
function initTabs() {
  const tabButtons = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');

  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.getAttribute('data-tab');
      state.currentTab = target;

      tabButtons.forEach(b => b.classList.remove('active', 'border-indigo-500', 'bg-indigo-500/10'));
      tabContents.forEach(c => c.classList.add('hidden'));

      btn.classList.add('active', 'border-indigo-500', 'bg-indigo-500/10');
      const targetContent = document.getElementById(`tab-content-${target}`);
      if (targetContent) {
        targetContent.classList.remove('hidden');
      }

      // Re-trigger chart resize if tab switched to single
      if (target === 'single' && state.radarChart) {
        state.radarChart.resize();
      }
    });
  });
}

/* ==========================================================================
   Text Area Counters & Shortcuts
   ========================================================================== */
function initTextCounters() {
  const input = document.getElementById('text-input');
  if (!input) return;

  input.addEventListener('input', updateTextCounters);
}

function updateTextCounters() {
  const input = document.getElementById('text-input');
  const charCounter = document.getElementById('char-count');
  const wordCounter = document.getElementById('word-count');
  if (!input) return;

  const text = input.value;
  const chars = text.length;
  const words = text.trim() ? text.trim().split(/\s+/).length : 0;

  if (charCounter) charCounter.textContent = `${chars} / 2000`;
  if (wordCounter) wordCounter.textContent = `${words} words`;
}

function initKeyboardShortcuts() {
  const input = document.getElementById('text-input');
  if (!input) return;

  input.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      handleSinglePrediction();
    }
  });
}

function initQuickPrompts() {
  const pills = document.querySelectorAll('.prompt-pill');
  const input = document.getElementById('text-input');

  pills.forEach(pill => {
    pill.addEventListener('click', () => {
      const sampleText = pill.getAttribute('data-sample');
      if (input && sampleText) {
        input.value = sampleText;
        updateTextCounters();
        handleSinglePrediction();
      }
    });
  });
}

/* ==========================================================================
   Speech-to-Text Recognition
   ========================================================================== */
function initSpeechRecognition() {
  const micBtn = document.getElementById('mic-btn');
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  if (!SpeechRecognition) {
    if (micBtn) {
      micBtn.title = "Speech recognition not supported in this browser";
      micBtn.classList.add('opacity-40', 'cursor-not-allowed');
    }
    return;
  }

  const recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = false;
  recognition.lang = 'en-US';

  recognition.onstart = () => {
    state.isRecording = true;
    micBtn.classList.add('recording-pulse');
    showToast('Listening... Speak now!', 'info');
  };

  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    const input = document.getElementById('text-input');
    if (input) {
      input.value = transcript;
      updateTextCounters();
      handleSinglePrediction();
    }
  };

  recognition.onerror = (event) => {
    state.isRecording = false;
    micBtn.classList.remove('recording-pulse');
    showToast(`Speech error: ${event.error}`, 'error');
  };

  recognition.onend = () => {
    state.isRecording = false;
    micBtn.classList.remove('recording-pulse');
  };

  if (micBtn) {
    micBtn.addEventListener('click', () => {
      if (state.isRecording) {
        recognition.stop();
      } else {
        recognition.start();
      }
    });
  }
}

/* ==========================================================================
   Single Prediction Flow
   ========================================================================== */
async function handleSinglePrediction() {
  const input = document.getElementById('text-input');
  const analyzeBtn = document.getElementById('analyze-btn');
  const spinner = document.getElementById('btn-spinner');
  const btnText = document.getElementById('btn-text');

  if (!input) return;
  const text = input.value.trim();

  if (!text) {
    showToast('Please type or speak some text first!', 'warning');
    input.focus();
    return;
  }

  // Loading UI
  if (analyzeBtn) analyzeBtn.disabled = true;
  if (spinner) spinner.classList.remove('hidden');
  if (btnText) btnText.textContent = 'Classifying...';

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });

    if (!response.ok) {
      const errData = await response.json();
      throw new Error(errData.detail || 'Prediction failed');
    }

    const data = await response.json();
    renderPredictionResults(data);
    addToHistory(data);
  } catch (error) {
    console.error(error);
    showToast(`Error: ${error.message}`, 'error');
  } finally {
    if (analyzeBtn) analyzeBtn.disabled = false;
    if (spinner) spinner.classList.add('hidden');
    if (btnText) btnText.textContent = 'Analyze Emotion';
  }
}

/* ==========================================================================
   Render Prediction Results
   ========================================================================== */
function renderPredictionResults(data) {
  const placeholder = document.getElementById('results-placeholder');
  const content = document.getElementById('results-content');
  if (placeholder) placeholder.classList.add('hidden');
  if (content) content.classList.remove('hidden');

  // Hero Card Elements
  const heroEmoji = document.getElementById('hero-emoji');
  const heroEmotion = document.getElementById('hero-emotion');
  const heroConfidence = document.getElementById('hero-confidence');
  const heroSentiment = document.getElementById('hero-sentiment');
  const heroIntensity = document.getElementById('hero-intensity');
  const heroDescription = document.getElementById('hero-description');
  const heroSpeed = document.getElementById('hero-speed');
  const heroCard = document.getElementById('hero-card');

  if (heroEmoji) heroEmoji.textContent = data.emoji;
  if (heroEmotion) {
    heroEmotion.textContent = data.predicted_emotion.toUpperCase();
    heroEmotion.style.color = data.color;
  }
  if (heroConfidence) {
    heroConfidence.textContent = `${data.confidence_percentage}% Confidence`;
    heroConfidence.style.borderColor = data.color;
  }
  if (heroSentiment) heroSentiment.textContent = data.sentiment;
  if (heroIntensity) heroIntensity.textContent = data.intensity;
  if (heroDescription) heroDescription.textContent = data.description;
  if (heroSpeed) heroSpeed.textContent = `⚡ ${data.processing_time_ms} ms`;

  // Dynamic card glow
  if (heroCard) {
    heroCard.style.boxShadow = `0 10px 40px -10px ${data.color}40`;
    heroCard.style.borderColor = `${data.color}60`;
  }

  // Render Horizontal Emotion Bars
  renderEmotionBars(data.breakdown);

  // Render Radar Chart
  renderRadarChart(data.probabilities);
}

function renderEmotionBars(breakdown) {
  const container = document.getElementById('emotion-bars-container');
  if (!container) return;

  container.innerHTML = '';

  breakdown.forEach(item => {
    const row = document.createElement('div');
    row.className = 'flex flex-col gap-1.5';
    row.innerHTML = `
      <div class="flex items-center justify-between text-sm font-medium">
        <span class="flex items-center gap-2">
          <span>${item.emoji}</span>
          <span class="capitalize">${item.label}</span>
          <span class="text-xs px-1.5 py-0.5 rounded text-gray-400 bg-white/5">${item.sentiment}</span>
        </span>
        <span class="font-mono text-xs font-semibold" style="color: ${item.color}">${item.percentage}%</span>
      </div>
      <div class="w-full bg-white/5 rounded-full h-2.5 overflow-hidden border border-white/5">
        <div class="progress-fill h-full rounded-full" style="width: 0%; background-color: ${item.color}"></div>
      </div>
    `;
    container.appendChild(row);

    // Trigger smooth fill animation
    setTimeout(() => {
      const fillBar = row.querySelector('.progress-fill');
      if (fillBar) fillBar.style.width = `${Math.max(item.percentage, 2)}%`;
    }, 50);
  });
}

function renderRadarChart(probabilities) {
  const canvas = document.getElementById('emotion-radar-chart');
  if (!canvas) return;

  state.lastRadarData = probabilities;
  const isDark = state.theme === 'dark';
  const gridColor = isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
  const pointLabelColor = isDark ? '#cbd5e1' : '#334155';

  const labels = ['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise'];
  const values = [
    (probabilities.sadness || 0) * 100,
    (probabilities.joy || 0) * 100,
    (probabilities.love || 0) * 100,
    (probabilities.anger || 0) * 100,
    (probabilities.fear || 0) * 100,
    (probabilities.surprise || 0) * 100
  ];

  if (state.radarChart) {
    state.radarChart.destroy();
  }

  const ctx = canvas.getContext('2d');
  state.radarChart = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Emotion Spectrum (%)',
        data: values,
        backgroundColor: 'rgba(99, 102, 241, 0.25)',
        borderColor: '#6366F1',
        borderWidth: 2,
        pointBackgroundColor: '#EC4899',
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: '#6366F1',
        pointRadius: 4,
        pointHoverRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          angleLines: { color: gridColor },
          grid: { color: gridColor },
          pointLabels: {
            font: { family: 'Plus Jakarta Sans', size: 12, weight: 600 },
            color: pointLabelColor
          },
          suggestedMin: 0,
          suggestedMax: 100,
          ticks: {
            stepSize: 25,
            backdropColor: 'transparent',
            color: isDark ? '#64748b' : '#94a3b8',
            font: { size: 10 }
          }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.label}: ${ctx.raw.toFixed(1)}%`
          }
        }
      }
    }
  });
}

/* ==========================================================================
   Batch Analysis & File Upload
   ========================================================================== */
let batchResultsData = [];

function handleFileUpload(e) {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (event) => {
    const text = event.target.result;
    const batchInput = document.getElementById('batch-text-input');
    if (batchInput) {
      batchInput.value = text;
      showToast(`Loaded ${file.name} successfully!`, 'info');
    }
  };
  reader.readAsText(file);
}

async function handleBatchPrediction() {
  const batchInput = document.getElementById('batch-text-input');
  const batchBtn = document.getElementById('batch-analyze-btn');
  if (!batchInput) return;

  const rawText = batchInput.value.trim();
  if (!rawText) {
    showToast('Please paste or upload text samples first!', 'warning');
    return;
  }

  // Split lines into non-empty sentences
  const lines = rawText
    .split('\n')
    .map(l => l.trim())
    .filter(l => l.length > 0);

  if (lines.length === 0) {
    showToast('No valid sentences found to process.', 'warning');
    return;
  }

  if (batchBtn) batchBtn.disabled = true;
  showToast(`Processing ${lines.length} sentences...`, 'info');

  try {
    const response = await fetch('/api/predict-batch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texts: lines.slice(0, 100) }) // Max 100
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || 'Batch analysis failed');
    }

    const data = await response.json();
    batchResultsData = data.results;
    renderBatchResults(data);
    showToast(`Successfully analyzed ${data.total} items!`, 'success');
  } catch (error) {
    console.error(error);
    showToast(`Batch Error: ${error.message}`, 'error');
  } finally {
    if (batchBtn) batchBtn.disabled = false;
  }
}

function renderBatchResults(data) {
  const resultsCard = document.getElementById('batch-results-card');
  const tableBody = document.getElementById('batch-table-body');
  const countBadge = document.getElementById('batch-total-count');
  const dominantBadge = document.getElementById('batch-dominant-emotion');
  const speedBadge = document.getElementById('batch-speed');

  if (resultsCard) resultsCard.classList.remove('hidden');
  if (countBadge) countBadge.textContent = `${data.total} Items`;
  if (dominantBadge) {
    const cfg = EMOTIONS_CONFIG[data.dominant_emotion] || {};
    dominantBadge.textContent = `${cfg.emoji || '✨'} ${data.dominant_emotion.toUpperCase()}`;
    dominantBadge.style.color = cfg.color || '#fff';
  }
  if (speedBadge) speedBadge.textContent = `⚡ ${data.processing_time_ms} ms`;

  if (tableBody) {
    tableBody.innerHTML = '';
    data.results.forEach((item, index) => {
      const cfg = EMOTIONS_CONFIG[item.predicted_emotion] || {};
      const tr = document.createElement('tr');
      tr.className = 'border-b border-white/5 hover:bg-white/5 transition-colors';
      tr.innerHTML = `
        <td class="py-3 px-4 font-mono text-xs text-gray-400">${index + 1}</td>
        <td class="py-3 px-4 text-sm max-w-xs truncate" title="${item.text}">${item.text}</td>
        <td class="py-3 px-4">
          <span class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold" style="background: ${cfg.color}20; color: ${cfg.color}; border: 1px solid ${cfg.color}40">
            <span>${item.emoji}</span>
            <span class="capitalize">${item.predicted_emotion}</span>
          </span>
        </td>
        <td class="py-3 px-4 text-xs font-mono font-medium">${item.confidence_percentage}%</td>
        <td class="py-3 px-4 text-xs text-gray-400">${item.sentiment}</td>
      `;
      tableBody.appendChild(tr);
    });
  }
}

function exportBatchToCSV() {
  if (!batchResultsData || batchResultsData.length === 0) {
    showToast('No batch results to export.', 'warning');
    return;
  }

  let csv = 'ID,Text,Predicted_Emotion,Confidence_Percentage,Sentiment\n';
  batchResultsData.forEach((r, i) => {
    const sanitizedText = `"${r.text.replace(/"/g, '""')}"`;
    csv += `${i + 1},${sanitizedText},${r.predicted_emotion},${r.confidence_percentage},${r.sentiment}\n`;
  });

  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `emotinet_results_${Date.now()}.csv`;
  a.click();
  URL.revokeObjectURL(url);
  showToast('Exported CSV file!', 'success');
}

function exportBatchToJSON() {
  if (!batchResultsData || batchResultsData.length === 0) {
    showToast('No batch results to export.', 'warning');
    return;
  }

  const jsonStr = JSON.stringify(batchResultsData, null, 2);
  const blob = new Blob([jsonStr], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `emotinet_results_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
  showToast('Exported JSON file!', 'success');
}

/* ==========================================================================
   History Management
   ========================================================================== */
function addToHistory(data) {
  const item = {
    id: Date.now(),
    text: data.text,
    emotion: data.predicted_emotion,
    emoji: data.emoji,
    confidence: data.confidence_percentage,
    color: data.color,
    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  };

  state.history.unshift(item);
  if (state.history.length > 20) state.history.pop();
  localStorage.setItem('emotinet_history', JSON.stringify(state.history));
  renderHistoryList();
}

function initHistoryUI() {
  renderHistoryList();
  const clearHistoryBtn = document.getElementById('clear-history-btn');
  if (clearHistoryBtn) {
    clearHistoryBtn.addEventListener('click', () => {
      state.history = [];
      localStorage.removeItem('emotinet_history');
      renderHistoryList();
      showToast('History cleared', 'info');
    });
  }
}

function renderHistoryList() {
  const container = document.getElementById('history-list');
  const countBadge = document.getElementById('history-count');
  if (!container) return;

  if (countBadge) countBadge.textContent = state.history.length;

  if (state.history.length === 0) {
    container.innerHTML = `
      <div class="p-6 text-center text-sm text-gray-500">
        <i data-lucide="clock" class="w-8 h-8 mx-auto mb-2 opacity-30"></i>
        <p>No recent predictions yet.</p>
      </div>
    `;
    if (window.lucide) lucide.createIcons();
    return;
  }

  container.innerHTML = '';
  state.history.forEach(item => {
    const div = document.createElement('div');
    div.className = 'p-3 rounded-lg border border-white/5 bg-white/5 hover:border-indigo-500/30 cursor-pointer transition-all flex items-center justify-between gap-3';
    div.innerHTML = `
      <div class="flex-1 min-w-0">
        <p class="text-xs text-gray-300 truncate">${item.text}</p>
        <span class="text-[10px] text-gray-500">${item.timestamp}</span>
      </div>
      <div class="flex items-center gap-1.5 shrink-0">
        <span class="text-sm">${item.emoji}</span>
        <span class="text-xs font-semibold capitalize font-mono" style="color: ${item.color}">${item.confidence}%</span>
      </div>
    `;
    div.addEventListener('click', () => {
      const input = document.getElementById('text-input');
      if (input) {
        input.value = item.text;
        updateTextCounters();
        handleSinglePrediction();
        // Switch to single tab if on another
        const singleTabBtn = document.querySelector('[data-tab="single"]');
        if (singleTabBtn) singleTabBtn.click();
      }
    });
    container.appendChild(div);
  });
}

/* ==========================================================================
   Code Snippets & Copy
   ========================================================================== */
function initCodeCopyButtons() {
  const copyButtons = document.querySelectorAll('.copy-code-btn');
  copyButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-target');
      const codeBlock = document.getElementById(targetId);
      if (codeBlock) {
        navigator.clipboard.writeText(codeBlock.textContent.trim());
        showToast('Code copied to clipboard!', 'success');
      }
    });
  });
}

/* ==========================================================================
   Toast Notifications
   ========================================================================== */
function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  const colors = {
    info: 'border-blue-500/50 bg-slate-900/90 text-blue-200',
    success: 'border-emerald-500/50 bg-slate-900/90 text-emerald-200',
    warning: 'border-amber-500/50 bg-slate-900/90 text-amber-200',
    error: 'border-rose-500/50 bg-slate-900/90 text-rose-200'
  };

  toast.className = `flex items-center gap-2 px-4 py-3 rounded-xl border backdrop-blur-md shadow-xl text-sm transition-all transform duration-300 translate-y-2 opacity-0 ${colors[type] || colors.info}`;
  toast.innerHTML = `<span>${message}</span>`;
  container.appendChild(toast);

  // Trigger animate in
  setTimeout(() => {
    toast.classList.remove('translate-y-2', 'opacity-0');
  }, 10);

  // Auto dismiss
  setTimeout(() => {
    toast.classList.add('opacity-0', 'translate-y-2');
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

/* ==========================================================================
   Server Health Check
   ========================================================================== */
async function checkServerHealth() {
  try {
    const res = await fetch('/health');
    const data = await res.json();
    const statusPill = document.getElementById('status-pill');
    const statusText = document.getElementById('status-text');

    if (data.status === 'healthy') {
      if (statusPill) statusPill.className = 'w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse';
      if (statusText) statusText.textContent = data.model_loaded ? 'BiGRU Online' : 'Engine Ready';
    }
  } catch (e) {
    const statusPill = document.getElementById('status-pill');
    const statusText = document.getElementById('status-text');
    if (statusPill) statusPill.className = 'w-2.5 h-2.5 rounded-full bg-rose-500';
    if (statusText) statusText.textContent = 'Server Offline';
  }
}
