/**
 * DPO Rank Mode — tri-state: GOOD / NEUTRAL / BAD.
 *
 * Click each completion to cycle:  ○ → ★ (good) → ✗ (bad) → ○
 * When you're done, click "Save pairs" to generate all good×bad combinations.
 * Neutral items are excluded entirely.
 *
 * Also supports legacy "Rank all 5" mode (toggle in UI) for when all
 * completions are on a smooth quality spectrum.
 */
let candidates = [];
let currentIdx = 0;
let pairs = [];
let rankings = []; // exported for audit trail

// state per prompt — Map<completionIdx, 'good' | 'bad' | null>
let marks = new Map();
let fullRankMode = false;
let fullRankings = new Map(); // completionIdx → rank (1=best)
let nextRank = 1;

const uploadArea = document.getElementById('upload-area');
const fileInput  = document.getElementById('file-input');
const rankingUi  = document.getElementById('ranking-ui');
const promptDisplay = document.getElementById('prompt-display');
const completionsList = document.getElementById('completions-list');
const promptIdxEl = document.getElementById('prompt-idx');
const promptTotalEl = document.getElementById('prompt-total');
const counterEl = document.getElementById('counter');
const exportBtn = document.getElementById('export-btn');
const skipBtn = document.getElementById('skip-btn');

fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (ev) => {
    try {
      candidates = JSON.parse(ev.target.result);
      if (!Array.isArray(candidates) || candidates.length === 0) {
        alert('JSON must be a non-empty array of {prompt, completions} objects.');
        return;
      }
      uploadArea.classList.add('has-file');
      uploadArea.innerHTML = `<p>✅ Loaded <strong>${candidates.length}</strong> prompts</p>`;
      rankingUi.classList.remove('hidden');
      promptTotalEl.textContent = candidates.length;
      renderCurrent();
    } catch (err) {
      alert('Invalid JSON: ' + err.message);
    }
  };
  reader.readAsText(file);
});

function renderCurrent() {
  if (currentIdx >= candidates.length) {
    promptDisplay.innerHTML = '<strong>All done!</strong> Every prompt has been processed.';
    completionsList.innerHTML = '';
    promptIdxEl.textContent = candidates.length;
    skipBtn.disabled = true;
    exportBtn.disabled = pairs.length === 0;
    return;
  }

  resetState();
  const item = candidates[currentIdx];
  const n = item.completions.length;

  promptDisplay.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem;">
      <span><strong>Prompt:</strong> ${escapeHtml(item.prompt)}</span>
      <label style="font-size:0.8rem;color:#888;cursor:pointer;display:flex;align-items:center;gap:0.5rem;">
        <input type="checkbox" id="mode-check" ${fullRankMode ? 'checked' : ''}>
        Rank all ${n}
      </label>
    </div>
    <div style="font-size:0.8rem;color:#666;">
      ${fullRankMode
        ? 'Click from <strong style="color:#4a9;">BEST (1)</strong> to <strong style="color:#a44;">WORST (' + n + ')</strong>'
        : 'Click each card to cycle:  ○ → <strong style="color:#4a9;">★ good</strong> → <strong style="color:#a44;">✗ bad</strong> → ○'}
    </div>
  `;

  document.getElementById('mode-check').addEventListener('change', (e) => {
    fullRankMode = e.target.checked;
    resetState();
    renderCards(item);
  });

  promptIdxEl.textContent = currentIdx + 1;
  renderCards(item);
  updateExportButton();
}

function resetState() {
  marks = new Map();
  fullRankings = new Map();
  nextRank = 1;
}

function renderCards(item) {
  completionsList.innerHTML = item.completions.map((c, i) => {
    if (fullRankMode) {
      const rank = fullRankings.get(i);
      let badge = '';
      let cls = 'completion';
      if (rank !== undefined) {
        badge = `<div class="rank-badge" style="background:${rankColor(rank, item.completions.length)};color:#fff;">${rank}</div>`;
        if (rank === 1) cls += ' chosen';
        if (rank === item.completions.length) cls += ' rejected';
      }
      return `
        <div class="${cls}" data-idx="${i}" onclick="assignRank(${i})">
          ${badge}
          <div style="flex:1;">
            <div class="completion-text">${escapeHtml(c.text)}</div>
            <div class="completion-meta">temp=${c.temperature}  seed=${c.seed}</div>
          </div>
        </div>
      `;
    } else {
      // tri-state: good / bad / neutral
      const mark = marks.get(i);
      let cls = 'completion';
      let indicator = '<div class="tri-state" style="width:2rem;height:2rem;border-radius:50%;border:2px solid #444;display:flex;align-items:center;justify-content:center;font-size:1rem;color:#666;flex-shrink:0;margin-right:0.75rem;">○</div>';
      if (mark === 'good') {
        cls += ' chosen';
        indicator = '<div class="tri-state" style="width:2rem;height:2rem;border-radius:50%;border:2px solid #4a9;background:#1a2a20;display:flex;align-items:center;justify-content:center;font-size:1rem;color:#4a9;flex-shrink:0;margin-right:0.75rem;">★</div>';
      } else if (mark === 'bad') {
        cls += ' rejected';
        indicator = '<div class="tri-state" style="width:2rem;height:2rem;border-radius:50%;border:2px solid #a44;background:#2a1a1a;display:flex;align-items:center;justify-content:center;font-size:1rem;color:#a44;flex-shrink:0;margin-right:0.75rem;">✗</div>';
      }
      return `
        <div class="${cls}" data-idx="${i}" onclick="cycleMark(${i})">
          ${indicator}
          <div style="flex:1;">
            <div class="completion-text">${escapeHtml(c.text)}</div>
            <div class="completion-meta">temp=${c.temperature}  seed=${c.seed}</div>
          </div>
        </div>
      `;
    }
  }).join('');
}

function cycleMark(idx) {
  const current = marks.get(idx);
  if (current === undefined || current === null) {
    marks.set(idx, 'good');
  } else if (current === 'good') {
    marks.set(idx, 'bad');
  } else {
    marks.set(idx, null);
  }
  renderCards(candidates[currentIdx]);
  updateExportButton();
}

function assignRank(idx) {
  if (fullRankings.has(idx)) return;
  fullRankings.set(idx, nextRank);
  nextRank++;
  renderCards(candidates[currentIdx]);
  if (fullRankings.size === candidates[currentIdx].completions.length) {
    setTimeout(() => saveFullRanking(), 300);
  }
}

function saveFullRanking() {
  const item = candidates[currentIdx];
  const n = item.completions.length;
  const rankedIndices = Array.from(fullRankings.entries())
    .sort((a, b) => a[1] - b[1])
    .map(([idx]) => idx);
  const rankingTexts = rankedIndices.map(i => item.completions[i].text);

  rankings.push({ prompt: item.prompt, ranking: rankingTexts });

  let newPairs = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      pairs.push({
        prompt: item.prompt,
        chosen: rankingTexts[i],
        rejected: rankingTexts[j],
        rank_gap: j - i
      });
      newPairs++;
    }
  }
  counterEl.textContent = pairs.length;
  currentIdx++;
  renderCurrent();
}

function updateExportButton() {
  if (fullRankMode) return;
  const goodCount = Array.from(marks.values()).filter(m => m === 'good').length;
  const badCount  = Array.from(marks.values()).filter(m => m === 'bad').length;
  // enable save button only when we have at least one good and one bad
  skipBtn.textContent = (goodCount > 0 && badCount > 0) ? 'Save pairs →' : 'Skip prompt →';
  skipBtn.onclick = (goodCount > 0 && badCount > 0) ? saveTriStatePairs : skipPrompt;
}

function saveTriStatePairs() {
  const item = candidates[currentIdx];
  const goodIndices = [];
  const badIndices = [];
  for (const [idx, mark] of marks) {
    if (mark === 'good') goodIndices.push(idx);
    if (mark === 'bad') badIndices.push(idx);
  }
  if (goodIndices.length === 0 || badIndices.length === 0) return;

  const goodTexts = goodIndices.map(i => item.completions[i].text);
  const badTexts  = badIndices.map(i => item.completions[i].text);

  rankings.push({ prompt: item.prompt, good: goodTexts, bad: badTexts });

  let newPairs = 0;
  for (const g of goodTexts) {
    for (const b of badTexts) {
      pairs.push({
        prompt: item.prompt,
        chosen: g,
        rejected: b,
        rank_gap: 1
      });
      newPairs++;
    }
  }
  counterEl.textContent = pairs.length;
  currentIdx++;
  renderCurrent();
}

function skipPrompt() {
  currentIdx++;
  renderCurrent();
}

exportBtn.addEventListener('click', () => {
  const blob = new Blob([JSON.stringify(rankings, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'dpo_rankings.json';
  a.click();
  URL.revokeObjectURL(url);
});

// REMOVE the global skip listener — we use dynamic onclick only
// skipBtn.addEventListener('click', ...) was here and conflicted with onclick

function rankColor(rank, total) {
  const colors = ['#4a9', '#7a7', '#994', '#a74', '#a44'];
  if (total <= 5) return colors[rank - 1] || '#666';
  // interpolate for larger sets
  const t = (rank - 1) / (total - 1);
  if (t < 0.25) return '#4a9';
  if (t < 0.5)  return '#7a7';
  if (t < 0.75) return '#a74';
  return '#a44';
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
