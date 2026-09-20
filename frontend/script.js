/* frontend/script.js - Password Security Agent, vanilla JS, no password persistence */

(() => {
  'use strict';

  // DOM refs
  const perspective = document.getElementById('perspective');
  const panel = document.getElementById('workspacePanel');
  const passwordInput = document.getElementById('passwordInput');
  const toggleBtn = document.getElementById('toggleVisibilityBtn');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const clearBtn = document.getElementById('clearBtn');
  const errorBanner = document.getElementById('errorBanner');
  const loadingState = document.getElementById('loadingState');
  const resultsEl = document.getElementById('results');
  const scoreValue = document.getElementById('scoreValue');
  const scoreProgress = document.getElementById('scoreProgress');
  const riskLevelEl = document.getElementById('riskLevel');
  const scoreExplain = document.getElementById('scoreExplain');
  const issuesList = document.getElementById('issuesList');
  const issuesEmpty = document.getElementById('issuesEmpty');
  const positiveList = document.getElementById('positiveList');
  const positiveEmpty = document.getElementById('positiveEmpty');
  const whySummary = document.getElementById('whySummary');
  const whyList = document.getElementById('whyList');
  const recBlock = document.getElementById('recommendationsBlock');
  const recList = document.getElementById('recommendationsList');
  const mlClass = document.getElementById('mlClass');
  const mlConfidence = document.getElementById('mlConfidence');
  const mlModel = document.getElementById('mlModel');
  const mlFeatureSet = document.getElementById('mlFeatureSet');
  const mlWarning = document.getElementById('mlWarning');
  const mlProbs = document.getElementById('mlProbs');
  const mlProbsDetails = document.getElementById('mlProbsDetails');
  const knowledgeList = document.getElementById('knowledgeList');
  const knowledgeEmpty = document.getElementById('knowledgeEmpty');
  const technicalInfo = document.getElementById('technicalInfo');
  const newAnalysisBtn = document.getElementById('newAnalysisBtn');
  const menuBtn = document.getElementById('menuBtn');
  const sidebar = document.querySelector('.sidebar');
  const overlay = document.getElementById('overlay');

  // Navigation & new panels
  const navItems = document.querySelectorAll('.nav-item[data-page]');
  const knowledgePanel = document.getElementById('knowledgePanel');
  const historyPanel = document.getElementById('historyPanel');
  const settingsPanel = document.getElementById('settingsPanel');
  const inputCard = document.querySelector('.input-card');
  const generatorCard = document.getElementById('generatorCard');
  // History refs
  const historyList = document.getElementById('historyList');
  const historyEmpty = document.getElementById('historyEmpty');
  const historyCount = document.getElementById('historyCount');
  const clearHistoryBtn = document.getElementById('clearHistoryBtn');
  // Settings refs
  const settingReducedMotion = document.getElementById('settingReducedMotion');
  const settingAutoAnalyze = document.getElementById('settingAutoAnalyze');
  const settingGenLength = document.getElementById('settingGenLength');
  const settingGenLengthValue = document.getElementById('settingGenLengthValue');
  const settingLower = document.getElementById('settingLower');
  const settingUpper = document.getElementById('settingUpper');
  const settingDigits = document.getElementById('settingDigits');
  const settingSymbols = document.getElementById('settingSymbols');

  // In-memory history (safe metadata only, never plaintext password)
  const analysisHistory = [];
  // Settings state (in-memory only, no password storage)
  const appSettings = {
    reducedMotion: window.matchMedia('(prefers-reduced-motion: reduce)').matches,
    autoAnalyzeGenerated: true,
    genLength: 16,
    genTypes: { lower: true, upper: true, digits: true, symbols: true },
  };

  // Generator refs
  const genLength = document.getElementById('genLength');
  const genLengthValue = document.getElementById('genLengthValue');
  const genLower = document.getElementById('genLower');
  const genUpper = document.getElementById('genUpper');
  const genDigits = document.getElementById('genDigits');
  const genSymbols = document.getElementById('genSymbols');
  const generateBtn = document.getElementById('generateBtn');
  const generatePassphraseBtn = document.getElementById('generatePassphraseBtn');
  const generatedOutput = document.getElementById('generatedOutput');
  const generatedPassword = document.getElementById('generatedPassword');
  const copyGeneratedBtn = document.getElementById('copyGeneratedBtn');
  const useGeneratedBtn = document.getElementById('useGeneratedBtn');
  const copyHint = document.getElementById('copyHint');

  // New composition/entropy/attack refs
  const compLength = document.getElementById('compLength');
  const compUnique = document.getElementById('compUnique');
  const compLower = document.getElementById('compLower');
  const compUpper = document.getElementById('compUpper');
  const compDigits = document.getElementById('compDigits');
  const compSpecial = document.getElementById('compSpecial');
  const entropyCharset = document.getElementById('entropyCharset');
  const entropyEstimated = document.getElementById('entropyEstimated');
  const entropyShannon = document.getElementById('entropyShannon');
  const bruteForce = document.getElementById('bruteForce');
  const attackList = document.getElementById('attackList');
  const attackSearchSpace = document.getElementById('attackSearchSpace');

  let isAnalyzing = false;
  let rafId = null;
  let currentTilt = { x: 0, y: 0 };
  let targetTilt = { x: 0, y: 0 };

  const API_BASE = 'http://127.0.0.1:8000';
  const ANALYZE_URL = `${API_BASE}/analyze`;
  const GENERATE_URL = `${API_BASE}/generate`;
  const CIRCUMFERENCE = 2 * Math.PI * 54;

  // ---------- Utilities ----------
  function escapeHtml(str) {
    return String(str).replace(/[&<>"']/g, (c) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
  }

  function setLoading(loading) {
    isAnalyzing = loading;
    analyzeBtn.disabled = loading;
    const label = analyzeBtn.querySelector('.btn-label');
    const spinner = analyzeBtn.querySelector('.btn-spinner');
    if (loading) {
      if (label) label.textContent = 'Analyzing…';
      if (spinner) spinner.style.display = 'block';
      loadingState.hidden = false;
      errorBanner.hidden = true;
      errorBanner.textContent = '';
    } else {
      if (label) label.textContent = 'Analyze Password';
      if (spinner) spinner.style.display = 'none';
      loadingState.hidden = true;
    }
  }

  function showError(message) {
    errorBanner.textContent = message;
    errorBanner.hidden = false;
  }

  function clearError() {
    errorBanner.hidden = true;
    errorBanner.textContent = '';
  }

  function updateAnalyzeButton() {
    analyzeBtn.disabled = isAnalyzing;
  }

  // ---------- 3D Tilt ----------
  function initializeTilt() {
    const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const isTouch = window.matchMedia('(pointer: coarse)').matches || 'ontouchstart' in window;
    if (prefersReduced || isTouch) return;
    if (!perspective || !panel) return;
    const maxDeg = 5;
    function handleTilt(e) {
      const rect = perspective.getBoundingClientRect();
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height / 2;
      const dx = (e.clientX - cx) / (rect.width / 2);
      const dy = (e.clientY - cy) / (rect.height / 2);
      targetTilt.x = Math.max(-1, Math.min(1, dy)) * maxDeg * -1;
      targetTilt.y = Math.max(-1, Math.min(1, dx)) * maxDeg;
      if (!rafId) rafId = requestAnimationFrame(applyTilt);
    }
    function applyTilt() {
      currentTilt.x += (targetTilt.x - currentTilt.x) * 0.12;
      currentTilt.y += (targetTilt.y - currentTilt.y) * 0.12;
      panel.classList.add('tilt');
      panel.style.transform = `rotateX(${currentTilt.x.toFixed(2)}deg) rotateY(${currentTilt.y.toFixed(2)}deg) translateZ(0)`;
      const diff = Math.abs(currentTilt.x - targetTilt.x) + Math.abs(currentTilt.y - targetTilt.y);
      if (diff > 0.05) {
        rafId = requestAnimationFrame(applyTilt);
      } else {
        rafId = null;
      }
    }
    function resetTilt() {
      targetTilt = { x: 0, y: 0 };
      if (!rafId) rafId = requestAnimationFrame(applyTilt);
      setTimeout(() => {
        panel.classList.remove('tilt');
        panel.style.transform = '';
        currentTilt = { x: 0, y: 0 };
        if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
      }, 300);
    }
    perspective.addEventListener('mousemove', handleTilt);
    perspective.addEventListener('mouseleave', resetTilt);
  }

  // ---------- Visibility ----------
  function toggleVisibility() {
    const isPassword = passwordInput.type === 'password';
    passwordInput.type = isPassword ? 'text' : 'password';
    toggleBtn.setAttribute('aria-pressed', String(isPassword));
    toggleBtn.setAttribute('aria-label', isPassword ? 'Hide password' : 'Show password');
    const eye = toggleBtn.querySelector('.icon-eye');
    const eyeOff = toggleBtn.querySelector('.icon-eye-off');
    if (eye && eyeOff) {
      eye.style.display = isPassword ? 'none' : 'block';
      eyeOff.style.display = isPassword ? 'block' : 'none';
    }
    passwordInput.focus();
  }

  // ---------- Rendering ----------
  function renderList(ul, emptyEl, items, emptyText) {
    if (!ul) return;
    ul.innerHTML = '';
    if (!items || items.length === 0) {
      if (emptyEl) {
        emptyEl.hidden = false;
        if (emptyText) emptyEl.textContent = emptyText;
      }
      return;
    }
    if (emptyEl) emptyEl.hidden = true;
    for (const it of items) {
      const li = document.createElement('li');
      const text = typeof it === 'string' ? it : (it.message || it.code || '');
      li.textContent = text;
      ul.appendChild(li);
    }
  }

  function animateScore(score) {
    const pct = Math.max(0, Math.min(100, score)) / 100;
    const offset = CIRCUMFERENCE * (1 - pct);
    if (scoreProgress) {
      scoreProgress.style.transition = 'stroke-dashoffset 900ms ease';
      void scoreProgress.offsetWidth;
      scoreProgress.style.strokeDashoffset = String(offset);
    }
    if (scoreValue) {
      let current = 0;
      const target = score;
      const duration = 700;
      const start = performance.now();
      function step(now) {
        const t = Math.min(1, (now - start) / duration);
        const eased = 1 - Math.pow(1 - t, 3);
        current = Math.round(eased * target);
        scoreValue.textContent = String(current);
        if (t < 1) requestAnimationFrame(step);
      }
      requestAnimationFrame(step);
    }
  }

  function renderResults(data) {
    const risk = data.risk || {};
    const patterns = data.patterns || {};
    const ml = data.ml || {};
    const rag = data.rag || {};
    const explanation = data.explanation || {};
    const features = data.features || {};

    const score = typeof risk.heuristic_score === 'number' ? risk.heuristic_score : 0;
    const level = risk.risk_level || 'Unknown';
    if (riskLevelEl) riskLevelEl.textContent = level.toUpperCase();
    if (scoreExplain) scoreExplain.textContent = risk.note || 'Heuristic Security Risk Score — engineering assessment, not probability.';
    animateScore(score);

    // Composition
    if (compLength) compLength.textContent = String(features.length ?? '—');
    if (compUnique) compUnique.textContent = String(features.unique_character_count ?? '—');
    if (compLower) compLower.textContent = String(features.lowercase_count ?? '—');
    if (compUpper) compUpper.textContent = String(features.uppercase_count ?? '—');
    if (compDigits) compDigits.textContent = String(features.digit_count ?? '—');
    if (compSpecial) compSpecial.textContent = String(features.special_count ?? '—');

    // Entropy & brute-force
    if (entropyCharset) entropyCharset.textContent = String(features.charset_size ?? '—');
    if (entropyEstimated) entropyEstimated.textContent = features.estimated_entropy != null ? `${features.estimated_entropy} bits` : '—';
    if (entropyShannon) entropyShannon.textContent = features.shannon_entropy != null ? `${features.shannon_entropy} bits` : '—';
    if (bruteForce) bruteForce.textContent = features.brute_force_search_space || (features.brute_force_combinations_log2 ? `2^${features.brute_force_combinations_log2}` : '—');

    // Entropy note: distinguish entropy vs guessability
    const entropyNote = document.getElementById('entropyNote');
    if (entropyNote && risk.entropy_analysis) {
      entropyNote.textContent = risk.entropy_analysis.note || entropyNote.textContent;
    }

    const issues = risk.issues || patterns.issues || [];
    const warnings = risk.warnings || patterns.warnings || [];
    const positives = risk.positive_signals || patterns.positive_signals || [];

    renderList(issuesList, issuesEmpty, issues.map(i => i.message || i.code), 'No major structural issues detected.');
    if (issues.length === 0 && warnings.length > 0) {
      renderList(issuesList, issuesEmpty, warnings.map(w => w.message || w.code));
      if (issuesEmpty) issuesEmpty.hidden = true;
    }
    renderList(positiveList, positiveEmpty, positives.map(p => p.message || p.code), 'No strong positive signals.');

    // Why
    if (whySummary) whySummary.textContent = explanation.summary || (risk.explanations && risk.explanations[0]) || '';
    const whyItems = explanation.why && explanation.why.length ? explanation.why : (risk.explanations || []);
    renderList(whyList, null, whyItems);

    const recs = explanation.recommendations || risk.recommendations || [];
    if (recs.length) {
      recBlock.hidden = false;
      renderList(recList, null, recs);
    } else {
      recBlock.hidden = true;
    }

    // ML
    if (mlClass) mlClass.textContent = ml.available ? `Class ${ml.predicted_class}` : 'Unavailable';
    if (mlConfidence) mlConfidence.textContent = ml.available && typeof ml.confidence === 'number' ? `${(ml.confidence * 100).toFixed(0)}%` : '—';
    if (mlModel) mlModel.textContent = ml.model_name || '—';
    if (mlFeatureSet) mlFeatureSet.textContent = ml.feature_set || '—';
    if (mlWarning) mlWarning.textContent = ml.warning || risk.ml_evidence?.note || 'ML evidence reflects project dataset, heavily length-driven.';
    if (mlProbs && mlProbsDetails) {
      if (ml.class_probabilities) {
        const probs = ml.class_probabilities;
        mlProbs.textContent = `0: ${(probs['0']*100).toFixed(1)}%  1: ${(probs['1']*100).toFixed(1)}%  2: ${(probs['2']*100).toFixed(1)}%`;
        mlProbsDetails.hidden = false;
      } else {
        mlProbsDetails.hidden = true;
      }
    }

    // RAG
    const ragResults = rag.results || [];
    if (ragResults.length) {
      knowledgeList.innerHTML = '';
      knowledgeEmpty.hidden = true;
      for (const r of ragResults) {
        const li = document.createElement('li');
        li.innerHTML = `<strong>${escapeHtml(r.source || 'knowledge')}</strong>: ${escapeHtml((r.content || '').slice(0, 160))}`;
        knowledgeList.appendChild(li);
      }
    } else {
      knowledgeList.innerHTML = '';
      knowledgeEmpty.hidden = false;
    }

    // Attack analysis
    const attackAnalysis = risk.attack_analysis || [];
    if (attackList) {
      attackList.innerHTML = '';
      if (attackAnalysis.length) {
        for (const a of attackAnalysis) {
          const li = document.createElement('li');
          li.textContent = `${a.attack} (${a.risk}): ${a.explanation}`;
          attackList.appendChild(li);
        }
      } else {
        const li = document.createElement('li');
        li.textContent = 'No specific attack pattern identified.';
        attackList.appendChild(li);
      }
    }
    if (attackSearchSpace) {
      const bf = risk.brute_force_analysis || {};
      attackSearchSpace.textContent = bf.search_space_display || bf.search_space_log2 || '—';
    }

    if (technicalInfo) {
      const info = `Score ${score}/100 • ${level} • ML ${ml.available ? ml.predicted_class + ' (' + (ml.confidence*100).toFixed(0) + '%)' : 'unavailable'} • RAG ${ragResults.length} sources • Provider ${explanation.provider || 'fallback'}`;
      technicalInfo.textContent = info;
    }

    resultsEl.hidden = false;
    resultsEl.setAttribute('aria-live', 'polite');
    resultsEl.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Add to in-memory history (safe metadata only, never password)
    try {
      addToHistory(data);
    } catch (_) {}
  }

  function clearResults() {
    if (resultsEl) resultsEl.hidden = true;
    if (scoreValue) scoreValue.textContent = '--';
    if (scoreProgress) scoreProgress.style.strokeDashoffset = String(CIRCUMFERENCE);
    if (riskLevelEl) riskLevelEl.textContent = '—';
    clearError();
  }

  function clearAll() {
    if (passwordInput) {
      passwordInput.value = '';
      passwordInput.type = 'password';
      toggleBtn.setAttribute('aria-pressed', 'false');
      toggleBtn.setAttribute('aria-label', 'Show password');
      const eye = toggleBtn.querySelector('.icon-eye');
      const eyeOff = toggleBtn.querySelector('.icon-eye-off');
      if (eye && eyeOff) { eye.style.display = 'block'; eyeOff.style.display = 'none'; }
    }
    clearResults();
    // Do not clear generated password on clear? Keep but hide hint
    if (copyHint) copyHint.textContent = '';
    updateAnalyzeButton();
    if (passwordInput) passwordInput.focus();
  }

  // ---------- API ----------
  async function analyzePassword() {
    if (isAnalyzing) return;
    const password = passwordInput.value; // may be empty, allowed
    clearError();
    setLoading(true);
    try {
      const resp = await fetch(ANALYZE_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password }),
      });
      if (!resp.ok) {
        if (resp.status === 422) {
          let data = null;
          try { data = await resp.json(); } catch (_) {}
          const msg = (data && data.detail) ? 'Validation error: please check your input.' : 'Invalid request.';
          throw new Error(msg);
        }
        if (resp.status === 500) {
          throw new Error('The security engine encountered an error. Please try again.');
        }
        throw new Error('Unable to connect to the security engine. Make sure the FastAPI backend is running on 127.0.0.1:8000.');
      }
      const data = await resp.json();
      if (!data || !data.risk || typeof data.risk.heuristic_score !== 'number') {
        throw new Error('Unexpected response from security engine.');
      }
      renderResults(data);
    } catch (err) {
      let message = 'Unable to connect to the security engine. Make sure the FastAPI backend is running on 127.0.0.1:8000.';
      if (err && err.message) {
        if (err.message.includes('Validation')) message = err.message;
        else if (err.message.includes('security engine encountered')) message = err.message;
        else if (err.message.includes('Unexpected')) message = err.message;
      }
      showError(message);
    } finally {
      setLoading(false);
      updateAnalyzeButton();
    }
  }

  // ---------- Generator ----------
  async function generatePassword(isPassphrase = false) {
    const length = parseInt(genLength ? genLength.value : '16', 10);
    const opts = {
      length: isPassphrase ? 16 : length,
      use_lower: genLower ? genLower.checked : true,
      use_upper: genUpper ? genUpper.checked : true,
      use_digits: genDigits ? genDigits.checked : true,
      use_symbols: genSymbols ? genSymbols.checked : true,
    };
    // For passphrase, use different endpoint? Use same but with length and all types
    // For now, use /generate with options
    try {
      if (copyHint) copyHint.textContent = 'Generating…';
      const resp = await fetch(GENERATE_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(opts),
      });
      if (!resp.ok) throw new Error('Generation failed');
      const data = await resp.json();
      const pw = data.password || data.passphrase || '';
      if (!pw) throw new Error('No password returned');
      if (generatedPassword) {
        generatedPassword.value = pw;
        generatedOutput.hidden = false;
        if (copyHint) copyHint.textContent = 'Generated securely. Click copy or use for analysis.';
      }
      // Analyze generated password if auto-analyze is enabled
      if (appSettings.autoAnalyzeGenerated) {
        if (data.analysis && data.analysis.risk) {
          renderResults(data.analysis);
        } else {
          const aResp = await fetch(ANALYZE_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ password: pw }),
          });
          if (aResp.ok) {
            const aData = await aResp.json();
            renderResults(aData);
          }
        }
      } else {
        if (copyHint) copyHint.textContent = 'Generated (auto-analyze disabled in Settings). Click Use to analyze.';
      }
    } catch (err) {
      if (copyHint) copyHint.textContent = 'Generation failed. Please try again.';
      showError('Password generation failed. Please try again.');
    }
  }

  async function copyGenerated() {
    if (!generatedPassword || !generatedPassword.value) return;
    try {
      await navigator.clipboard.writeText(generatedPassword.value);
      if (copyHint) {
        copyHint.textContent = 'Copied to clipboard.';
        setTimeout(() => { if (copyHint.textContent === 'Copied to clipboard.') copyHint.textContent = ''; }, 2000);
      }
    } catch (_) {
      // Fallback: select and execCommand
      generatedPassword.select();
      try { document.execCommand('copy'); if (copyHint) copyHint.textContent = 'Copied.'; } catch (_) {}
    }
  }

  function useGenerated() {
    if (!generatedPassword || !generatedPassword.value) return;
    passwordInput.value = generatedPassword.value;
    // Do not keep generated in visible history beyond input
    passwordInput.focus();
    analyzePassword();
  }

  // ---------- Mobile nav ----------
  function toggleSidebar(open) {
    const isOpen = typeof open === 'boolean' ? open : !sidebar.classList.contains('open');
    if (sidebar) sidebar.classList.toggle('open', isOpen);
    if (overlay) overlay.hidden = !isOpen;
    if (menuBtn) menuBtn.setAttribute('aria-expanded', String(isOpen));
    document.body.style.overflow = isOpen ? 'hidden' : '';
  }

  // ---------- SPA Navigation ----------
  function showPage(page) {
    const isDashboard = page === 'dashboard';
    // Hide/show dashboard elements
    if (inputCard) inputCard.hidden = !isDashboard;
    if (generatorCard) generatorCard.hidden = !isDashboard;
    if (loadingState) loadingState.hidden = isDashboard ? loadingState.hidden : true; // keep loading hidden when leaving dashboard
    if (resultsEl) resultsEl.hidden = isDashboard ? resultsEl.hidden : true;
    const rawDetails = document.getElementById('rawDetails');
    if (rawDetails) rawDetails.hidden = !isDashboard ? true : rawDetails.hidden;
    // Toggle knowledge/history/settings panels
    if (knowledgePanel) knowledgePanel.hidden = page !== 'knowledge';
    if (historyPanel) historyPanel.hidden = page !== 'history';
    if (settingsPanel) settingsPanel.hidden = page !== 'settings';
    // Update nav active states
    navItems.forEach(item => {
      const isActive = item.dataset.page === page;
      item.classList.toggle('active', isActive);
      if (isActive) {
        item.setAttribute('aria-current', 'page');
      } else {
        item.removeAttribute('aria-current');
      }
    });
    // Close mobile drawer after navigation
    if (sidebar && sidebar.classList.contains('open')) toggleSidebar(false);
    // Focus management
    if (isDashboard && passwordInput) {
      passwordInput.focus();
    }
  }

  function handleNavClick(e) {
    e.preventDefault();
    const target = e.currentTarget;
    const page = target.dataset.page;
    if (page) showPage(page);
  }

  // ---------- History (in-memory, safe metadata only) ----------
  function addToHistory(data) {
    const risk = data.risk || {};
    const features = data.features || {};
    const patterns = data.patterns || {};
    const entry = {
      timestamp: new Date().toLocaleString(),
      score: risk.heuristic_score ?? 0,
      level: risk.risk_level || 'Unknown',
      length: features.length ?? 0,
      issues: (risk.issues || []).map(i => i.code).slice(0, 3).join(', ') || (risk.warnings || []).map(w => w.code).slice(0, 2).join(', ') || 'none',
      entropy: features.estimated_entropy != null ? `${features.estimated_entropy} bits` : '—',
      searchSpace: features.brute_force_search_space || `2^${features.brute_force_combinations_log2 || 0}`,
    };
    analysisHistory.unshift(entry);
    // Keep only last 20
    if (analysisHistory.length > 20) analysisHistory.pop();
    renderHistory();
  }

  function renderHistory() {
    if (!historyList || !historyEmpty || !historyCount) return;
    historyList.innerHTML = '';
    historyCount.textContent = `${analysisHistory.length} ${analysisHistory.length === 1 ? 'analysis' : 'analyses'}`;
    if (analysisHistory.length === 0) {
      historyEmpty.hidden = false;
      historyList.hidden = true;
      return;
    }
    historyEmpty.hidden = true;
    historyList.hidden = false;
    analysisHistory.forEach((entry, idx) => {
      const li = document.createElement('li');
      li.className = 'history-item';
      li.tabIndex = 0;
      li.setAttribute('role', 'button');
      li.setAttribute('aria-label', `View analysis from ${entry.timestamp}, score ${entry.score}`);
      li.innerHTML = `
        <div class="history-meta">
          <span class="history-time">${escapeHtml(entry.timestamp)}</span>
          <span class="history-score">${escapeHtml(entry.level)} • ${entry.score}/100 • len ${entry.length}</span>
          <span class="history-issues">${escapeHtml(entry.issues)}</span>
        </div>
        <span class="mono-sm">${escapeHtml(entry.entropy)}</span>
      `;
      li.addEventListener('click', () => viewHistoryEntry(idx));
      li.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          viewHistoryEntry(idx);
        }
      });
      historyList.appendChild(li);
    });
  }

  function viewHistoryEntry(idx) {
    const entry = analysisHistory[idx];
    if (!entry) return;
    // Show dashboard and highlight that this is historical data
    showPage('dashboard');
    // Optionally show a toast or update the results area with historical metadata
    // For now, show an alert-like banner via errorBanner (safe, no password)
    showError(`Viewing historical analysis from ${entry.timestamp}: ${entry.level} ${entry.score}/100, length ${entry.length}. This is metadata only — password not stored.`);
    setTimeout(clearError, 4000);
  }

  function clearHistory() {
    analysisHistory.length = 0;
    renderHistory();
  }

  // ---------- Settings (functional) ----------
  function applySettings() {
    // Reduced motion
    const reduce = appSettings.reducedMotion;
    if (reduce) {
      document.documentElement.style.setProperty('--reduce-motion', '1');
      // Disable tilt by removing listeners? Simply add class that CSS respects
      document.body.classList.add('reduce-motion');
      if (panel) panel.style.transform = '';
    } else {
      document.documentElement.style.removeProperty('--reduce-motion');
      document.body.classList.remove('reduce-motion');
    }
    // Generator defaults
    if (genLength) genLength.value = String(appSettings.genLength);
    if (genLengthValue) genLengthValue.textContent = String(appSettings.genLength);
    if (settingGenLength) settingGenLength.value = String(appSettings.genLength);
    if (settingGenLengthValue) settingGenLengthValue.textContent = String(appSettings.genLength);
    if (genLower) genLower.checked = appSettings.genTypes.lower;
    if (genUpper) genUpper.checked = appSettings.genTypes.upper;
    if (genDigits) genDigits.checked = appSettings.genTypes.digits;
    if (genSymbols) genSymbols.checked = appSettings.genTypes.symbols;
    if (settingLower) settingLower.checked = appSettings.genTypes.lower;
    if (settingUpper) settingUpper.checked = appSettings.genTypes.upper;
    if (settingDigits) settingDigits.checked = appSettings.genTypes.digits;
    if (settingSymbols) settingSymbols.checked = appSettings.genTypes.symbols;
    if (settingReducedMotion) settingReducedMotion.checked = appSettings.reducedMotion;
    if (settingAutoAnalyze) settingAutoAnalyze.checked = appSettings.autoAnalyzeGenerated;
  }

  function handleSettingChange() {
    // Sync from settings panel to appSettings and generator
    if (settingReducedMotion) appSettings.reducedMotion = settingReducedMotion.checked;
    if (settingAutoAnalyze) appSettings.autoAnalyzeGenerated = settingAutoAnalyze.checked;
    if (settingGenLength) {
      const v = parseInt(settingGenLength.value, 10);
      if (!isNaN(v) && v >= 8 && v <= 32) {
        appSettings.genLength = v;
        if (settingGenLengthValue) settingGenLengthValue.textContent = String(v);
        if (genLength) genLength.value = String(v);
        if (genLengthValue) genLengthValue.textContent = String(v);
      }
    }
    if (settingLower) appSettings.genTypes.lower = settingLower.checked;
    if (settingUpper) appSettings.genTypes.upper = settingUpper.checked;
    if (settingDigits) appSettings.genTypes.digits = settingDigits.checked;
    if (settingSymbols) appSettings.genTypes.symbols = settingSymbols.checked;
    // Sync to generator checkboxes
    if (genLower) genLower.checked = appSettings.genTypes.lower;
    if (genUpper) genUpper.checked = appSettings.genTypes.upper;
    if (genDigits) genDigits.checked = appSettings.genTypes.digits;
    if (genSymbols) genSymbols.checked = appSettings.genTypes.symbols;
    applySettings();
  }

  // ---------- Init ----------
  function init() {
    initializeTilt();
    updateAnalyzeButton();
    applySettings();
    renderHistory();
    showPage('dashboard');

    if (analyzeBtn) analyzeBtn.addEventListener('click', analyzePassword);
    if (clearBtn) clearBtn.addEventListener('click', clearAll);
    if (toggleBtn) toggleBtn.addEventListener('click', toggleVisibility);
    if (passwordInput) {
      passwordInput.addEventListener('input', updateAnalyzeButton);
      passwordInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          analyzePassword();
        }
      });
    }
    if (newAnalysisBtn) newAnalysisBtn.addEventListener('click', () => {
      clearAll();
      showPage('dashboard');
      if (sidebar && sidebar.classList.contains('open')) toggleSidebar(false);
    });
    if (menuBtn) menuBtn.addEventListener('click', () => toggleSidebar());
    if (overlay) overlay.addEventListener('click', () => toggleSidebar(false));
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && sidebar && sidebar.classList.contains('open')) toggleSidebar(false);
    });

    // Navigation
    navItems.forEach(item => {
      item.addEventListener('click', handleNavClick);
      item.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          handleNavClick(e);
        }
      });
    });
    document.querySelectorAll('[data-back="dashboard"]').forEach(btn => {
      btn.addEventListener('click', () => showPage('dashboard'));
    });
    if (clearHistoryBtn) clearHistoryBtn.addEventListener('click', clearHistory);

    // Settings
    if (settingReducedMotion) settingReducedMotion.addEventListener('change', handleSettingChange);
    if (settingAutoAnalyze) settingAutoAnalyze.addEventListener('change', handleSettingChange);
    if (settingGenLength) settingGenLength.addEventListener('input', handleSettingChange);
    if (settingLower) settingLower.addEventListener('change', handleSettingChange);
    if (settingUpper) settingUpper.addEventListener('change', handleSettingChange);
    if (settingDigits) settingDigits.addEventListener('change', handleSettingChange);
    if (settingSymbols) settingSymbols.addEventListener('change', handleSettingChange);
    // Sync generator controls to settings
    if (genLength) genLength.addEventListener('input', () => {
      const v = parseInt(genLength.value, 10);
      if (!isNaN(v)) {
        appSettings.genLength = v;
        if (settingGenLength) settingGenLength.value = String(v);
        if (settingGenLengthValue) settingGenLengthValue.textContent = String(v);
        if (genLengthValue) genLengthValue.textContent = String(v);
      }
    });
    [genLower, genUpper, genDigits, genSymbols].forEach(el => {
      if (el) el.addEventListener('change', () => {
        appSettings.genTypes.lower = genLower ? genLower.checked : true;
        appSettings.genTypes.upper = genUpper ? genUpper.checked : true;
        appSettings.genTypes.digits = genDigits ? genDigits.checked : true;
        appSettings.genTypes.symbols = genSymbols ? genSymbols.checked : true;
        if (settingLower) settingLower.checked = appSettings.genTypes.lower;
        if (settingUpper) settingUpper.checked = appSettings.genTypes.upper;
        if (settingDigits) settingDigits.checked = appSettings.genTypes.digits;
        if (settingSymbols) settingSymbols.checked = appSettings.genTypes.symbols;
      });
    });

    // Generator
    if (genLength && genLengthValue) {
      genLength.addEventListener('input', () => { genLengthValue.textContent = genLength.value; });
      genLengthValue.textContent = genLength.value;
    }
    if (generateBtn) generateBtn.addEventListener('click', () => generatePassword(false));
    if (generatePassphraseBtn) generatePassphraseBtn.addEventListener('click', () => generatePassword(true));
    if (copyGeneratedBtn) copyGeneratedBtn.addEventListener('click', copyGenerated);
    if (useGeneratedBtn) useGeneratedBtn.addEventListener('click', useGenerated);

    if (scoreProgress) {
      scoreProgress.style.strokeDasharray = String(CIRCUMFERENCE);
      scoreProgress.style.strokeDashoffset = String(CIRCUMFERENCE);
    }
    if (passwordInput) passwordInput.focus();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
