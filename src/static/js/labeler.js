(() => {
  const sessionSelect = document.getElementById('sessionSelect');
  const startBtn = document.getElementById('startBtn');
  const detectBtn = document.getElementById('detectBtn');
  const regenManifestsBtn = document.getElementById('regenManifestsBtn');
  const statusEl = document.getElementById('status');
  const metaEl = document.getElementById('meta');
  const imgEl = document.getElementById('cellImg');
  const labelButtons = document.getElementById('labelButtons');
  const skipBtn = document.getElementById('skipBtn');
  const predictionHint = document.getElementById('predictionHint');
  const predictedLabel = document.getElementById('predictedLabel');
  const predictedConf = document.getElementById('predictedConf');

  // Training UI
  const trainStartBtn = document.getElementById('trainStartBtn');
  const trainStopBtn = document.getElementById('trainStopBtn');
  const trainReloadModelBtn = document.getElementById('trainReloadModelBtn');
  const trainRefreshBtn = document.getElementById('trainRefreshBtn');
  const trainStatusEl = document.getElementById('trainStatus');
  const trainLogsEl = document.getElementById('trainLogs');

  const trainEpochsEl = document.getElementById('trainEpochs');
  const trainBatchSizeEl = document.getElementById('trainBatchSize');
  const trainLrEl = document.getElementById('trainLr');
  const trainValFractionEl = document.getElementById('trainValFraction');
  const trainDeviceEl = document.getElementById('trainDevice');
  const trainNumWorkersEl = document.getElementById('trainNumWorkers');
  const trainImageSizeEl = document.getElementById('trainImageSize');
  const trainSeedEl = document.getElementById('trainSeed');
  const trainNoAugmentEl = document.getElementById('trainNoAugment');
  const trainAugRotationEl = document.getElementById('trainAugRotation');
  const trainAugTranslateXEl = document.getElementById('trainAugTranslateX');
  const trainAugTranslateYEl = document.getElementById('trainAugTranslateY');
  const trainAugScaleMinEl = document.getElementById('trainAugScaleMin');
  const trainAugScaleMaxEl = document.getElementById('trainAugScaleMax');
  const trainAugMultiplierEl = document.getElementById('trainAugMultiplier');

  let trainPollTimer = null;
  let lossChart = null;
  let accChart = null;

  const EPS_LOSS = 1e-4;
  const EPS_ACC = 1e-3;

  function ensureCharts() {
    if (lossChart && accChart) return;
    if (!window.Chart) {
      console.warn('Chart.js not available; training plots disabled.');
      return;
    }
    const lossCtx = document.getElementById('trainLossChart')?.getContext('2d');
    const accCtx = document.getElementById('trainAccChart')?.getContext('2d');
    if (!lossCtx || !accCtx) return;

    lossChart = new window.Chart(lossCtx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: 'train_loss',
          data: [],
          borderColor: '#e67e22',
          backgroundColor: 'rgba(230, 126, 34, 0.15)',
          tension: 0.15,
        }],
      },
      options: {
        animation: false,
        responsive: true,
        scales: {
          y: {
            type: 'logarithmic',
            min: EPS_LOSS,
            ticks: {
              callback: (v) => {
                const n = Number(v);
                if (!Number.isFinite(n)) return v;
                return n >= 1 ? String(n) : n.toPrecision(2);
              }
            }
          },
        },
      },
    });

    accChart = new window.Chart(accCtx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'train_acc',
            data: [],
            borderColor: '#2ecc71',
            backgroundColor: 'rgba(46, 204, 113, 0.15)',
            tension: 0.15,
          },
          {
            label: 'val_acc',
            data: [],
            borderColor: '#3498db',
            backgroundColor: 'rgba(52, 152, 219, 0.15)',
            tension: 0.15,
          },
        ],
      },
      options: {
        animation: false,
        responsive: true,
        scales: {
          y: {
            type: 'logarithmic',
            min: 0.5,
            max: 1,
            ticks: {
              callback: (v) => {
                const n = Number(v);
                if (!Number.isFinite(n)) return v;
                return n >= 1 ? '1.0' : n.toPrecision(2);
              }
            }
          },
        },
      },
    });
  }

  function setTrainStatus(text) {
    if (trainStatusEl) trainStatusEl.textContent = text;
  }

  function updateTrainUI(status) {
    ensureCharts();

    const running = !!status.running;
    const err = status.error;
    const exitCode = status.exitCode;

    if (running) {
      const last = (status.history && status.history.length) ? status.history[status.history.length - 1] : null;
      if (last && typeof last.epoch === 'number') {
        setTrainStatus(`Training… epoch ${last.epoch} (exit: running)`);
      } else {
        setTrainStatus('Training…');
      }
    } else if (err) {
      setTrainStatus(`Error: ${err}`);
    } else if (exitCode !== null && exitCode !== undefined) {
      setTrainStatus(`Finished (exit code ${exitCode})`);
    } else {
      setTrainStatus('Idle.');
    }

    if (trainLogsEl) {
      const logs = status.logs || [];
      if (logs.length > 0) {
        trainLogsEl.style.display = 'block';
        trainLogsEl.textContent = logs.slice(-200).join('\n');
        trainLogsEl.scrollTop = trainLogsEl.scrollHeight;
      } else {
        trainLogsEl.style.display = 'none';
        trainLogsEl.textContent = '';
      }
    }

    const history = status.history || [];
    if (lossChart) {
      lossChart.data.labels = history.map(h => String(h.epoch ?? ''));
      lossChart.data.datasets[0].data = history.map(h => {
        if (typeof h.train_loss !== 'number') return null;
        return Math.max(EPS_LOSS, h.train_loss);
      });
      lossChart.update();
    }
    if (accChart) {
      accChart.data.labels = history.map(h => String(h.epoch ?? ''));
      accChart.data.datasets[0].data = history.map(h => {
        if (typeof h.train_acc !== 'number') return null;
        return Math.max(EPS_ACC, Math.min(1, h.train_acc));
      });
      accChart.data.datasets[1].data = history.map(h => {
        if (typeof h.val_acc !== 'number') return null;
        return Math.max(EPS_ACC, Math.min(1, h.val_acc));
      });
      accChart.update();
    }
  }

  async function fetchTrainStatus() {
    const res = await fetch('/api/train/status');
    const data = await res.json();
    updateTrainUI(data);
    if (data.running) {
      if (!trainPollTimer) {
        trainPollTimer = setInterval(fetchTrainStatus, 1000);
      }
    } else {
      if (trainPollTimer) {
        clearInterval(trainPollTimer);
        trainPollTimer = null;
      }
    }
  }

  const LABELS = ['.', '?'].concat('ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split(''));
  let current = null;
  let session = null;
  let reviewQueue = null;  // Array of {session, file, predictedLabel, confidence} for mislabel review

  async function fetchSessions() {
    const res = await fetch('/api/label/sessions');
    const data = await res.json();
    sessionSelect.innerHTML = '';
    data.sessions.forEach((s) => {
      const opt = document.createElement('option');
      opt.value = s.id;
      opt.textContent = `${s.id} (${s.cells} cells)`;
      sessionSelect.appendChild(opt);
    });
    if (data.sessions.length === 0) {
      statusEl.textContent = 'No sessions with extract_debug cells found.';
    }
  }

  async function loadNext() {
    statusEl.textContent = 'Loading next cell...';
    
    // If in review mode, load from queue
    if (reviewQueue && reviewQueue.length > 0) {
      const item = reviewQueue.shift();
      session = item.session;  // Set session from the review item
      current = {
        file: item.file,
        imageUrl: `/files/label_store/images/${item.session}/${item.file}`,
        predictedLabel: item.predictedLabel,
        confidence: item.confidence,
        currentLabel: item.currentLabel,
      };
      console.log('Loading review item:', current);
      imgEl.src = current.imageUrl;
      imgEl.style.display = 'block';
      metaEl.textContent = `File: ${current.file} | Current label: ${item.currentLabel} | Reviewing: ${reviewQueue.length} remaining`;
      statusEl.textContent = 'Review this label (suggested by model).';
      
      // Show prediction hint
      predictionHint.style.display = 'block';
      predictedLabel.textContent = item.predictedLabel;
      predictedConf.textContent = (item.confidence * 100).toFixed(1) + '%';
      return;
    }
    
    // Normal mode - load next unlabeled
    if (!session) return;  // Only check session requirement for normal mode
    predictionHint.style.display = 'none';
    const res = await fetch('/api/label/next', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session }),
    });
    const data = await res.json();
    if (data.done) {
      statusEl.textContent = `Done. Labeled ${data.labeled}/${data.total}`;
      imgEl.src = '';
      metaEl.textContent = '';
      current = null;
      return;
    }
    current = data;
    imgEl.src = data.imageUrl;
    metaEl.textContent = `File: ${data.file} | Labeled: ${data.labeled} | Remaining: ${data.remaining}/${data.total}`;
    statusEl.textContent = 'Click a label to save.';
  }

  async function saveLabel(label) {
    if (!session || !current) return;
    statusEl.textContent = 'Saving...';
    
    // For review mode, we need to update the existing label in label_store
    const targetSession = session;
    
    const res = await fetch('/api/label/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session: targetSession,
        file: current.file,
        label,
      }),
    });
    const data = await res.json();
    if (!data.ok) {
      statusEl.textContent = data.error || 'Save failed';
      return;
    }
    loadNext();
  }

  LABELS.forEach((lab) => {
    const btn = document.createElement('button');
    btn.textContent = lab;
    btn.className = 'label-btn';
    btn.addEventListener('click', () => saveLabel(lab));
    labelButtons.appendChild(btn);
  });

  skipBtn.addEventListener('click', () => loadNext());
  startBtn.addEventListener('click', () => {
    session = sessionSelect.value;
    if (!session) {
      statusEl.textContent = 'Select a session first.';
      return;
    }
    reviewQueue = null;  // Clear review mode
    loadNext();
  });

  detectBtn.addEventListener('click', async () => {
    statusEl.textContent = 'Detecting mislabels...';
    const minConfThreshold = parseFloat(document.getElementById('minConfThreshold').value) || 0.5;
    const res = await fetch('/api/label/detect-mislabels', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ minConf: minConfThreshold, topN: 50 }),
    });
    if (!res.ok) {
      const err = await res.json();
      statusEl.textContent = `Error: ${err.error || 'Detection failed'}`;
      console.error('Detection error:', err);
      return;
    }
    const data = await res.json();
    console.log('Detection result:', data);
    if (data.total === 0) {
      statusEl.textContent = 'No mislabels detected! All labels look good.';
      return;
    }
    reviewQueue = data.cases.slice();  // Clone array
    console.log('Review queue:', reviewQueue);
    statusEl.textContent = `Found ${data.total} suspicious labels. Starting review...`;
    loadNext();
  });

  regenManifestsBtn?.addEventListener('click', async () => {
    statusEl.textContent = 'Cleaning up / archiving old manifests...';
    const res = await fetch('/api/label/regenerate-manifests', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ append: true }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      statusEl.textContent = `Error: ${data.error || 'Manifest regeneration failed'}`;
      return;
    }
    if (data.migrated) {
      statusEl.textContent = `Manifest migrated. Archived ${data.archived?.length || 0} file(s).`;
    } else {
      statusEl.textContent = `Cleanup done. Archived ${data.archived?.length || 0} file(s).`;
    }
    console.log('Regenerate manifests result:', data);
  });

  trainStartBtn?.addEventListener('click', async () => {
    setTrainStatus('Starting training…');
    const payload = {
      epochs: parseInt(trainEpochsEl?.value || '15', 10),
      batchSize: parseInt(trainBatchSizeEl?.value || '64', 10),
      lr: parseFloat(trainLrEl?.value || '0.001'),
      valFraction: parseFloat(trainValFractionEl?.value || '0.2'),
      device: (trainDeviceEl?.value || ''),
      numWorkers: parseInt(trainNumWorkersEl?.value || '0', 10),
      imageSize: parseInt(trainImageSizeEl?.value || '32', 10),
      seed: parseInt(trainSeedEl?.value || '1337', 10),
      noAugment: !!trainNoAugmentEl?.checked,
      augRotation: parseFloat(trainAugRotationEl?.value || '7'),
      augTranslateX: parseFloat(trainAugTranslateXEl?.value || '0.05'),
      augTranslateY: parseFloat(trainAugTranslateYEl?.value || '0.05'),
      augScaleMin: parseFloat(trainAugScaleMinEl?.value || '0.9'),
      augScaleMax: parseFloat(trainAugScaleMaxEl?.value || '1.1'),
      augMultiplier: parseInt(trainAugMultiplierEl?.value || '1', 10),
    };

    const res = await fetch('/api/train/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      setTrainStatus(`Error: ${data.error || 'Failed to start training'}`);
      return;
    }
    console.log('Training start:', data);
    // Force polling immediately (avoid any race where status briefly shows running=false).
    if (!trainPollTimer) {
      trainPollTimer = setInterval(fetchTrainStatus, 1000);
    }
    await fetchTrainStatus();
  });

  trainStopBtn?.addEventListener('click', async () => {
    setTrainStatus('Stopping…');
    const res = await fetch('/api/train/stop', { method: 'POST' });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      setTrainStatus(`Error: ${data.error || 'Stop failed'}`);
      return;
    }
    await fetchTrainStatus();
  });

  trainReloadModelBtn?.addEventListener('click', async () => {
    setTrainStatus('Reloading model…');
    const res = await fetch('/api/label/reload-model', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ device: 'cpu' }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      setTrainStatus(`Error: ${data.error || 'Reload failed'}`);
      return;
    }
    setTrainStatus(`Model reloaded (imageSize=${data.imageSize ?? '?'}, mtime=${data.mtime ?? '?'})`);
    console.log('Reload model:', data);
  });

  trainRefreshBtn?.addEventListener('click', async () => {
    await fetchTrainStatus();
  });

  fetchSessions();
  fetchTrainStatus();
})();
