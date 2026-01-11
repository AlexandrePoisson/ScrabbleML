let sessionId = null;
let sourceUrl = null;
let suggested = null;

const canvas = document.getElementById('src-canvas');
const ctx = canvas.getContext('2d');
const imgEl = new Image();
let corners = [];
let dragIndex = -1;
const R = 8;

function draw() {
  if (!imgEl.src) return;
  canvas.width = imgEl.naturalWidth;
  canvas.height = imgEl.naturalHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(imgEl, 0, 0);
  // Draw corners
  ctx.fillStyle = 'rgba(255,0,0,0.8)';
  ctx.strokeStyle = 'rgba(0,255,0,0.8)';
  ctx.lineWidth = 2;
  for (let i = 0; i < corners.length; i++) {
    const [x, y] = corners[i];
    ctx.beginPath();
    ctx.arc(x, y, R, 0, Math.PI * 2);
    ctx.fill();
  }
  // Draw polygon
  if (corners.length === 4) {
    ctx.beginPath();
    ctx.moveTo(corners[0][0], corners[0][1]);
    ctx.lineTo(corners[1][0], corners[1][1]);
    ctx.lineTo(corners[2][0], corners[2][1]);
    ctx.lineTo(corners[3][0], corners[3][1]);
    ctx.closePath();
    ctx.stroke();
  }
}

function setStatus(msg) {
  document.getElementById('status').textContent = msg || '';
}

async function uploadAndSuggest(file) {
  setStatus('Uploading...');
  const fd = new FormData();
  fd.append('image', file);
  const res = await fetch('/api/suggest', { method: 'POST', body: fd });
  if (!res.ok) {
    setStatus('Upload failed');
    return null;
  }
  return await res.json();
}

async function warpWithCurrentCorners() {
  if (!sessionId || corners.length !== 4) return null;
  setStatus('Warping...');
  const res = await fetch('/api/warp', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session: sessionId, corners })
  });
  if (!res.ok) {
    setStatus('Warp failed');
    return null;
  }
  return await res.json();
}

async function extractWithUiSettings() {
  if (!sessionId) return null;
  setStatus('Extracting board...');
  clearBoardAndPreview();
  stopPreviewSequence();

  const engineEl = document.getElementById('ocr-engine');
  const innerMarginEl = document.getElementById('inner-margin');
  const outerMarginEl = document.getElementById('outer-margin');
  const minConfEl = document.getElementById('min-conf');
  const psmEl = document.getElementById('psm');
  const binEl = document.getElementById('binarize');
  const isoEl = document.getElementById('isolate-main');
  const engine = engineEl.value;
  const innerMargin = parseFloat(innerMarginEl.value);
  const outerMargin = parseFloat(outerMarginEl.value);
  const minConf = parseFloat(minConfEl.value);
  const psm = parseInt(psmEl.value, 10);
  const binarize = !!binEl.checked;
  const isolateMain = !!isoEl.checked;

  const res = await fetch('/api/extract', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session: sessionId, engine, innerMargin, outerMargin, minConf, psm, psmFallback: 8, binarize, isolateMain })
  });
  if (!res.ok) {
    const contentType = (res.headers.get('content-type') || '').toLowerCase();
    if (contentType.includes('application/json')) {
      const data = await res.json().catch(() => ({}));
      setStatus(data.error || 'Extract failed');
      return null;
    }
    const text = await res.text().catch(() => '');
    setStatus(text || `Extract failed (HTTP ${res.status})`);
    return null;
  }
  return await res.json();
}

document.getElementById('upload-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const fileInput = document.getElementById('image-input');
  if (!fileInput.files.length) return;
  const data = await uploadAndSuggest(fileInput.files[0]);
  if (!data) return;
  sessionId = data.session;
  sourceUrl = data.sourceUrl;
  imgEl.onload = () => { corners = normalizeCorners(data.suggestedCorners); draw(); enableWarp(); };
  imgEl.src = sourceUrl;
  setStatus('Suggested corners loaded. Drag to adjust.');
});

function enableWarp() {
  document.getElementById('warp-btn').disabled = false;
  document.getElementById('extract-btn').disabled = true; // enable after warp
}

function normalizeCorners(suggest) {
  // Expect [[x,y], ...] in TL,TR,BR,BL; if not present, create a rectangle inset
  const w = imgEl.naturalWidth || canvas.width || 1000;
  const h = imgEl.naturalHeight || canvas.height || 1000;
  if (!suggest || suggest.length !== 4) {
    const insetX = w * 0.1, insetY = h * 0.1;
    return [
      [insetX, insetY],
      [w - insetX, insetY],
      [w - insetX, h - insetY],
      [insetX, h - insetY],
    ];
  }
  return suggest.map(p => [p[0], p[1]]);
}

canvas.addEventListener('mousedown', (e) => {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const x = (e.clientX - rect.left) * scaleX;
  const y = (e.clientY - rect.top) * scaleY;
  dragIndex = -1;
  for (let i = 0; i < corners.length; i++) {
    const [cx, cy] = corners[i];
    const d2 = (cx - x) ** 2 + (cy - y) ** 2;
    if (d2 <= (R * 2) ** 2) { dragIndex = i; break; }
  }
});

canvas.addEventListener('mousemove', (e) => {
  if (dragIndex < 0) return;
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  let x = (e.clientX - rect.left) * scaleX;
  let y = (e.clientY - rect.top) * scaleY;
  x = Math.max(0, Math.min(canvas.width - 1, x));
  y = Math.max(0, Math.min(canvas.height - 1, y));
  corners[dragIndex] = [x, y];
  draw();
});

window.addEventListener('mouseup', () => { dragIndex = -1; });

document.getElementById('warp-btn').addEventListener('click', async () => {
  if (!sessionId || corners.length !== 4) return;
  const data = await warpWithCurrentCorners();
  if (!data) return;
  document.getElementById('grid-img').src = data.gridUrl + `?t=${Date.now()}`;
  setStatus('Warp complete.');
  document.getElementById('extract-btn').disabled = false;
});

document.getElementById('extract-btn').addEventListener('click', async () => {
  const data = await extractWithUiSettings();
  if (!data) return;
  renderBoardTable(data.letters);
  playPreviewSequence(data.previews || [], data.lastPreview);
  setStatus('Extraction complete.');
});

function clearBoardAndPreview() {
  const tbl = document.getElementById('board-table');
  if (tbl) while (tbl.firstChild) tbl.removeChild(tbl.firstChild);
  const img = document.getElementById('preview-img');
  if (img) img.removeAttribute('src');
  const pc = document.getElementById('preview-char');
  if (pc) pc.textContent = '';
}

function renderBoardTable(letters) {
  const tbl = document.getElementById('board-table');
  if (!tbl) return;
  // Clear
  while (tbl.firstChild) tbl.removeChild(tbl.firstChild);
  // Header row
  const thead = document.createElement('thead');
  const hr = document.createElement('tr');
  const h0 = document.createElement('th');
  h0.textContent = '';
  hr.appendChild(h0);
  for (let c = 1; c <= 15; c++) {
    const th = document.createElement('th');
    th.textContent = String(c);
    hr.appendChild(th);
  }
  thead.appendChild(hr);
  tbl.appendChild(thead);
  // Body rows A..O
  const tbody = document.createElement('tbody');
  for (let r = 0; r < 15; r++) {
    const tr = document.createElement('tr');
    const rl = document.createElement('th');
    rl.textContent = String.fromCharCode('A'.charCodeAt(0) + r);
    tr.appendChild(rl);
    for (let c = 0; c < 15; c++) {
      const td = document.createElement('td');
      let ch = (letters && letters[r] && letters[r][c]) || '.';
      td.textContent = ch || '.';
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  tbl.appendChild(tbody);
}

function renderPreview(preview) {
  const img = document.getElementById('preview-img');
  const pc = document.getElementById('preview-char');
  if (!img || !pc) return;
  if (!preview || !preview.url) {
    img.removeAttribute('src');
    pc.textContent = '';
    return;
  }
  img.src = preview.url + `?t=${Date.now()}`;
  const rc = (typeof preview.row === 'number' && typeof preview.col === 'number') ? ` (${String.fromCharCode('A'.charCodeAt(0) + preview.row)}${preview.col + 1})` : '';
  pc.textContent = `${preview.char || '.'}${rc}`;
  img.alt = `Cell ${rc}`;
}

let previewTimer = null;

function stopPreviewSequence() {
  if (previewTimer) {
    clearTimeout(previewTimer);
    previewTimer = null;
  }
}

function playPreviewSequence(seq, fallbackPreview) {
  stopPreviewSequence();
  if (!seq || !seq.length) {
    renderPreview(fallbackPreview);
    return;
  }
  let idx = 0;
  const step = () => {
    renderPreview(seq[idx]);
    idx += 1;
    if (idx < seq.length) {
      previewTimer = setTimeout(step, 200);
    }
  };
  step();
}
