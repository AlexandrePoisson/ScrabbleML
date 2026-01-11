(() => {
  const premiums = window.SCRABBLE_PREMIUMS;
  const letterScores = window.SCRABBLE_LETTER_SCORES || {};

  const langEl = document.getElementById('lang');
  const dictKeyEl = document.getElementById('dictKey');
  const rackEl = document.getElementById('rack');
  const randomRackBtn = document.getElementById('randomRackBtn');
  const findBestBtn = document.getElementById('findBestBtn');
  const placeBestBtn = document.getElementById('placeBestBtn');
  const moveStatus = document.getElementById('moveStatus');
  const moveDetails = document.getElementById('moveDetails');
  const boardStringEl = document.getElementById('boardString');
  const playBoard = document.getElementById('playBoard');
  const clearBoardBtn = document.getElementById('clearBoardBtn');
  const loadFromTextBtn = document.getElementById('loadFromTextBtn');
  const loadFromPictureBtn = document.getElementById('loadFromPictureBtn');

  // Picture loader elements (optional; only present on play page)
  const pictureLoaderEl = document.getElementById('pictureLoader');
  const picturePickBtn = document.getElementById('picturePickBtn');
  const pictureInput = document.getElementById('pictureInput');
  const pictureStatus = document.getElementById('pictureStatus');
  const pictureCloseBtn = document.getElementById('pictureCloseBtn');
  const pictureCanvas = document.getElementById('pictureCanvas');
  const pictureWarpBtn = document.getElementById('pictureWarpBtn');
  const pictureExtractBtn = document.getElementById('pictureExtractBtn');
  const pictureWarpPreview = document.getElementById('pictureWarpPreview');

  let lastMove = null; // {word,row,col,dir,score,placed,boardAfterString}

  // ---- Picture loader state ----
  let pictureSessionId = null;
  const pictureImg = new Image();
  let pictureCorners = [];
  let pictureDragIndex = -1;
  const pictureCornerRadius = 8;

  function setPictureStatus(text, isError = false) {
    if (!pictureStatus) return;
    pictureStatus.textContent = text || '';
    pictureStatus.style.color = isError ? '#b00020' : '#666';
  }

  function showPictureLoader(show) {
    if (!pictureLoaderEl) return;
    pictureLoaderEl.style.display = show ? 'block' : 'none';
  }

  function normalizeCornersForImage(img, suggest) {
    const w = img?.naturalWidth || 1000;
    const h = img?.naturalHeight || 1000;
    if (!suggest || !Array.isArray(suggest) || suggest.length !== 4) {
      const insetX = w * 0.1;
      const insetY = h * 0.1;
      return [
        [insetX, insetY],
        [w - insetX, insetY],
        [w - insetX, h - insetY],
        [insetX, h - insetY],
      ];
    }
    return suggest.map(p => [p[0], p[1]]);
  }

  function drawPictureCanvas() {
    if (!pictureCanvas || !pictureImg?.src) return;
    const ctx = pictureCanvas.getContext('2d');
    if (!ctx) return;

    pictureCanvas.width = pictureImg.naturalWidth;
    pictureCanvas.height = pictureImg.naturalHeight;
    ctx.clearRect(0, 0, pictureCanvas.width, pictureCanvas.height);
    ctx.drawImage(pictureImg, 0, 0);

    // Draw corners
    ctx.fillStyle = 'rgba(255,0,0,0.8)';
    ctx.strokeStyle = 'rgba(0,255,0,0.8)';
    ctx.lineWidth = 2;
    for (let i = 0; i < pictureCorners.length; i++) {
      const [x, y] = pictureCorners[i];
      ctx.beginPath();
      ctx.arc(x, y, pictureCornerRadius, 0, Math.PI * 2);
      ctx.fill();
    }
    if (pictureCorners.length === 4) {
      ctx.beginPath();
      ctx.moveTo(pictureCorners[0][0], pictureCorners[0][1]);
      ctx.lineTo(pictureCorners[1][0], pictureCorners[1][1]);
      ctx.lineTo(pictureCorners[2][0], pictureCorners[2][1]);
      ctx.lineTo(pictureCorners[3][0], pictureCorners[3][1]);
      ctx.closePath();
      ctx.stroke();
    }
  }

  function canvasEventToImageXY(canvasEl, evt) {
    const rect = canvasEl.getBoundingClientRect();
    const scaleX = canvasEl.width / rect.width;
    const scaleY = canvasEl.height / rect.height;
    const x = (evt.clientX - rect.left) * scaleX;
    const y = (evt.clientY - rect.top) * scaleY;
    return [x, y];
  }

  async function uploadPicture(file) {
    setPictureStatus('Uploading...');
    const fd = new FormData();
    fd.append('image', file);
    const res = await fetch('/api/suggest', { method: 'POST', body: fd });
    if (!res.ok) {
      setPictureStatus('Upload failed.', true);
      return null;
    }
    return await res.json();
  }

  async function warpPicture() {
    if (!pictureSessionId || pictureCorners.length !== 4) return null;
    setPictureStatus('Warping...');
    const res = await fetch('/api/warp', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session: pictureSessionId, corners: pictureCorners }),
    });
    if (!res.ok) {
      setPictureStatus('Warp failed.', true);
      return null;
    }
    return await res.json();
  }

  async function extractAndLoadBoard() {
    if (!pictureSessionId) return null;
    setPictureStatus('Extracting (CNN)...');

    const basePayload = {
      session: pictureSessionId,
      innerMargin: 0.01,
      outerMargin: 0.05,
      minConf: 75,
      psm: 10,
      psmFallback: 8,
      binarize: false,
      isolateMain: false,
    };

    async function callExtract(engine) {
      const payload = { ...basePayload, engine };
      const res = await fetch('/api/extract', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (res.ok) return await res.json();
      const contentType = (res.headers.get('content-type') || '').toLowerCase();
      if (contentType.includes('application/json')) {
        const data = await res.json().catch(() => ({}));
        return { __error: true, ...data };
      }
      const text = await res.text().catch(() => '');
      return { __error: true, error: text || `Extract failed (HTTP ${res.status}).` };
    }

    // Prefer CNN; if unavailable, fall back to Tesseract for a working baseline.
    const cnnOut = await callExtract('cnn');
    if (!cnnOut || cnnOut.__error) {
      const code = cnnOut && cnnOut.code;
      if (code === 'cnn_unavailable' || code === 'cnn_checkpoint_missing' || code === 'cnn_load_failed') {
        setPictureStatus('CNN unavailable; falling back to Tesseract...');
        const tessOut = await callExtract('tesseract');
        if (tessOut && !tessOut.__error) return tessOut;
        setPictureStatus((tessOut && tessOut.error) || 'Extract failed.', true);
        return null;
      }
      setPictureStatus((cnnOut && cnnOut.error) || 'Extract failed.', true);
      return null;
    }
    return cnnOut;
  }

  function emptyBoardString() {
    const row = '.'.repeat(15);
    return Array.from({ length: 15 }, () => row).join('\n');
  }

  function normalizeBoardString(s) {
    const lines = (s || '').trim().split(/\r?\n/).filter(Boolean);
    if (lines.length !== 15) return null;
    for (const line of lines) {
      if (line.length !== 15) return null;
    }
    return lines.join('\n');
  }

  function parseBoard(s) {
    const norm = normalizeBoardString(s);
    if (!norm) return null;
    const lines = norm.split('\n');
    const grid = lines.map(line => line.split('').map(ch => (ch === '.' ? null : ch)));
    return grid;
  }

  function boardToString(grid) {
    return grid.map(row => row.map(ch => (ch == null ? '.' : ch)).join('')).join('\n');
  }

  function setStatus(text, isError = false) {
    moveStatus.textContent = text || '';
    moveStatus.style.color = isError ? '#b00020' : '';
  }

  function setMoveDetails(html) {
    if (!html) {
      moveDetails.style.display = 'none';
      moveDetails.innerHTML = '';
      return;
    }
    moveDetails.style.display = 'block';
    moveDetails.innerHTML = html;
  }

  function renderBoard(grid) {
    playBoard.innerHTML = '';
    const lang = langEl.value || 'EN';
    for (let r = 0; r < 15; r++) {
      for (let c = 0; c < 15; c++) {
        const cell = document.createElement('div');
        cell.className = 'play-cell';
        cell.dataset.row = String(r);
        cell.dataset.col = String(c);
        cell.title = `(${r + 1},${c + 1}) Click to edit`;

        const prem = premiums?.[r]?.[c] || '..';
        if (prem === 'TW') cell.classList.add('tw');
        if (prem === 'TL') cell.classList.add('tl');
        if (prem === 'DL') cell.classList.add('dl');
        if (prem === 'DW') cell.classList.add('dw');
        if (prem === 'DW') cell.classList.add('dw');
        if (prem === 'DL') cell.classList.add('dl');

        const ch = grid[r][c];
        if (ch != null) {
          cell.classList.add('filled');
          const displayLetter = String(ch).toUpperCase();

          const letterEl = document.createElement('span');
          letterEl.className = 'play-letter';
          letterEl.textContent = displayLetter;
          cell.appendChild(letterEl);

          const scoreEl = document.createElement('span');
          scoreEl.className = 'play-score';
          const isBlank = String(ch) >= 'a' && String(ch) <= 'z';
          const score = isBlank ? 0 : (letterScores?.[lang]?.[displayLetter] ?? 0);
          scoreEl.textContent = String(score);
          cell.appendChild(scoreEl);

          if (isBlank) cell.classList.add('blank');
        }

        // tiny premium hint for empty squares
        if (ch == null && prem !== '..') {
          const hint = document.createElement('span');
          hint.className = 'play-premium';
          hint.textContent = prem;
          cell.appendChild(hint);
        }

        playBoard.appendChild(cell);
      }
    }
  }

  function _parseSingleTileInput(raw) {
    const s = String(raw ?? '').trim();
    if (!s) return { ok: false, error: 'Empty input.' };
    if (s === '.' || s.toLowerCase() === 'empty') return { ok: true, value: null };
    if (s === '?') return { ok: true, value: '?' };
    if (/^[A-Za-z]$/.test(s)) {
      // Uppercase = normal tile, lowercase = blank tile representing that letter.
      return { ok: true, value: s };
    }
    return { ok: false, error: "Invalid tile. Use A-Z, a-z (blank), '.', or '?'" };
  }

  function editCellAt(row, col) {
    const grid = parseBoard(boardStringEl.value);
    if (!grid) {
      setStatus('Board string must be 15 lines of 15 chars.', true);
      return;
    }
    const current = grid?.[row]?.[col];
    const promptMsg =
      `Set tile at (${row + 1},${col + 1}).\n` +
      `- A-Z: normal tile\n` +
      `- a-z: blank tile for that letter\n` +
      `- ?: unknown/blank tile\n` +
      `- . : clear\n\n` +
      `Current: ${current == null ? '.' : String(current)}`;

    const nextRaw = window.prompt(promptMsg, current == null ? '' : String(current));
    if (nextRaw == null) return; // cancelled
    const parsed = _parseSingleTileInput(nextRaw);
    if (!parsed.ok) {
      setStatus(parsed.error, true);
      return;
    }

    grid[row][col] = parsed.value;
    boardStringEl.value = boardToString(grid);
    renderBoard(grid);

    // Editing invalidates the current best-move preview.
    lastMove = null;
    placeBestBtn.disabled = true;
    setMoveDetails('');
    setStatus(`Updated (${row + 1},${col + 1}) to ${parsed.value == null ? '.' : String(parsed.value).toUpperCase()}.`);
  }

  function applyMoveToGrid(grid, move) {
    const { word, row, col, dir } = move;
    const dr = dir === 'H' ? 0 : 1;
    const dc = dir === 'H' ? 1 : 0;

    for (let i = 0; i < word.length; i++) {
      const r = row + i * dr;
      const c = col + i * dc;
      if (grid[r][c] == null) {
        grid[r][c] = word[i];
      }
    }
  }

  function makeBagEN() {
    const dist = {
      A: 9, B: 2, C: 2, D: 4, E: 12, F: 2, G: 3, H: 2, I: 9, J: 1, K: 1, L: 4,
      M: 2, N: 6, O: 8, P: 2, Q: 1, R: 6, S: 4, T: 6, U: 4, V: 2, W: 2, X: 1,
      Y: 2, Z: 1, '?': 2,
    };
    const bag = [];
    for (const [ch, n] of Object.entries(dist)) {
      for (let i = 0; i < n; i++) bag.push(ch);
    }
    return bag;
  }

  function makeBagFR() {
    const dist = {
      A: 9, B: 2, C: 2, D: 3, E: 15, F: 2, G: 2, H: 2, I: 8, J: 1, K: 1, L: 5,
      M: 3, N: 6, O: 6, P: 2, Q: 1, R: 6, S: 6, T: 6, U: 6, V: 2, W: 1, X: 1,
      Y: 1, Z: 1, '?': 2,
    };
    const bag = [];
    for (const [ch, n] of Object.entries(dist)) {
      for (let i = 0; i < n; i++) bag.push(ch);
    }
    return bag;
  }

  function randomRack(lang) {
    const bag = (lang === 'FR') ? makeBagFR() : makeBagEN();
    // shuffle
    for (let i = bag.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [bag[i], bag[j]] = [bag[j], bag[i]];
    }
    return bag.slice(0, 7).join('');
  }

  async function findBest() {
    setStatus('Searching...');
    setMoveDetails('');
    placeBestBtn.disabled = true;
    lastMove = null;

    const grid = parseBoard(boardStringEl.value);
    if (!grid) {
      setStatus('Board string must be 15 lines of 15 chars.', true);
      return;
    }

    const rack = (rackEl.value || '').trim();
    if (!rack) {
      setStatus('Rack is empty.', true);
      return;
    }

    const payload = {
      boardString: boardToString(grid),
      rack,
      lang: langEl.value,
      dictKey: dictKeyEl.value,
    };

    let res;
    let data;
    try {
      res = await fetch('/api/move/best', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      data = await res.json();
    } catch (err) {
      setStatus(
        "Webapp backend not reachable. Start it with: python -m scrabble.webapp",
        true
      );
      return;
    }
    if (!res.ok) {
      setStatus(data.error || 'Failed to compute best move.', true);
      return;
    }

    if (!data.move) {
      setStatus('No valid moves found.');
      return;
    }

    lastMove = data.move;
    placeBestBtn.disabled = false;

    setStatus('Best move found.');
    setMoveDetails(
      `<div><strong>Best:</strong> ${lastMove.word} at (${lastMove.row},${lastMove.col}) ${lastMove.dir} score=${lastMove.score}</div>`
    );

    // Preview board after move
    const preview = parseBoard(payload.boardString);
    applyMoveToGrid(preview, lastMove);
    renderBoard(preview);
  }

  function placeBest() {
    if (!lastMove) return;
    const grid = parseBoard(boardStringEl.value);
    if (!grid) {
      setStatus('Board string must be 15 lines of 15 chars.', true);
      return;
    }

    applyMoveToGrid(grid, lastMove);
    boardStringEl.value = boardToString(grid);
    renderBoard(grid);

    setStatus('Move placed.');
    placeBestBtn.disabled = true;
    lastMove = null;
  }

  function loadBoardFromText() {
    const grid = parseBoard(boardStringEl.value);
    if (!grid) {
      setStatus('Board string must be 15 lines of 15 chars.', true);
      return;
    }
    setStatus('');
    renderBoard(grid);
  }

  function clearBoard() {
    boardStringEl.value = emptyBoardString();
    renderBoard(parseBoard(boardStringEl.value));
    setStatus('');
    setMoveDetails('');
    lastMove = null;
    placeBestBtn.disabled = true;
  }

  // Init
  boardStringEl.value = emptyBoardString();
  renderBoard(parseBoard(boardStringEl.value));

  // Picture loader init/events
  if (loadFromPictureBtn && pictureLoaderEl) {
    loadFromPictureBtn.addEventListener('click', () => {
      showPictureLoader(true);
      setPictureStatus('Select an image to begin.');
    });
  }

  if (pictureCloseBtn) {
    pictureCloseBtn.addEventListener('click', () => {
      showPictureLoader(false);
      setPictureStatus('');
    });
  }

  if (picturePickBtn && pictureInput) {
    picturePickBtn.addEventListener('click', () => pictureInput.click());
  }

  if (pictureInput) {
    pictureInput.addEventListener('change', async () => {
      if (!pictureInput.files || !pictureInput.files.length) return;
      const data = await uploadPicture(pictureInput.files[0]);
      if (!data) return;

      pictureSessionId = data.session;
      const sourceUrl = data.sourceUrl;

      pictureWarpPreview.style.display = 'none';
      pictureWarpPreview.removeAttribute('src');
      pictureWarpBtn.disabled = true;
      pictureExtractBtn.disabled = true;

      pictureImg.onload = () => {
        pictureCorners = normalizeCornersForImage(pictureImg, data.suggestedCorners);
        drawPictureCanvas();
        if (pictureWarpBtn) pictureWarpBtn.disabled = false;
        if (pictureExtractBtn) pictureExtractBtn.disabled = false;
        setPictureStatus('Adjust corners, then Extract & load.');
      };
      pictureImg.src = sourceUrl;
    });
  }

  if (pictureCanvas) {
    pictureCanvas.addEventListener('mousedown', (e) => {
      if (!pictureCorners || pictureCorners.length !== 4) return;
      const [x, y] = canvasEventToImageXY(pictureCanvas, e);
      pictureDragIndex = -1;
      for (let i = 0; i < pictureCorners.length; i++) {
        const [cx, cy] = pictureCorners[i];
        const d2 = (cx - x) ** 2 + (cy - y) ** 2;
        if (d2 <= (pictureCornerRadius * 2) ** 2) {
          pictureDragIndex = i;
          break;
        }
      }
    });

    pictureCanvas.addEventListener('mousemove', (e) => {
      if (pictureDragIndex < 0) return;
      const [x0, y0] = canvasEventToImageXY(pictureCanvas, e);
      const x = Math.max(0, Math.min(pictureCanvas.width - 1, x0));
      const y = Math.max(0, Math.min(pictureCanvas.height - 1, y0));
      pictureCorners[pictureDragIndex] = [x, y];
      drawPictureCanvas();
    });
  }

  window.addEventListener('mouseup', () => {
    pictureDragIndex = -1;
  });

  if (pictureWarpBtn) {
    pictureWarpBtn.addEventListener('click', async () => {
      const warpData = await warpPicture();
      if (!warpData) return;
      if (pictureWarpPreview) {
        pictureWarpPreview.style.display = 'block';
        pictureWarpPreview.src = warpData.gridUrl + `?t=${Date.now()}`;
      }
      setPictureStatus('Warped. You can extract now.');
    });
  }

  if (pictureExtractBtn) {
    pictureExtractBtn.addEventListener('click', async () => {
      // Ensure warp is up-to-date before extracting.
      const warpData = await warpPicture();
      if (!warpData) return;
      if (pictureWarpPreview) {
        pictureWarpPreview.style.display = 'block';
        pictureWarpPreview.src = warpData.gridUrl + `?t=${Date.now()}`;
      }

      const extracted = await extractAndLoadBoard();
      if (!extracted) return;

      const bs = extracted.boardString;
      const norm = normalizeBoardString(bs);
      if (!norm) {
        setPictureStatus('Extract returned invalid board string.', true);
        return;
      }

      boardStringEl.value = norm;
      loadBoardFromText();
      showPictureLoader(false);
      setStatus('Board loaded from picture.');
    });
  }

  // Events
  playBoard.addEventListener('click', (e) => {
    const cellEl = e.target?.closest?.('.play-cell');
    if (!cellEl || !playBoard.contains(cellEl)) return;
    const r = parseInt(cellEl.dataset.row, 10);
    const c = parseInt(cellEl.dataset.col, 10);
    if (!Number.isFinite(r) || !Number.isFinite(c)) return;
    editCellAt(r, c);
  });

  randomRackBtn.addEventListener('click', () => {
    rackEl.value = randomRack(langEl.value);
  });

  findBestBtn.addEventListener('click', () => {
    findBest().catch((err) => setStatus(String(err), true));
  });

  placeBestBtn.addEventListener('click', () => placeBest());
  clearBoardBtn.addEventListener('click', () => clearBoard());
  loadFromTextBtn.addEventListener('click', () => loadBoardFromText());

  // Keep dict in sync with language selection by default
  langEl.addEventListener('change', () => {
    if (langEl.value === 'FR') dictKeyEl.value = 'fr_ods8';
    else dictKeyEl.value = 'en_small';
    // Re-render so tile values update with language.
    loadBoardFromText();
  });
})();
