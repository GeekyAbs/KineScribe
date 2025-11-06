document.addEventListener('DOMContentLoaded', () => {
    const API_BASE = "http://127.0.0.1:5000"; // Backend URL
    const mediaInput = document.getElementById('media-input');
    const fileName = document.getElementById('file-name');
    const analyzeBtn = document.getElementById('analyze-btn');
    const telegramId = document.getElementById('telegram-id');
    const previewImg = document.getElementById('preview-img');
    const previewVid = document.getElementById('preview-vid');
    const previewPlaceholder = document.getElementById('preview-placeholder');
    const logOutput = document.getElementById('log-output');
    const clearLogBtn = document.getElementById('clear-log');
    const statusBadge = document.querySelector('.status-badge');
    const steps = {
      yolo: document.getElementById('step-1').querySelector('.step-status'),
      gemini: document.getElementById('step-2').querySelector('.step-status'),
      telegram: document.getElementById('step-3').querySelector('.step-status')
    };
  
    let selectedFile = null;
    let objectUrl = null;
  
    function log(type, msg) {
      const now = new Date().toLocaleTimeString();
      const div = document.createElement('div');
      div.className = `log-entry ${type}`;
      div.innerHTML = `<span class="time">[${now}]</span> ${msg}`;
      logOutput.appendChild(div);
      logOutput.scrollTop = logOutput.scrollHeight;
    }
  
    function resetSteps() {
      steps.yolo.textContent = '—';
      steps.gemini.textContent = '—';
      steps.telegram.textContent = '—';
    }
  
    mediaInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (objectUrl) URL.revokeObjectURL(objectUrl);
  
      if (!file) {
        fileName.textContent = "No file selected";
        previewPlaceholder.style.display = 'flex';
        previewImg.style.display = 'none';
        previewVid.style.display = 'none';
        selectedFile = null;
        analyzeBtn.disabled = true;
        return;
      }
  
      selectedFile = file;
      fileName.textContent = file.name.length > 25 ? file.name.slice(0, 22) + '...' : file.name;
      analyzeBtn.disabled = !(file && telegramId.value.trim());
  
      objectUrl = URL.createObjectURL(file);
      previewPlaceholder.style.display = 'none';
      if (file.type.startsWith('image/')) {
        previewImg.src = objectUrl;
        previewImg.style.display = 'block';
        previewVid.style.display = 'none';
      } else if (file.type.startsWith('video/')) {
        previewVid.src = objectUrl;
        previewVid.style.display = 'block';
        previewImg.style.display = 'none';
      }
    });
  
    telegramId.addEventListener('input', () => {
      analyzeBtn.disabled = !(selectedFile && telegramId.value.trim());
    });
  
    clearLogBtn.addEventListener('click', () => {
      logOutput.innerHTML = '';
    });
  
    analyzeBtn.addEventListener('click', async () => {
      if (!selectedFile || !telegramId.value.trim()) return;
  
      resetSteps();
      statusBadge.className = 'status-badge processing';
      statusBadge.textContent = 'Uploading...';
      log('info', `📤 Uploading ${selectedFile.name} (${(selectedFile.size / 1024 / 1024).toFixed(2)} MB)`);
  
      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("telegram_id", telegramId.value.trim());
  
      try {
        const res = await fetch(`${API_BASE}/predict`, {
          method: "POST",
          body: formData
        });
  
        const data = await res.json();
  
        if (!res.ok) {
          throw new Error(data.error || "Unknown backend error");
        }
  
        log('info', `✅ Backend response received`);
        steps.yolo.textContent = data.yolo_triggered ? '✅ Triggered' : '❌ Not detected';
        log('info', `🔍 YOLO: ${data.yolo_detections.length} detections`);
  
        if (data.yolo_triggered) {
          steps.yolo.textContent = '✅ Detected';
          steps.gemini.textContent = '⏳ Calling...';
  
          if (data.gemini_result?.error) {
            steps.gemini.textContent = '⚠️ Failed';
            log('warn', `❌ LLM Error: ${data.gemini_result.error}`);
          } else if (data.gemini_result) {
            const conf = data.gemini_result.confidence || 0;
            const fall = data.gemini_result.fall_detected;
            steps.gemini.textContent = fall ? '✅ Confirmed' : '❌ Rejected';
            log('success', `🧠 LLM: ${fall ? 'Fall confirmed' : 'False alarm'} (${conf.toFixed(1)}%)`);
  
            if (data.telegram_sent) {
              steps.telegram.textContent = '✅ Sent!';
              log('success', `📲 Telegram alert sent to ${telegramId.value}`);
              statusBadge.className = 'status-badge alert';
              statusBadge.textContent = 'ALERT SENT';
            } else if (data.reason_skipped) {
              steps.telegram.textContent = '⏭️ Skipped';
              log('info', `⏭️ Alert skipped: ${data.reason_skipped}`);
              statusBadge.className = 'status-badge success';
              statusBadge.textContent = 'Processed';
            }
          }
        } else {
          steps.yolo.textContent = '❌ No fall';
          statusBadge.className = 'status-badge success';
          statusBadge.textContent = 'No Fall';
          log('info', `✅ No high-confidence fall detected.`);
        }
      } catch (err) {
        log('warn', `💥 Upload/Prediction failed: ${err.message}`);
        statusBadge.className = 'status-badge warn';
        statusBadge.textContent = 'Error';
        resetSteps();
      }
    });
  
    // Optional: Test backend connection on load
    fetch(`${API_BASE}/`)
      .then(() => log('info', '🔌 Connected to KineScribe backend'))
      .catch(() => log('warn', '⚠️ Backend unreachable (is Flask running?)'));
    
    // ==========================================
// 🖱️ CUSTOM CURSOR + PARTICLES
// ==========================================
(function() {
    // Cursor elements
    const cursor = document.createElement('div');
    const follower = document.createElement('div');
    cursor.className = 'custom-cursor';
    follower.className = 'custom-cursor follower';
    document.body.appendChild(cursor);
    document.body.appendChild(follower);
  
    let posX = 0, posY = 0;
    let mouseX = 0, mouseY = 0;
  
    document.addEventListener('mousemove', (e) => {
      mouseX = e.clientX;
      mouseY = e.clientY;
    });
  
    function animateCursor() {
      const distX = mouseX - posX;
      const distY = mouseY - posY;
      posX += distX * 0.1;
      posY += distY * 0.1;
  
      cursor.style.left = posX + 'px';
      cursor.style.top = posY + 'px';
      follower.style.left = posX + 'px';
      follower.style.top = posY + 'px';
  
      requestAnimationFrame(animateCursor);
    }
    animateCursor();
  
    // Enlarge on interactive elements
    const interactive = 'a, button, input, .card, .step-icon';
    document.querySelectorAll(interactive).forEach(el => {
      el.addEventListener('mouseenter', () => {
        cursor.style.transform = 'translate(-50%, -50%) scale(1.4)';
        cursor.style.opacity = '1';
      });
      el.addEventListener('mouseleave', () => {
        cursor.style.transform = 'translate(-50%, -50%) scale(1)';
        cursor.style.opacity = '0.8';
      });
    });
  
    // ==========================================
    // 🌠 FLOATING PARTICLES (light, GPU-friendly)
    // ==========================================
    const particleCount = Math.min(12, window.innerWidth / 80);
    for (let i = 0; i < particleCount; i++) {
      const p = document.createElement('div');
      p.className = 'particle';
      
      // Random size & position
      const size = Math.random() * 4 + 1;
      const x = Math.random() * 100;
      const y = Math.random() * 100;
      const duration = Math.random() * 20 + 20;
      const delay = Math.random() * 5;
  
      p.style.width = `${size}px`;
      p.style.height = `${size}px`;
      p.style.left = `${x}vw`;
      p.style.top = `${y}vh`;
      p.style.animation = `float ${duration}s ease-in-out ${delay}s infinite`;
  
      document.body.appendChild(p);
    }
  
    // Particle float keyframes (add once)
    if (!document.getElementById('particle-anim')) {
      const style = document.createElement('style');
      style.id = 'particle-anim';
      style.textContent = `
        @keyframes float {
          0%, 100% { transform: translate(0, 0) rotate(0deg); opacity: 0.3; }
          25% { transform: translate(${Math.random() * 40 - 20}px, ${Math.random() * 40 - 20}px) rotate(5deg); opacity: 0.5; }
          50% { transform: translate(${Math.random() * 60 - 30}px, ${Math.random() * 60 - 30}px) rotate(-5deg); opacity: 0.2; }
          75% { transform: translate(${Math.random() * 50 - 25}px, ${Math.random() * 50 - 25}px) rotate(3deg); opacity: 0.4; }
        }
      `;
      document.head.appendChild(style);
    }
  })();
  });