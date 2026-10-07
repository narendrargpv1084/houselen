/* ═══════════════════════════════════════════════════════════════
   HouseLens — Frontend Logic
   ═══════════════════════════════════════════════════════════════ */

'use strict';

// ── Stepper buttons ──────────────────────────────────────────────────────────
document.querySelectorAll('.stepper').forEach(stepper => {
  const input = stepper.querySelector('input[type="number"]');
  stepper.querySelectorAll('.step-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const dir = parseInt(btn.dataset.dir, 10);
      const min = parseInt(input.min, 10);
      const max = parseInt(input.max, 10);
      const val = parseInt(input.value, 10) + dir;
      input.value = Math.min(max, Math.max(min, val));
    });
  });
});

// ── Reset button ─────────────────────────────────────────────────────────────
document.getElementById('reset-btn').addEventListener('click', () => {
  document.getElementById('area').value = '';
  document.getElementById('bedrooms').value = 3;
  document.getElementById('bathrooms').value = 2;
  document.getElementById('stories').value = 2;
  document.getElementById('parking').value = 1;
  document.getElementById('furnishingstatus').value = 'semi-furnished';

  document.querySelectorAll('.toggle-switch input[type="checkbox"]').forEach(cb => {
    cb.checked = cb.id === 'mainroad' || cb.id === 'airconditioning';
  });

  clearFieldErrors();
  showPlaceholder();
});

// ── Recalc / retry buttons ───────────────────────────────────────────────────
document.getElementById('recalc-btn').addEventListener('click', () => showPlaceholder());
document.getElementById('retry-btn').addEventListener('click', () => showPlaceholder());

// ── Form submit ──────────────────────────────────────────────────────────────
document.getElementById('predict-form').addEventListener('submit', async e => {
  e.preventDefault();
  if (!validateForm()) return;

  setLoading(true);
  hideAll();

  const payload = buildPayload();

  try {
    const res  = await fetch('/api/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok || data.error) {
      showError(data.error || 'Prediction failed. Please try again.');
      return;
    }

    renderResult(data);
  } catch (err) {
    showError('Network error — is the server running?');
  } finally {
    setLoading(false);
  }
});

// ── Build payload from form ──────────────────────────────────────────────────
function buildPayload() {
  const bool = id => document.getElementById(id).checked ? 'yes' : 'no';
  return {
    area:             parseInt(document.getElementById('area').value, 10),
    bedrooms:         parseInt(document.getElementById('bedrooms').value, 10),
    bathrooms:        parseInt(document.getElementById('bathrooms').value, 10),
    stories:          parseInt(document.getElementById('stories').value, 10),
    parking:          parseInt(document.getElementById('parking').value, 10),
    furnishingstatus: document.getElementById('furnishingstatus').value,
    mainroad:         bool('mainroad'),
    guestroom:        bool('guestroom'),
    basement:         bool('basement'),
    hotwaterheating:  bool('hotwaterheating'),
    airconditioning:  bool('airconditioning'),
    prefarea:         bool('prefarea'),
  };
}

// ── Render result ────────────────────────────────────────────────────────────
function renderResult(data) {
  const price = data.predicted_price;
  const low   = data.range_low;
  const high  = data.range_high;

  document.getElementById('result-price').textContent = formatPrice(price);
  document.getElementById('result-range').textContent =
    `${formatPrice(low)}  –  ${formatPrice(high)}`;
  document.getElementById('meta-model').textContent =
    data.model.replace('Regressor', '').replace('Regression', 'Reg.');

  document.getElementById('bar-label-low').textContent  = formatPrice(low);
  document.getElementById('bar-label-mid').textContent  = formatPrice(price);
  document.getElementById('bar-label-high').textContent = formatPrice(high);

  document.getElementById('result-placeholder').classList.add('hidden');
  document.getElementById('result-error').classList.add('hidden');
  const content = document.getElementById('result-content');
  content.classList.remove('hidden');
  // re-trigger animation
  content.style.animation = 'none';
  content.offsetHeight;          // reflow
  content.style.animation = '';

  // scroll result into view on mobile
  if (window.innerWidth <= 1024) {
    setTimeout(() => {
      document.getElementById('result-panel').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 150);
  }
}

// ── Helpers ──────────────────────────────────────────────────────────────────
function formatPrice(n) {
  if (n >= 1e7) return `₹ ${(n / 1e7).toFixed(2)} Cr`;
  if (n >= 1e5) return `₹ ${(n / 1e5).toFixed(2)} L`;
  return `₹ ${n.toLocaleString('en-IN')}`;
}

function setLoading(on) {
  const btn     = document.getElementById('submit-btn');
  const label   = document.getElementById('btn-label');
  const icon    = document.getElementById('btn-icon');
  const spinner = document.getElementById('btn-spinner');
  btn.disabled  = on;
  label.textContent = on ? 'Predicting…' : 'Predict Price';
  icon.classList.toggle('hidden', on);
  spinner.classList.toggle('hidden', !on);
}

function showPlaceholder() {
  document.getElementById('result-placeholder').classList.remove('hidden');
  document.getElementById('result-content').classList.add('hidden');
  document.getElementById('result-error').classList.add('hidden');
}

function hideAll() {
  document.getElementById('result-placeholder').classList.add('hidden');
  document.getElementById('result-content').classList.add('hidden');
  document.getElementById('result-error').classList.add('hidden');
}

function showError(msg) {
  document.getElementById('error-msg').textContent = msg;
  document.getElementById('result-error').classList.remove('hidden');
}

// ── Validation ───────────────────────────────────────────────────────────────
function validateForm() {
  clearFieldErrors();
  let valid = true;

  const area = document.getElementById('area');
  const val  = parseInt(area.value, 10);
  if (!area.value || isNaN(val) || val < 500 || val > 50000) {
    setFieldError(area, 'Enter a valid area (500 – 50,000 sq ft)');
    valid = false;
  }

  return valid;
}

function setFieldError(input, msg) {
  input.classList.add('error');
  const errEl = input.closest('.field')?.querySelector('.field-error');
  if (errEl) errEl.textContent = msg;
}

function clearFieldErrors() {
  document.querySelectorAll('.field input.error').forEach(el => {
    el.classList.remove('error');
  });
  document.querySelectorAll('.field-error').forEach(el => {
    el.textContent = '';
  });
}

// ── Navbar scroll shadow ─────────────────────────────────────────────────────
window.addEventListener('scroll', () => {
  document.querySelector('.navbar').style.boxShadow =
    window.scrollY > 10 ? '0 2px 20px rgba(0,0,0,.5)' : 'none';
}, { passive: true });
