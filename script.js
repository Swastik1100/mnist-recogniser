const canvas = document.getElementById('canvas');
const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');
const predictionEl = document.getElementById('prediction');

const ctx = canvas.getContext('2d');
ctx.fillStyle = '#000';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = '#fff';
ctx.lineWidth = 18;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

let drawing = false;

function getPosition(event) {
  const rect = canvas.getBoundingClientRect();
  const point = event.touches ? event.touches[0] : event;
  return {
    x: ((point.clientX - rect.left) / rect.width) * canvas.width,
    y: ((point.clientY - rect.top) / rect.height) * canvas.height,
  };
}

function startDrawing(event) {
  drawing = true;
  const { x, y } = getPosition(event);
  ctx.beginPath();
  ctx.moveTo(x, y);
  event.preventDefault();
}

function draw(event) {
  if (!drawing) return;
  const { x, y } = getPosition(event);
  ctx.lineTo(x, y);
  ctx.stroke();
  event.preventDefault();
}

function stopDrawing() {
  drawing = false;
}

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseleave', stopDrawing);
canvas.addEventListener('touchstart', startDrawing, { passive: false });
canvas.addEventListener('touchmove', draw, { passive: false });
canvas.addEventListener('touchend', stopDrawing);

clearBtn.addEventListener('click', () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  predictionEl.textContent = '-';
});

predictBtn.addEventListener('click', async () => {
  predictionEl.textContent = '...';
  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: canvas.toDataURL('image/png') }),
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Prediction failed');
    predictionEl.textContent = data.prediction;
  } catch (error) {
    predictionEl.textContent = 'Err';
    console.error(error);
  }
});
