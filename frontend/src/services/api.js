/**
 * API Service for Indian Sign Language Recognition Backend
 */

const API_BASE = '/api';

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}

export async function startCamera(cameraId = 0, frameSkip = 2) {
  const res = await fetch(`${API_BASE}/camera/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ camera_id: cameraId, frame_skip: frameSkip })
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || 'Unable to start camera');
  return data;
}

export async function stopCamera() {
  const res = await fetch(`${API_BASE}/camera/stop`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' }
  });
  return res.json();
}

export async function fetchCameraStatus() {
  const res = await fetch(`${API_BASE}/camera/status`);
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || 'Unable to fetch camera status');
  return data;
}

export async function fetchModelInfo() {
  const res = await fetch(`${API_BASE}/model/info`);
  return res.json();
}

export async function executeTextAction(action) {
  const res = await fetch(`${API_BASE}/text/action`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action })
  });
  return res.json();
}

export async function synthesizeTTS(text, lang = 'en') {
  const res = await fetch(`${API_BASE}/tts`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, lang })
  });
  return res.json();
}

export async function fetchHistory() {
  const res = await fetch(`${API_BASE}/history`);
  return res.json();
}

export function createPredictionWebSocket(onMessage, onError, onClose) {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/api/ws/predictions`;
  
  const ws = new WebSocket(wsUrl);
  ws.onopen = () => console.info('[WebSocket] Prediction stream connected');
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch (err) {
      console.error('[WebSocket] Parse error:', err);
    }
  };
  ws.onerror = (err) => {
    if (onError) onError(err);
  };
  ws.onclose = () => {
    if (onClose) onClose();
  };
  return ws;
}
