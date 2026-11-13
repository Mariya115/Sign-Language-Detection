import React, { useState, useEffect, useRef } from 'react';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import RecognitionPage from './pages/RecognitionPage';
import ModelInfoPage from './pages/ModelInfoPage';
import HistoryPage from './pages/HistoryPage';
import { fetchHealth, fetchCameraStatus, fetchModelInfo, fetchHistory, createPredictionWebSocket } from './services/api';

function mergeLiveResult(previous, incoming) {
  const next = { ...previous, ...incoming };
  const boxesDetected = Array.isArray(next.boxes) && next.boxes.length > 0;
  const landmarks = Array.isArray(next.landmarks) ? next.landmarks : [];
  const landmarksDetected = next.landmarks_detected === true
    || next.landmark_count > 0
    || landmarks.length > 0;

  return {
    ...next,
    landmarks_detected: landmarksDetected,
    hand_detected: next.hand_detected === true || boxesDetected || landmarksDetected,
  };
}

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [backendHealthy, setBackendHealthy] = useState(false);
  const [cameraRunning, setCameraRunning] = useState(false);
  const [latestResult, setLatestResult] = useState({
    prediction: 'Camera Idle',
    confidence: 0.0,
    fps: 0.0,
    latency_ms: 0.0,
    hand_detected: false,
    landmarks_detected: false,
    landmark_count: 0,
    gesture_recognized: false,
    predicted_class: null,
    kannada_label: null,
    status: 'camera_stopped',
    num_hands: 0,
    sentence: '',
    probabilities: {},
  });
  const [history, setHistory] = useState([]);
  const [modelInfo, setModelInfo] = useState(null);

  const wsRef = useRef(null);

  // Initial health check and metadata loading
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const health = await fetchHealth();
        if (health.status === 'healthy') {
          setBackendHealthy(true);
        }
      } catch (err) {
        setBackendHealthy(false);
      }
    };

    const loadMetadata = async () => {
      try {
        const info = await fetchModelInfo();
        setModelInfo(info);
        const hist = await fetchHistory();
        if (hist.history) setHistory(hist.history);
        const camStatus = await fetchCameraStatus();
        setCameraRunning(camStatus.is_running);
        if (camStatus.result) {
          setLatestResult((previous) => mergeLiveResult(previous, {
            ...camStatus.result,
            is_running: camStatus.is_running,
          }));
        }
      } catch (err) {
        console.error('Metadata load error:', err);
      }
    };

    checkBackend();
    loadMetadata();
    const interval = setInterval(checkBackend, 10000);
    const statusInterval = setInterval(async () => {
      try {
        const status = await fetchCameraStatus();
        setCameraRunning(status.is_running);
        if (status.result) {
          setLatestResult((previous) => mergeLiveResult(previous, {
            ...status.result,
            is_running: status.is_running,
          }));
        }
      } catch (err) {
        console.warn('Camera status refresh failed:', err);
      }
    }, 1000);
    return () => {
      clearInterval(interval);
      clearInterval(statusInterval);
    };
  }, []);

  // Real-time WebSocket for live prediction stream
  useEffect(() => {
    let active = true;
    const connectWS = () => {
      if (!active) return;
      try {
        wsRef.current = createPredictionWebSocket(
          (data) => {
            setLatestResult((previous) => mergeLiveResult(previous, data));
            setCameraRunning(data.is_running);

            // Record to history if new confident sign
            const pred = data.prediction;
            if (data.gesture_recognized && pred) {
              setHistory((prev) => {
                if (prev.length > 0 && prev[0].sign === pred) return prev;
                return [
                  {
                    id: prev.length + 1,
                    sign: pred,
                    confidence: data.confidence,
                    timestamp: new Date().toLocaleTimeString(),
                    num_hands: data.num_hands,
                  },
                  ...prev.slice(0, 49),
                ];
              });
            }
          },
          (err) => {
            console.warn('[WebSocket] Warning:', err);
          },
          () => {
            // Reconnect after 2s
            if (active) setTimeout(connectWS, 2000);
          }
        );
      } catch (err) {
        if (active) setTimeout(connectWS, 3000);
      }
    };

    connectWS();
    return () => {
      active = false;
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col selection:bg-emerald-500 selection:text-white">
      {/* Top Navbar */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        backendHealthy={backendHealthy}
      />

      {/* Main Content Viewport */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'dashboard' && (
          <Dashboard
            setActiveTab={setActiveTab}
            cameraRunning={cameraRunning}
            latestResult={latestResult}
            history={history}
            modelInfo={modelInfo}
          />
        )}

        {activeTab === 'recognition' && (
          <RecognitionPage
            cameraRunning={cameraRunning}
            setCameraRunning={setCameraRunning}
            latestResult={latestResult}
            modelInfo={modelInfo}
          />
        )}

        {activeTab === 'model' && (
          <ModelInfoPage modelInfo={modelInfo} />
        )}

        {activeTab === 'history' && (
          <HistoryPage
            history={history}
            setHistory={setHistory}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950/80 py-6 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>Real-Time Indian Sign Language Detection & Kannada Gesture Recognition</span>
          <span>B.Tech CSIT Major Project | PyTorch, YOLO & Hand Keypoints</span>
        </div>
      </footer>
    </div>
  );
}
