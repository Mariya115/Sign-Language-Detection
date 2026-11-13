import React, { useState, useRef, useEffect } from 'react';
import { Camera, StopCircle, Play, Volume2, RotateCcw, Delete, Space, Activity, Sparkles, Check, AlertCircle } from 'lucide-react';
import { startCamera, stopCamera, fetchCameraStatus, executeTextAction, synthesizeTTS } from '../services/api';

export default function RecognitionPage({ cameraRunning, setCameraRunning, latestResult, modelInfo }) {
  const [loading, setLoading] = useState(false);
  const [ttsLoading, setTtsLoading] = useState(false);
  const [activeLang, setActiveLang] = useState('kn');
  const [audioError, setAudioError] = useState(null);
  const [sentenceOverride, setSentenceOverride] = useState(null);
  const audioRef = useRef(null);

  const prediction = latestResult?.prediction || (cameraRunning ? 'Analyzing...' : 'Camera Stopped');
  const rawConfidence = latestResult?.raw_confidence || 0;
  const confidence = latestResult?.confidence || rawConfidence;
  const fps = latestResult?.fps || 0;
  const latency = latestResult?.latency_ms || 0;
  const boxesDetected = Array.isArray(latestResult?.boxes) && latestResult.boxes.length > 0;
  const landmarks = Array.isArray(latestResult?.landmarks) ? latestResult.landmarks : [];
  const handDetected = latestResult?.hand_detected === true
    || boxesDetected
    || latestResult?.landmarks_detected === true
    || latestResult?.landmark_count > 0
    || landmarks.length > 0;
  const landmarksDetected = latestResult?.landmarks_detected === true
    || latestResult?.landmark_count > 0
    || landmarks.length > 0;
  const landmarkCount = latestResult?.landmark_count || 0;
  const recognitionStatus = latestResult?.status || 'unknown';
  const sentence = sentenceOverride !== null ? sentenceOverride : (latestResult?.sentence || '');
  const probabilities = latestResult?.probabilities || {};
  const kannadaPrediction = latestResult?.kannada_prediction || prediction;
  const gesture = latestResult?.gesture;

  const isRecognized = latestResult?.gesture_recognized === true || recognitionStatus === 'recognized';

  useEffect(() => {
    if (sentenceOverride !== null && latestResult?.sentence === sentenceOverride) {
      setSentenceOverride(null);
    }
  }, [latestResult?.sentence, sentenceOverride]);

  const handleToggleCamera = async () => {
    setLoading(true);
    try {
      if (cameraRunning) {
        await stopCamera();
        setCameraRunning(false);
      } else {
        const response = await startCamera(0, 2);
        const status = await fetchCameraStatus();
        setCameraRunning(response.is_running && status.is_running);
      }
    } catch (err) {
      console.error('Camera toggle error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleTextAction = async (action) => {
    try {
      const response = await executeTextAction(action);
      if (response?.state && typeof response.state.sentence === 'string') {
        setSentenceOverride(response.state.sentence);
      }
    } catch (err) {
      console.error('Text action error:', err);
    }
  };

  const handleSpeak = async (lang = 'kn') => {
    const textToSpeak = sentence.trim() || gesture || (isRecognized ? kannadaPrediction : '');
    if (!textToSpeak) return;

    setTtsLoading(true);
    setActiveLang(lang);
    setAudioError(null);
    try {
      const res = await synthesizeTTS(textToSpeak, lang);
      if (res.audio_base64 && audioRef.current) {
        audioRef.current.src = res.audio_base64;
        audioRef.current.play();
      }
    } catch (err) {
      console.error('TTS error:', err);
      setAudioError('Text-to-speech synthesis failed. Ensure backend has internet for multi-lingual synthesis.');
    } finally {
      setTtsLoading(false);
    }
  };

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Hidden Audio Player for Speech */}
      <audio ref={audioRef} className="hidden" />

      {/* Main Grid: Camera Video & Live Prediction HUD */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Live Webcam Stream */}
        <div className="lg:col-span-8 space-y-4">
          <div className="relative rounded-2xl overflow-hidden bg-slate-900 border border-slate-800 shadow-2xl aspect-video flex items-center justify-center">
            {cameraRunning ? (
              <img
                src="/api/camera/stream"
                alt="Real-time Indian Sign Language Detection Stream"
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="text-center p-8 space-y-4">
                <div className="w-16 h-16 rounded-2xl bg-slate-800/80 border border-slate-700 flex items-center justify-center mx-auto text-slate-500">
                  <Camera className="w-8 h-8" />
                </div>
                <div>
                  <h3 className="text-lg font-bold text-white">Live Camera is Stopped</h3>
                  <p className="text-sm text-slate-400 mt-1 max-w-sm mx-auto">
                    Click "Start Camera" below to activate real-time YOLO hand detection and gesture recognition.
                  </p>
                </div>
                <button
                  onClick={handleToggleCamera}
                  disabled={loading}
                  className="inline-flex items-center space-x-2 px-6 py-3 rounded-xl bg-emerald-500 hover:bg-emerald-600 text-slate-950 font-bold shadow-lg shadow-emerald-500/20 transition-all"
                >
                  <Play className="w-5 h-5 fill-current" />
                  <span>Start Camera</span>
                </button>
              </div>
            )}

            {/* In-Video HUD Header Badge */}
            {cameraRunning && (
              <div className="absolute top-4 left-4 flex items-center space-x-2 bg-slate-950/80 backdrop-blur-md px-3 py-1.5 rounded-xl border border-slate-700/80">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
                <span className="text-xs font-semibold text-slate-200">Live Video (YOLO Active)</span>
              </div>
            )}

            {/* In-Video Performance Pill */}
            {cameraRunning && (
              <div className="absolute top-4 right-4 flex items-center space-x-3 bg-slate-950/80 backdrop-blur-md px-3 py-1.5 rounded-xl border border-slate-700/80 text-xs font-mono">
                <span className="text-cyan-400 font-bold">{fps} FPS</span>
                <span className="text-slate-400">|</span>
                <span className="text-slate-300">{latency} ms</span>
              </div>
            )}
          </div>

          {/* Camera Controls Bar */}
          <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center space-x-3">
              <button
                onClick={handleToggleCamera}
                disabled={loading}
                className={`inline-flex items-center space-x-2 px-5 py-2.5 rounded-xl font-bold text-sm transition-all shadow-md ${
                  cameraRunning
                    ? 'bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 border border-rose-500/40'
                    : 'bg-emerald-500 hover:bg-emerald-600 text-slate-950 shadow-emerald-500/20'
                }`}
              >
                {cameraRunning ? (
                  <>
                    <StopCircle className="w-4 h-4" />
                    <span>Stop Camera</span>
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 fill-current" />
                    <span>Start Camera</span>
                  </>
                )}
              </button>

              <span className="text-xs text-slate-400 hidden sm:inline">
                Detector: <span className="text-emerald-400 font-semibold">YOLOv8 Hand + MediaPipe Landmarks</span>
              </span>
            </div>

            <div className="text-xs text-slate-400 font-mono">
              Confidence Gate: <span className="text-white font-semibold">55%</span> | Smoothing: <span className="text-white font-semibold">5-Frame Majority</span>
            </div>
          </div>
        </div>

        {/* Right Column: Live Recognition Card & Top Probabilities */}
        <div className="lg:col-span-4 space-y-6">
          {/* Main Prediction Display Card */}
          <div className="p-6 rounded-2xl bg-gradient-to-b from-slate-900 to-slate-950 border border-slate-800 shadow-xl space-y-5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-400">ಕನ್ನಡ ಸೈನ್</span>
              <span className={`px-2 py-0.5 rounded-full text-[11px] font-semibold border ${
                isRecognized
                  ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
                  : prediction === 'Gesture Not Recognized'
                  ? 'bg-rose-500/15 text-rose-400 border-rose-500/30'
                  : 'bg-slate-800 text-slate-400 border-slate-700'
              }`}>
                {isRecognized ? 'ಖಚಿತ' : handDetected ? 'ಕೈ ಪತ್ತೆಯಾಗಿದೆ' : 'ಕೈ ಕಾಣುತ್ತಿಲ್ಲ'}
              </span>
            </div>

            {/* Big Sign Name */}
            <div className="text-center py-4 rounded-xl bg-slate-950/60 border border-slate-850">
              <span className={`text-5xl font-black tracking-tight ${
                isRecognized
                  ? 'text-emerald-400 drop-shadow-[0_0_20px_rgba(52,211,153,0.3)]'
                  : prediction === 'Gesture Not Recognized'
                  ? 'text-rose-400 text-2xl'
                  : 'text-slate-500 text-2xl'
              }`}>
                {isRecognized
                  ? (latestResult?.kannada_label || kannadaPrediction)
                  : handDetected
                    ? 'ಸೈನ್ ಗುರುತಿಸಲಾಗಿಲ್ಲ'
                    : 'ಕೈ ಕಾಣುತ್ತಿಲ್ಲ'}
              </span>
              {isRecognized && (
                <p className="text-sm text-slate-300 mt-2">{latestResult?.english_label || 'Indian Sign Language gesture'}</p>
              )}
              <p className="text-xs text-slate-400 mt-2">
                {handDetected
                  ? landmarksDetected
                    ? `${landmarkCount} landmark${landmarkCount === 1 ? '' : 's'} detected`
                    : recognitionStatus === 'gesture_model_not_trained'
                      ? 'Hand detected; keypoint/gesture model not trained'
                      : 'Hand detected; landmarks unavailable'
                  : 'No hand detected'}
              </p>
            </div>

            {/* Confidence Bar */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-slate-400">Confidence Score:</span>
                <span className="font-mono font-bold text-white">{(confidence * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full h-2.5 rounded-full bg-slate-800 overflow-hidden">
                <div
                  className={`h-full transition-all duration-200 rounded-full ${
                    confidence >= 0.75
                      ? 'bg-emerald-400'
                      : confidence >= 0.55
                      ? 'bg-amber-400'
                      : 'bg-rose-400'
                  }`}
                  style={{ width: `${Math.min(100, Math.max(0, confidence * 100))}%` }}
                />
              </div>
            </div>

            {/* Top 5 Probability Breakdown */}
            {Object.keys(probabilities).length > 0 && (
              <div className="space-y-2 pt-2 border-t border-slate-800/80">
                <span className="text-xs font-semibold text-slate-400">Top Predicted Classes:</span>
                <div className="space-y-1.5">
                  {Object.entries(probabilities).map(([cls, prob], idx) => (
                    <div key={idx} className="flex items-center justify-between text-xs">
                      <span className="font-bold text-slate-300">Sign '{cls}'</span>
                      <span className="font-mono text-slate-400">{(prob * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Quick Signs Reference */}
          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 text-xs space-y-2">
            <span className="font-semibold text-slate-300">5 ಕನ್ನಡ ಗೆಸ್ಚರ್‌ಗಳು:</span>
            <div className="text-slate-400">ನಮಸ್ಕಾರ · ಹೌದು · ಇಲ್ಲ · ಸರಿ · ಮುಗಿದಿದೆ</div>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-1 text-center">
              {['ನಮಸ್ಕಾರ', 'ಹೌದು', 'ಇಲ್ಲ', 'ಸರಿ', 'ಮುಗಿದಿದೆ'].map((label) => (
                <span
                  key={label}
                  className={`py-1 rounded font-bold transition-colors ${
                    isRecognized && latestResult?.kannada_label === label
                      ? 'bg-emerald-500 text-slate-950 shadow-md'
                      : 'bg-slate-800/80 text-slate-300'
                  }`}
                >
                  {label}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Text Conversion & Sentence Builder Bar */}
      <div className="p-6 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-xl space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center space-x-2">
            <Sparkles className="w-5 h-5 text-emerald-400" />
            <h2 className="text-base font-bold text-white">ಕನ್ನಡ ವಾಕ್ಯ ಮತ್ತು ಧ್ವನಿ</h2>
          </div>

          {/* TTS Actions */}
          <div className="flex items-center space-x-2">
            <button
              onClick={() => handleSpeak('kn')}
              disabled={ttsLoading || (!sentence && !gesture && !isRecognized)}
              className="inline-flex items-center space-x-1.5 px-3.5 py-1.5 rounded-lg bg-emerald-500/15 hover:bg-emerald-500/25 border border-emerald-500/30 text-emerald-400 text-xs font-semibold disabled:opacity-40 transition-all"
            >
              <Volume2 className="w-3.5 h-3.5" />
              <span>{ttsLoading && activeLang === 'kn' ? 'ಧ್ವನಿ ಸಿದ್ಧಪಡಿಸಲಾಗುತ್ತಿದೆ...' : 'ಕನ್ನಡದಲ್ಲಿ ಮಾತನಾಡಿ'}</span>
            </button>

            <button
              onClick={() => handleSpeak('kn')}
              disabled={ttsLoading || (!sentence && !gesture && !isRecognized)}
              className="inline-flex items-center space-x-1.5 px-3.5 py-1.5 rounded-lg bg-teal-500/15 hover:bg-teal-500/25 border border-teal-500/30 text-teal-300 text-xs font-semibold disabled:opacity-40 transition-all"
            >
              <Volume2 className="w-3.5 h-3.5" />
              <span>{ttsLoading && activeLang === 'kn' ? 'Synthesizing...' : 'Speak (Kannada)'}</span>
            </button>
          </div>
        </div>

        {/* Sentence Output Display */}
        <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 min-h-[64px] flex flex-col gap-3">
          <div className="w-full min-w-0 max-h-28 overflow-y-auto overflow-x-hidden pr-1">
            <span className="block text-lg font-mono font-medium text-white tracking-wide break-words whitespace-pre-wrap">
            {sentence || (
              <span className="text-slate-600 italic">Recognized characters will accumulate into words and sentences here...</span>
            )}
            </span>
          </div>

          <div className="flex w-full flex-wrap items-center justify-end gap-2">
            <button
              onClick={() => handleTextAction('space')}
              title="Add Space"
              className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium flex items-center space-x-1"
            >
              <Space className="w-3.5 h-3.5" />
              <span>Space</span>
            </button>
            <button
              onClick={() => handleTextAction('backspace')}
              title="Backspace"
              className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium flex items-center space-x-1"
            >
              <Delete className="w-3.5 h-3.5" />
              <span>Backspace</span>
            </button>
            <button
              onClick={() => handleTextAction('clear')}
              title="Clear All Text"
              className="px-3 py-1.5 rounded-lg bg-rose-500/15 hover:bg-rose-500/25 border border-rose-500/30 text-rose-400 text-xs font-medium flex items-center space-x-1"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Clear</span>
            </button>
          </div>
        </div>

        {audioError && (
          <p className="text-xs text-amber-400 flex items-center space-x-1">
            <AlertCircle className="w-3.5 h-3.5" />
            <span>{audioError}</span>
          </p>
        )}
      </div>
    </div>
  );
}
