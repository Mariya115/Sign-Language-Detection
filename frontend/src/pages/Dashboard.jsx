import React from 'react';
import { Camera, Sparkles, Layers, Volume2, ArrowRight, ShieldCheck, CheckCircle2, TrendingUp, Cpu } from 'lucide-react';

export default function Dashboard({ setActiveTab, cameraRunning, latestResult, history, modelInfo }) {
  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Hero Section */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-slate-900 via-slate-850 to-slate-900 border border-slate-800 p-8 sm:p-10 shadow-2xl">
        <div className="absolute top-0 right-0 -mt-12 -mr-12 w-96 h-96 bg-emerald-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="max-w-3xl relative z-10 space-y-4">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-semibold uppercase tracking-wider">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Final-Year B.Tech CSIT Major Research Project</span>
          </div>
          <h1 className="text-3xl sm:text-5xl font-extrabold text-white tracking-tight leading-tight">
            Real-Time <span className="bg-gradient-to-r from-emerald-400 via-teal-300 to-cyan-400 bg-clip-text text-transparent">Indian Sign Language</span> Recognition
          </h1>
          <p className="text-slate-300 text-base sm:text-lg leading-relaxed">
            An intelligent computer vision system combining <strong>Ultralytics YOLO</strong> for hand detection and deep learning classifiers for authentic ISL gesture recognition and multi-lingual sentence synthesis.
          </p>
          <div className="pt-4 flex flex-wrap gap-4">
            <button
              onClick={() => setActiveTab('recognition')}
              className="inline-flex items-center space-x-2.5 px-6 py-3 rounded-xl bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-600 hover:to-teal-600 text-slate-950 font-bold shadow-lg shadow-emerald-500/25 transition-all transform hover:-translate-y-0.5 active:translate-y-0"
            >
              <Camera className="w-5 h-5" />
              <span>Launch Live Recognition</span>
              <ArrowRight className="w-4 h-4 ml-1" />
            </button>
            <button
              onClick={() => setActiveTab('model')}
              className="inline-flex items-center space-x-2 px-5 py-3 rounded-xl bg-slate-800/80 hover:bg-slate-700/80 border border-slate-700 text-slate-200 font-semibold transition-all"
            >
              <Cpu className="w-4 h-4 text-emerald-400" />
              <span>Model Architecture & Metrics</span>
            </button>
          </div>
        </div>
      </div>

      {/* Key Architectural Highlights */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        <div className="p-5 rounded-xl bg-slate-900/60 border border-slate-800 hover:border-slate-700 transition-all">
          <div className="w-10 h-10 rounded-lg bg-emerald-500/10 flex items-center justify-center text-emerald-400 mb-3">
            <ShieldCheck className="w-5 h-5" />
          </div>
          <h3 className="font-bold text-white text-base">YOLO + Hand Landmarks</h3>
          <p className="text-xs text-slate-400 mt-1">
            Uses Ultralytics YOLOv8 for hand localization and MediaPipe Hands for real 21-point landmarks.
          </p>
        </div>

        <div className="p-5 rounded-xl bg-slate-900/60 border border-slate-800 hover:border-slate-700 transition-all">
          <div className="w-10 h-10 rounded-lg bg-cyan-500/10 flex items-center justify-center text-cyan-400 mb-3">
            <Layers className="w-5 h-5" />
          </div>
          <h3 className="font-bold text-white text-base">Authentic ISL Dataset</h3>
          <p className="text-xs text-slate-400 mt-1">
            Trained on 12,637 genuine Indian Sign Language samples across 26 classes (A-Z) with subject-independent 70-15-15 split.
          </p>
        </div>

        <div className="p-5 rounded-xl bg-slate-900/60 border border-slate-800 hover:border-slate-700 transition-all">
          <div className="w-10 h-10 rounded-lg bg-purple-500/10 flex items-center justify-center text-purple-400 mb-3">
            <TrendingUp className="w-5 h-5" />
          </div>
          <h3 className="font-bold text-white text-base">Flicker-Free Smoothing</h3>
          <p className="text-xs text-slate-400 mt-1">
            Temporal majority voting and confidence thresholding (55%) prevents prediction flicker and rejects unconfident noise.
          </p>
        </div>

        <div className="p-5 rounded-xl bg-slate-900/60 border border-slate-800 hover:border-slate-700 transition-all">
          <div className="w-10 h-10 rounded-lg bg-amber-500/10 flex items-center justify-center text-amber-400 mb-3">
            <Volume2 className="w-5 h-5" />
          </div>
          <h3 className="font-bold text-white text-base">English & Kannada TTS</h3>
          <p className="text-xs text-slate-400 mt-1">
            Converts recognized signs and constructed sentences into audible speech with multi-lingual voice synthesis.
          </p>
        </div>
      </div>

      {/* Live Status & Quick Feed */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Real-time Status Card */}
        <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-bold text-white">Live Engine Status</h2>
            <span className={`px-2.5 py-0.5 rounded-full text-xs font-semibold ${
              cameraRunning ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' : 'bg-slate-800 text-slate-400'
            }`}>
              {cameraRunning ? 'Active Stream' : 'Camera Idle'}
            </span>
          </div>

          <div className="space-y-3 pt-2">
            <div className="flex justify-between items-center text-sm py-2 border-b border-slate-800/80">
              <span className="text-slate-400">ಪ್ರಸ್ತುತ ಕನ್ನಡ ಸೈನ್:</span>
              <span className="font-bold text-emerald-400 text-base">{latestResult?.gesture || latestResult?.kannada_prediction || 'ಯಾವುದೂ ಇಲ್ಲ'}</span>
            </div>
            <div className="flex justify-between items-center text-sm py-2 border-b border-slate-800/80">
              <span className="text-slate-400">Confidence Score:</span>
              <span className="font-mono font-semibold text-white">
                {latestResult?.confidence ? `${(latestResult.confidence * 100).toFixed(1)}%` : '0.0%'}
              </span>
            </div>
            <div className="flex justify-between items-center text-sm py-2 border-b border-slate-800/80">
              <span className="text-slate-400">Inference Throughput:</span>
              <span className="font-mono text-cyan-400 font-semibold">{latestResult?.fps || 0} FPS</span>
            </div>
            <div className="flex justify-between items-center text-sm py-2">
              <span className="text-slate-400">Hands In View:</span>
              <span className="font-semibold text-slate-200">{latestResult?.num_hands || 0}</span>
            </div>
          </div>

          <button
            onClick={() => setActiveTab('recognition')}
            className="w-full py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700/80 text-sm font-semibold text-slate-200 transition-colors"
          >
            Open Live Camera View
          </button>
        </div>

        {/* Recent Detections Preview */}
        <div className="lg:col-span-2 p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-bold text-white">Recent Recognition History</h2>
            <button
              onClick={() => setActiveTab('history')}
              className="text-xs font-semibold text-emerald-400 hover:text-emerald-300"
            >
              View All History →
            </button>
          </div>

          {history && history.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400 text-xs uppercase">
                    <th className="py-2.5 px-3">Sign</th>
                    <th className="py-2.5 px-3">Confidence</th>
                    <th className="py-2.5 px-3">Hands</th>
                    <th className="py-2.5 px-3">Timestamp</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-850">
                  {history.slice(0, 5).map((item, idx) => (
                    <tr key={idx} className="hover:bg-slate-850/50">
                      <td className="py-2.5 px-3 font-bold text-emerald-400">{item.sign}</td>
                      <td className="py-2.5 px-3 font-mono text-slate-300">{(item.confidence * 100).toFixed(1)}%</td>
                      <td className="py-2.5 px-3 text-slate-400">{item.num_hands || 1}</td>
                      <td className="py-2.5 px-3 text-xs text-slate-400 font-mono">{item.timestamp}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="py-12 text-center text-slate-500 text-sm">
              No signs recognized yet. Launch the live recognition page and begin signing in front of the webcam.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
