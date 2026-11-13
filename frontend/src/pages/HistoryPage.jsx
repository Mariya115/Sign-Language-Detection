import React, { useState } from 'react';
import { History, Download, Trash2, Search, Filter } from 'lucide-react';

export default function HistoryPage({ history, setHistory }) {
  const [searchTerm, setSearchTerm] = useState('');

  const filtered = history.filter((item) =>
    item.sign.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleClear = () => {
    setHistory([]);
  };

  const handleExportJSON = () => {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(history, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', dataStr);
    downloadAnchor.setAttribute('download', `isl_recognition_history_${Date.now()}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  const handleExportCSV = () => {
    if (!history.length) return;
    const headers = ['ID', 'Sign', 'Confidence', 'Hands', 'Timestamp'];
    const rows = history.map((item) => [
      item.id,
      item.sign,
      (item.confidence * 100).toFixed(1) + '%',
      item.num_hands || 1,
      item.timestamp,
    ]);
    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map((e) => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `isl_recognition_history_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    link.remove();
  };

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header & Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight">Recognition History</h1>
          <p className="text-slate-400 text-sm mt-1">
            Logged timeline of all confident gestures recognized during active sessions.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={handleExportCSV}
            disabled={!history.length}
            className="inline-flex items-center space-x-1.5 px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold disabled:opacity-40 transition-all border border-slate-700"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Export CSV</span>
          </button>
          <button
            onClick={handleExportJSON}
            disabled={!history.length}
            className="inline-flex items-center space-x-1.5 px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold disabled:opacity-40 transition-all border border-slate-700"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Export JSON</span>
          </button>
          <button
            onClick={handleClear}
            disabled={!history.length}
            className="inline-flex items-center space-x-1.5 px-3.5 py-2 rounded-xl bg-rose-500/15 hover:bg-rose-500/25 border border-rose-500/30 text-rose-400 text-xs font-semibold disabled:opacity-40 transition-all"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Clear History</span>
          </button>
        </div>
      </div>

      {/* Search Input Bar */}
      <div className="relative">
        <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
        <input
          type="text"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          placeholder="Filter recognized signs (e.g., A, B, NAMASTE)..."
          className="w-full pl-10 pr-4 py-2.5 rounded-xl bg-slate-900 border border-slate-800 text-white placeholder-slate-500 text-sm focus:outline-none focus:border-emerald-500 transition-colors"
        />
      </div>

      {/* History Table */}
      <div className="rounded-2xl bg-slate-900/80 border border-slate-800 overflow-hidden shadow-xl">
        {filtered.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 text-xs uppercase bg-slate-950/60">
                  <th className="py-3 px-4"># ID</th>
                  <th className="py-3 px-4">Recognized Sign</th>
                  <th className="py-3 px-4">Confidence Score</th>
                  <th className="py-3 px-4">Detected Hands</th>
                  <th className="py-3 px-4">Timestamp</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-850">
                {filtered.map((item, index) => (
                  <tr key={index} className="hover:bg-slate-850/50 transition-colors">
                    <td className="py-3 px-4 font-mono text-slate-500 text-xs">#{item.id || index + 1}</td>
                    <td className="py-3 px-4 font-bold text-emerald-400 text-base">{item.sign}</td>
                    <td className="py-3 px-4 font-mono text-white">{(item.confidence * 100).toFixed(1)}%</td>
                    <td className="py-3 px-4 text-slate-300">{item.num_hands || 1}</td>
                    <td className="py-3 px-4 font-mono text-slate-400 text-xs">{item.timestamp}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="py-16 text-center text-slate-500 text-sm space-y-2">
            <History className="w-10 h-10 mx-auto text-slate-600" />
            <p>No recognition history records found.</p>
          </div>
        )}
      </div>
    </div>
  );
}

