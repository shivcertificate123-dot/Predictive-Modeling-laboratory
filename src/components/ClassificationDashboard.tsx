import { ClassificationMetrics } from '../types';
import { Target, Maximize, AlertCircle, Sparkles } from 'lucide-react';

interface ClassificationDashboardProps {
  metrics: ClassificationMetrics;
  threshold: number;
  setThreshold: (t: number) => void;
  testSamplesCount: number;
}

export default function ClassificationDashboard({
  metrics,
  threshold,
  setThreshold,
  testSamplesCount
}: ClassificationDashboardProps) {

  const cm = metrics.confusionMatrix;
  const totalCorrect = cm.tp + cm.tn;
  const totalIncorrect = cm.fp + cm.fn;
  const correctPct = testSamplesCount > 0 ? (totalCorrect / testSamplesCount) * 100 : 0;

  // Let's find the point on the ROC curve nearest to our current threshold
  const nearestPoint = metrics.rocCurve.reduce((prev, curr) => {
    return Math.abs(curr.threshold - threshold) < Math.abs(prev.threshold - threshold) ? curr : prev;
  }, metrics.rocCurve[0] || { fpr: 0, tpr: 0, threshold: 0.5 });

  return (
    <div className="space-y-6">
      {/* Classification Metrics Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Accuracy */}
        <div className="bg-white p-4 rounded-xl border border-slate-100 shadow-xs relative overflow-hidden">
          <div className="text-[11px] font-bold uppercase tracking-wider text-slate-400 flex justify-between items-center">
            Accuracy
            <span className="w-2 h-2 rounded-full bg-indigo-500" />
          </div>
          <div className="text-2xl font-black font-mono text-slate-800 mt-1">
            {(metrics.accuracy * 100).toFixed(1)}%
          </div>
          <p className="text-[10px] text-slate-500 mt-1">
            {metrics.accuracy > 0.85 ? 'Highly accurate predictions' : 'Moderate diagnostic fidelity'}
          </p>
        </div>

        {/* Precision */}
        <div className="bg-white p-4 rounded-xl border border-slate-100 shadow-xs">
          <div className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
            Precision (PV+)
          </div>
          <div className="text-2xl font-black font-mono text-slate-800 mt-1">
            {(metrics.precision * 100).toFixed(1)}%
          </div>
          <p className="text-[10px] text-slate-400 mt-1">TP / (TP + FP) - Specificity of hits</p>
        </div>

        {/* Recall */}
        <div className="bg-white p-4 rounded-xl border border-slate-100 shadow-xs">
          <div className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
            Recall (Sensitivity)
          </div>
          <div className="text-2xl font-black font-mono text-slate-800 mt-1">
            {(metrics.recall * 100).toFixed(1)}%
          </div>
          <p className="text-[10px] text-slate-400 mt-1">TP / (TP + FN) - Retention rate</p>
        </div>

        {/* F1 Score */}
        <div className="bg-white p-4 rounded-xl border border-slate-100 shadow-xs">
          <div className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
            F1-Harmonic Score
          </div>
          <div className="text-2xl font-black font-mono text-slate-800 mt-1">
            {metrics.f1.toFixed(3)}
          </div>
          <p className="text-[10px] text-slate-400 mt-1">Balanced precision & recall index</p>
        </div>
      </div>

      {/* Threshold Selector with Interactive Info Box */}
      <div className="bg-slate-50 border border-slate-100 rounded-2xl p-5 space-y-4">
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2">
          <div>
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700 flex items-center gap-1.5">
              <Sparkles className="w-4 h-4 text-amber-500 animate-pulse" />
              Tune Binary Decision Threshold
            </h3>
            <p className="text-xs text-slate-500">Slide to alter positive prediction strictness (affects Confusion Matrix & ROC in real-time)</p>
          </div>
          <span className="font-mono text-xs font-bold text-slate-700 bg-white border border-slate-200 px-3 py-1 rounded-lg">
            Active Threshold: <span className="text-indigo-600 text-sm font-black">{threshold.toFixed(2)}</span>
          </span>
        </div>

        <input
          type="range"
          id="threshold-slider"
          min="0.05"
          max="0.95"
          step="0.05"
          value={threshold}
          onChange={e => setThreshold(parseFloat(e.target.value))}
          className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
        />

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-1">
          <div className="bg-white p-3 rounded-lg border border-slate-100 text-xs leading-relaxed text-slate-600">
            <span className="font-semibold text-slate-800">Low Threshold (e.g. &lt; 0.3):</span> Easy to flag positive classes. High recall (retains positive cases) but may trigger more False Alarms (losses in precision).
          </div>
          <div className="bg-white p-3 rounded-lg border border-slate-100 text-xs leading-relaxed text-slate-600">
            <span className="font-semibold text-slate-800">High Threshold (e.g. &gt; 0.7):</span> Conservative prediction. Excellent precision (highly trusted positive claims) but risks missing subtle cases (low recall).
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Confusion Matrix */}
        <div className="bg-white p-5 rounded-2xl border border-slate-100 shadow-xs space-y-4">
          <div>
            <h3 className="text-sm font-semibold text-slate-700 flex items-center gap-1.5">
              <Target className="w-4 h-4 text-indigo-500" />
              Confusion Matrix
            </h3>
            <p className="text-[11px] text-slate-400">Analysis of true condition labels vs predictions</p>
          </div>

          <div id="confusion-matrix-display" className="grid grid-cols-12 gap-1.5 max-w-sm mx-auto text-center text-xs font-semibold">
            {/* Corner header */}
            <div className="col-span-4" />
            <div className="col-span-8 text-center text-[10px] text-slate-400 uppercase tracking-widest font-bold pb-2">PREDICTED CONDITION</div>

            <div className="col-span-4" />
            <div className="col-span-4 text-slate-500 py-1 bg-slate-50 border border-slate-100 rounded-md">Positive (1)</div>
            <div className="col-span-4 text-slate-500 py-1 bg-slate-50 border border-slate-100 rounded-md">Negative (0)</div>

            <div className="col-span-4 flex items-center justify-center p-2 bg-slate-50 border border-slate-100 rounded-md text-[10px] text-slate-400 uppercase tracking-widest writing-vertical select-none font-bold">
              TRUE EVENT
            </div>

            {/* Matrix Cells */}
            {/* TP */}
            <div className="col-span-4 bg-emerald-50 border border-emerald-100/60 rounded-xl p-4 flex flex-col justify-center items-center transition-all hover:ring-2 hover:ring-emerald-300">
              <span className="text-2xl font-black font-mono text-emerald-800">{cm.tp}</span>
              <span className="text-[9px] text-emerald-600 uppercase mt-0.5 font-bold">True Pos</span>
              <span className="text-[8px] text-emerald-500 font-mono">{(testSamplesCount > 0 ? (cm.tp / testSamplesCount) * 100 : 0).toFixed(0)}% of total</span>
            </div>

            {/* FN */}
            <div className="col-span-4 bg-red-50 border border-red-100 rounded-xl p-4 flex flex-col justify-center items-center transition-all hover:ring-2 hover:ring-red-300">
              <span className="text-2xl font-black font-mono text-red-800">{cm.fn}</span>
              <span className="text-[9px] text-red-600 uppercase mt-0.5 font-bold">False Neg</span>
              <span className="text-[8px] text-red-500 font-mono">{(testSamplesCount > 0 ? (cm.fn / testSamplesCount) * 100 : 0).toFixed(0)}% of total</span>
            </div>

            {/* Empty space left side offset */}
            <div className="col-span-4" />

            {/* FP */}
            <div className="col-span-4 bg-amber-50 border border-amber-100 rounded-xl p-4 flex flex-col justify-center items-center transition-all hover:ring-2 hover:ring-amber-300">
              <span className="text-2xl font-black font-mono text-amber-800">{cm.fp}</span>
              <span className="text-[9px] text-amber-600 uppercase mt-0.5 font-bold">False Pos</span>
              <span className="text-[8px] text-amber-500 font-mono">{(testSamplesCount > 0 ? (cm.fp / testSamplesCount) * 100 : 0).toFixed(0)}% of total</span>
            </div>

            {/* TN */}
            <div className="col-span-4 bg-emerald-50 border border-emerald-100/60 rounded-xl p-4 flex flex-col justify-center items-center transition-all hover:ring-2 hover:ring-emerald-300">
              <span className="text-2xl font-black font-mono text-emerald-800">{cm.tn}</span>
              <span className="text-[9px] text-emerald-600 uppercase mt-0.5 font-bold">True Neg</span>
              <span className="text-[8px] text-emerald-500 font-mono">{(testSamplesCount > 0 ? (cm.tn / testSamplesCount) * 100 : 0).toFixed(0)}% of total</span>
            </div>
          </div>

          <div className="bg-slate-50 p-3 rounded-lg text-[11px] text-slate-500 font-mono flex items-center gap-1.5 justify-center">
            <AlertCircle className="w-4 h-4 text-slate-400" />
            <span>Success Rate: <strong className="text-emerald-700">{correctPct.toFixed(1)}% ({totalCorrect} correct)</strong> • Errors: <strong className="text-red-700">{totalIncorrect} ({(100 - correctPct).toFixed(1)}%)</strong></span>
          </div>
        </div>

        {/* ROC Curve Chart */}
        <div id="roc-curve-container" className="bg-white p-5 rounded-2xl border border-slate-100 shadow-xs space-y-4">
          <div className="flex justify-between items-start">
            <div>
              <h3 className="text-sm font-semibold text-slate-700 flex items-center gap-1.5">
                <Maximize className="w-4 h-4 text-indigo-500" />
                ROC Performance Curve
              </h3>
              <p className="text-[11px] text-slate-400">Receiver Operating Characteristic / Sensitivity trade-off</p>
            </div>
            <div className="text-right">
              <span className="text-[10px] text-slate-400 font-bold block uppercase">AUC Area</span>
              <span className="text-lg font-black text-indigo-600 font-mono">{metrics.auc.toFixed(3)}</span>
            </div>
          </div>

          <div className="relative w-full h-[220px] bg-slate-50/50 rounded-xl border border-slate-100 p-4 font-mono text-[9px] text-slate-400">
            <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
              {/* Shaded Area Under ROC Curve */}
              {metrics.rocCurve.length > 0 && (
                <path
                  d={`M 0 100 ` + metrics.rocCurve.map(pt => `L ${pt.fpr * 100} ${100 - pt.tpr * 100}`).join(' ') + ` L 100 100 Z`}
                  className="fill-indigo-500/10"
                />
              )}

              {/* Diagonal Line (Random guessing, AUC = 0.5) */}
              <line x1="0" y1="100" x2="100" y2="0" stroke="#cbd5e1" strokeWidth="0.75" strokeDasharray="3" />

              {/* ROC Curve Path */}
              {metrics.rocCurve.length > 0 && (
                <path
                  d={metrics.rocCurve.map((pt, i) => `${i === 0 ? 'M' : 'L'} ${pt.fpr * 100} ${100 - pt.tpr * 100}`).join(' ')}
                  fill="none"
                  stroke="#4f46e5"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              )}

              {/* Current Threshold Position on Curve */}
              <circle
                cx={nearestPoint.fpr * 100}
                cy={100 - nearestPoint.tpr * 100}
                r="3.5"
                className="fill-indigo-600 stroke-white stroke-1 animate-pulse"
              />
            </svg>

            {/* Curve Legend */}
            <div className="absolute top-2 left-2 flex flex-col gap-1 text-[9px]">
              <div className="flex items-center gap-1">
                <span className="w-1.5 h-1.5 bg-indigo-600 rounded-full" />
                <span>ROC Path (AUC = {metrics.auc.toFixed(3)})</span>
              </div>
              <div className="flex items-center gap-1 text-slate-500">
                <span className="w-1.5 h-1.5 bg-indigo-600 rounded-full border border-white animate-pulse" />
                <span>Current Operating Point (FPR: {nearestPoint.fpr.toFixed(2)}, TPR: {nearestPoint.tpr.toFixed(2)})</span>
              </div>
            </div>

            {/* Labels */}
            <div className="absolute bottom-1 right-2 text-slate-400">False Pos Rate (1-Spec)</div>
            <div className="absolute top-1 right-2 transform rotate-90 origin-top-right translate-y-12 text-slate-400">True Pos Rate (Recall)</div>
          </div>
        </div>
      </div>
    </div>
  );
}
