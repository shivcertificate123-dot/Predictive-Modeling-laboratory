import { RegressionMetrics, TrainingProgress, TrainedModel } from '../types';
import { ArrowUpRight, TrendingUp, Compass, Settings } from 'lucide-react';

interface RegressionDashboardProps {
  metrics: RegressionMetrics | null;
  history: TrainingProgress[];
  model: TrainedModel | null;
  testSamplesCount: number;
  actualVsPred: Array<{ actual: number; pred: number }>;
}

export default function RegressionDashboard({
  metrics,
  history,
  model,
  testSamplesCount,
  actualVsPred
}: RegressionDashboardProps) {

  if (!metrics || !model) {
    return (
      <div className="bg-slate-50 border border-dashed border-slate-200 rounded-2xl p-8 text-center text-slate-400">
        Click "Train Model" to run linear regression training on your split dataset and visualize fit diagnostics.
      </div>
    );
  }

  // Find min/max for scale scaling of our scatter plot
  const allY = [...actualVsPred.map(d => d.actual), ...actualVsPred.map(d => d.pred)];
  const minY = Math.min(...allY, 0) * 0.95;
  const maxY = Math.max(...allY, 1000) * 1.05;
  const rangeY = maxY - minY || 1;

  // Render loss curve variables
  const lossValues = history.map(h => h.loss);
  const minLoss = Math.min(...lossValues);
  const maxLoss = Math.max(...lossValues);
  const lossRange = maxLoss - minLoss || 1;

  return (
    <div className="space-y-6">
      {/* Metrics Row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {/* R^2 Card */}
        <div className="bg-white p-4 rounded-xl border border-slate-100 shadow-xs">
          <div className="text-[11px] font-bold uppercase tracking-wider text-slate-400 flex items-center justify-between">
            R-Squared (R²)
            <span className="w-2 h-2 rounded-full bg-emerald-500" />
          </div>
          <div className="text-2xl font-black font-mono text-slate-800 mt-1">
            {metrics.r2.toFixed(4)}
          </div>
          <p className="text-[10px] text-slate-500 mt-1">
            {metrics.r2 > 0.8
              ? 'Excellent predictive fit'
              : metrics.r2 > 0.5
              ? 'Moderate correlation fit'
              : 'Weak regression alignment'}
          </p>
        </div>

        {/* MSE Card */}
        <div className="bg-white p-4 rounded-xl border border-slate-100 shadow-xs">
          <div className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
            Mean Squared Error (MSE)
          </div>
          <div className="text-2xl font-black font-mono text-slate-800 mt-1">
            {metrics.mse.toLocaleString(undefined, { maximumFractionDigits: 0 })}
          </div>
          <p className="text-[10px] text-slate-500 mt-1">Average squared prediction errors</p>
        </div>

        {/* RMSE Card */}
        <div className="bg-white p-4 rounded-xl border border-slate-100 shadow-xs">
          <div className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
            Root MSE (RMSE)
          </div>
          <div className="text-2xl font-black font-mono text-slate-800 mt-1">
            {metrics.rmse.toLocaleString(undefined, { maximumFractionDigits: 1 })}
          </div>
          <p className="text-[10px] text-slate-500 mt-1">Average error in raw target units</p>
        </div>

        {/* MAE Card */}
        <div className="bg-white p-4 rounded-xl border border-slate-100 shadow-xs">
          <div className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
            Mean Abs Error (MAE)
          </div>
          <div className="text-2xl font-black font-mono text-slate-800 mt-1">
            {metrics.mae.toLocaleString(undefined, { maximumFractionDigits: 1 })}
          </div>
          <p className="text-[10px] text-slate-500 mt-1">Average magnitude of raw absolute errors</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Scatter Plot */}
        <div className="bg-white p-5 rounded-2xl border border-slate-100 shadow-xs space-y-3">
          <div>
            <h3 className="text-sm font-semibold text-slate-700 flex items-center gap-1.5">
              <TrendingUp className="w-4 h-4 text-emerald-500" />
              Actual vs. Predicted Fit (Test Set: {testSamplesCount} samples)
            </h3>
            <p className="text-[11px] text-slate-400">Points close to the diagonal line indicate perfect forecast matches</p>
          </div>

          <div className="relative w-full h-[220px] bg-slate-50/50 rounded-xl border border-slate-100 p-4 font-mono text-[9px] text-slate-400">
            <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
              {/* Grid Lines */}
              <line x1="0" y1="25" x2="100" y2="25" stroke="#f1f5f9" strokeWidth="0.5" />
              <line x1="0" y1="50" x2="100" y2="50" stroke="#f1f5f9" strokeWidth="0.5" />
              <line x1="0" y1="75" x2="100" y2="75" stroke="#f1f5f9" strokeWidth="0.5" />

              <line x1="25" y1="0" x2="25" y2="100" stroke="#f1f5f9" strokeWidth="0.5" />
              <line x1="50" y1="0" x2="50" y2="100" stroke="#f1f5f9" strokeWidth="0.5" />
              <line x1="75" y1="0" x2="75" y2="100" stroke="#f1f5f9" strokeWidth="0.5" />

              {/* Diagonal Line (Perfect Fit Line) */}
              <line
                x1={((minY - minY) / rangeY) * 100}
                y1={100 - ((minY - minY) / rangeY) * 100}
                x2={((maxY - minY) / rangeY) * 100}
                y2={100 - ((maxY - minY) / rangeY) * 100}
                stroke="#10b981"
                strokeWidth="1"
                strokeDasharray="2"
              />

              {/* Scatter Points */}
              {actualVsPred.map((pt, i) => {
                const cx = ((pt.actual - minY) / rangeY) * 100;
                // inverted y
                const cy = 100 - ((pt.pred - minY) / rangeY) * 100;

                return (
                  <circle
                    key={i}
                    cx={cx}
                    cy={cy}
                    r="2"
                    className="fill-indigo-500 stroke-white stroke-[0.5] hover:fill-indigo-600 hover:r-3 cursor-pointer transition-all"
                  >
                    <title>Actual: {pt.actual.toFixed(0)}, Predicted: {pt.pred.toFixed(0)}</title>
                  </circle>
                );
              })}
            </svg>

            {/* Y label */}
            <div className="absolute top-2 left-2 flex items-center gap-1">
              <span className="w-1.5 h-1.5 bg-indigo-500 rounded-full" />
              <span>Predicted Value</span>
            </div>
            {/* X label */}
            <div className="absolute bottom-2 right-2 flex items-center gap-1">
              <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full" />
              <span>Actual Value</span>
            </div>
          </div>
        </div>

        {/* Gradient Descent Loss History */}
        <div className="bg-white p-5 rounded-2xl border border-slate-100 shadow-xs space-y-3">
          <div>
            <h3 className="text-sm font-semibold text-slate-700 flex items-center gap-1.5">
              <Compass className="w-4 h-4 text-indigo-500" />
              Gradient Descent Convergence (Loss History)
            </h3>
            <p className="text-[11px] text-slate-400">Shows minimization of Mean Squared Error over {history.length} epochs</p>
          </div>

          <div className="relative w-full h-[220px] bg-slate-50/50 rounded-xl border border-slate-100 p-4 text-[9px] text-slate-400">
            <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
              {/* Background grid */}
              <line x1="0" y1="33" x2="100" y2="33" stroke="#f1f5f9" strokeWidth="0.5" />
              <line x1="0" y1="66" x2="100" y2="66" stroke="#f1f5f9" strokeWidth="0.5" />

              {/* Training Loss Path */}
              {history.length > 1 && (
                <path
                  d={history
                    .map((pt, idx) => {
                      const x = (idx / (history.length - 1)) * 100;
                      const y = 100 - ((pt.loss - minLoss) / lossRange) * 100;
                      return `${idx === 0 ? 'M' : 'L'} ${x} ${y}`;
                    })
                    .join(' ')}
                  fill="none"
                  stroke="#4f46e5"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              )}
            </svg>

            <div className="absolute top-2 left-2 flex flex-col">
              <span className="text-slate-500 font-bold font-mono">Max MSE: {maxLoss.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
              <span className="text-indigo-600 font-bold font-mono mt-0.5">Final MSE: {minLoss.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
            </div>
            <div className="absolute bottom-2 right-2 text-slate-500 text-[9px] font-mono">
              Epochs: 1 → {history.length}
            </div>
          </div>
        </div>
      </div>

      {/* Model Parameter Inspection (Coefficients/Weights & Intercept) */}
      <div className="bg-slate-50 border border-slate-100 rounded-xl p-5 space-y-3">
        <h3 className="text-xs font-bold uppercase tracking-widest text-slate-500 flex items-center gap-1">
          <Settings className="w-3.5 h-3.5" />
          Model Weights Interpretation (Linear Coefficients)
        </h3>
        <p className="text-xs text-slate-500 leading-relaxed">
          The fitted function is: <code className="font-mono bg-white px-1.5 py-0.5 rounded-sm border border-slate-200">
            y = {model.intercept?.toFixed(4)}
            {Object.entries(model.weights || {}).map(([key, val]) => ` + (${val.toFixed(4)}) * scaled(${key})`).join('')}
          </code>
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-2">
          {Object.entries(model.weights || {}).map(([featureKey, val]) => {
            const pctImpact = Math.min(100, Math.max(5, (Math.abs(val) / (Object.values(model.weights || {}).reduce((sum, w) => sum + Math.abs(w), 0) || 1)) * 100));
            return (
              <div key={featureKey} className="bg-white p-3 rounded-lg border border-slate-200/60 shadow-xxs">
                <div className="flex justify-between items-center">
                  <span className="font-semibold text-xs text-slate-700">{featureKey}</span>
                  <span className={`text-[10px] font-mono font-bold px-1 py-0.5 rounded flex items-center ${val >= 0 ? 'bg-emerald-50 text-emerald-700' : 'bg-red-50 text-red-700'}`}>
                    {val >= 0 ? '+' : ''}{val.toFixed(4)}
                    <ArrowUpRight className={`w-3 h-3 ml-0.5 ${val < 0 ? 'rotate-90' : ''}`} />
                  </span>
                </div>
                <div className="w-full bg-slate-100 h-1.5 rounded-full mt-2 overflow-hidden">
                  <div
                    style={{ width: `${pctImpact}%` }}
                    className={`h-full rounded-full ${val >= 0 ? 'bg-indigo-500' : 'bg-red-500'}`}
                  />
                </div>
                <div className="text-[9px] text-slate-400 mt-1 text-right">
                  Relative feature weight impact: {Math.round(pctImpact)}%
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
