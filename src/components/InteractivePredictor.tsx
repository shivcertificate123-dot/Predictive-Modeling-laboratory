import React, { useState, useEffect } from 'react';
import { DATA_SCENARIOS, predictLinear, predictLogisticProb, predictDecisionTreeProb } from '../ml';
import { DatasetScenario, TrainedModel, Scaler } from '../types';
import { Play, Sparkles, TrendingUp, HelpCircle } from 'lucide-react';

interface InteractivePredictorProps {
  scenario: DatasetScenario;
  model: TrainedModel | null;
  scaler: Scaler | null;
  targetScaler: { min: number; max: number } | null;
  classificationThreshold: number;
}

export default function InteractivePredictor({
  scenario,
  model,
  scaler,
  targetScaler,
  classificationThreshold
}: InteractivePredictorProps) {
  const activeSpec = DATA_SCENARIOS[scenario];
  const features = activeSpec.features;

  // Initialize input values state with the midpoint of each feature
  const [inputs, setInputs] = useState<Record<string, number>>({});

  useEffect(() => {
    const initialInputs: Record<string, number> = {};
    features.forEach(f => {
      initialInputs[f.key] = Math.round((f.min + f.max) / 2);
    });
    setInputs(initialInputs);
  }, [scenario, features]);

  if (!model) {
    return (
      <div className="bg-slate-50 border border-dashed border-slate-200 rounded-2xl p-6 text-center text-slate-400 text-xs text-slate-500">
        Active interactive inference playground displays once a model is trained. Click "Train Model" above to start.
      </div>
    );
  }

  const handleInputChange = (key: string, val: number) => {
    setInputs(prev => ({
      ...prev,
      [key]: val
    }));
  };

  // Run predictions based on active inputs
  let prediction: number | null = null;
  let isClassification = activeSpec.targetType === 'classification';

  if (model.type === 'linear' && scaler && targetScaler) {
    prediction = predictLinear(inputs, model, scaler, targetScaler);
  } else if (model.type === 'logistic' && scaler) {
    prediction = predictLogisticProb(inputs, model, scaler);
  } else if (model.type === 'tree') {
    prediction = predictDecisionTreeProb(inputs, model.tree);
  }

  return (
    <div className="bg-white rounded-2xl border border-slate-100 shadow-xs p-6 space-y-6">
      <div className="flex items-center space-x-3">
        <div className="w-10 h-10 rounded-xl bg-purple-50 flex items-center justify-center text-purple-600">
          <Play className="w-5 h-5 fill-purple-600 text-purple-600" />
        </div>
        <div>
          <h2 className="text-lg font-semibold text-slate-800">3. Interactive Single-Sample Predictor</h2>
          <p className="text-xs text-slate-500">Test how the trained model forecasts individual customized situations</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-1">
        {/* Sliders Input */}
        <div className="space-y-4">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">Model Input Features</h3>
          {features.map(f => {
            const val = inputs[f.key] ?? (f.min + f.max) / 2;
            return (
              <div key={f.key} className="space-y-1.5 p-3.5 bg-slate-50/50 rounded-xl border border-slate-100">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-semibold text-slate-700">{f.name}</span>
                  <span className="font-mono text-xs font-bold text-slate-800 bg-white border border-slate-100 px-2 py-0.5 rounded-md">
                    {val.toLocaleString()} {f.unit || ''}
                  </span>
                </div>
                <input
                  type="range"
                  min={f.min}
                  max={f.max}
                  step={f.type === 'binary' ? 1 : Math.round((f.max - f.min) / 40) || 1}
                  value={val}
                  onChange={e => handleInputChange(f.key, parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                />
                <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                  <span>Min: {f.min}</span>
                  <span>Max: {f.max}</span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Prediction Outputs */}
        <div className="bg-slate-900 text-white rounded-2xl p-6 flex flex-col justify-between relative overflow-hidden shadow-md">
          {/* Subtle background circles for premium aesthetic */}
          <div className="absolute top-0 right-0 w-44 h-44 bg-purple-500/10 rounded-full blur-2xl pointer-events-none" />
          <div className="absolute bottom-0 left-0 w-32 h-32 bg-indigo-500/10 rounded-full blur-xl pointer-events-none" />

          <div className="z-10 space-y-2">
            <span className="inline-flex items-center gap-1 text-[10px] font-bold text-purple-300 uppercase bg-purple-500/20 px-2 py-0.5 rounded-full tracking-widest leading-none">
              <Sparkles className="w-3 h-3" />
              Live Prediction Output
            </span>
            <p className="text-xs text-slate-300">Outputs are generated by applying the real-time calculated weights or splitting structures.</p>
          </div>

          <div className="z-10 py-6 text-center">
            {prediction !== null ? (
              isClassification ? (
                // Classification Prediction Output UI (Probability + outcome)
                <div className="space-y-4">
                  <div className="text-xs text-slate-400 uppercase tracking-widest font-semibold">Predicted Positive Probability</div>
                  <div className="text-4xl font-extrabold font-mono text-purple-400 flex justify-center items-baseline select-all">
                    {(prediction * 100).toFixed(1)}%
                    <span className="text-xs text-slate-400 font-normal ml-0.5">chance</span>
                  </div>

                  {/* Probability track bar */}
                  <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden max-w-xs mx-auto">
                    <div
                      style={{ width: `${prediction * 100}%` }}
                      className={`h-full rounded-full transition-all ${
                        prediction >= classificationThreshold ? 'bg-red-500' : 'bg-emerald-500'
                      }`}
                    />
                  </div>

                  {/* Decision Tag */}
                  <div className="inline-flex items-center gap-1.5 pt-2">
                    <span className="text-xs text-slate-300">Decision Outcome:</span>
                    <strong
                      className={`text-xs px-2.5 py-1 rounded-full font-bold uppercase tracking-wider ${
                        prediction >= classificationThreshold
                          ? 'bg-red-500/20 text-red-300 border border-red-500/30'
                          : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                      }`}
                    >
                      {prediction >= classificationThreshold ? 'Class 1 (Flagged)' : 'Class 0 (Normal)'}
                    </strong>
                  </div>
                </div>
              ) : (
                // Regression Valuation UI
                <div className="space-y-2">
                  <div className="text-xs text-slate-400 uppercase tracking-widest font-semibold flex items-center justify-center gap-1">
                    <TrendingUp className="w-3.5 h-3.5 text-emerald-400" />
                    Forecasted Numeric Valuation
                  </div>
                  <div id="regression-forecast-label" className="text-4xl font-black font-mono text-emerald-400 tracking-tight leading-none py-1 select-all">
                    ${prediction.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                  </div>
                  <div className="text-[10px] text-slate-400 uppercase">Original Scenario Target Unit</div>
                </div>
              )
            ) : (
              <span className="text-slate-400 text-xs">Awaiting data input trigger...</span>
            )}
          </div>

          <div className="z-10 bg-slate-800/40 p-3 rounded-lg border border-slate-700/40 text-[10px] text-slate-300 flex items-center justify-between">
            <span className="flex items-center gap-1">
              Active model configuration:
              <HelpCircle className="w-3 h-3 text-slate-400 animate-bounce" title="Specifies split threshold rules or linear parameters." />
            </span>
            <span className="font-mono text-slate-400">{model.type.toUpperCase()} MODEL</span>
          </div>
        </div>
      </div>
    </div>
  );
}
