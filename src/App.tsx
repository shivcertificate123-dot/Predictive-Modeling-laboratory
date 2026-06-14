import { useState, useEffect, useMemo } from 'react';
import {
  DatasetScenario,
  DataSample,
  TrainedModel,
  TrainingProgress,
  RegressionMetrics,
  ClassificationMetrics,
  Scaler
} from './types';
import {
  DATA_SCENARIOS,
  generateDataset,
  splitDataset,
  createScaler,
  scaleSample,
  trainDecisionTree,
  evaluateRegression,
  evaluateClassification,
  predictLinear,
  predictLogisticProb,
  predictDecisionTreeProb
} from './ml';

import DatasetManager from './components/DatasetManager';
import RegressionDashboard from './components/RegressionDashboard';
import ClassificationDashboard from './components/ClassificationDashboard';
import DecisionTreeVisualizer from './components/DecisionTreeVisualizer';
import InteractivePredictor from './components/InteractivePredictor';

import {
  BrainCircuit,
  Cpu,
  TrendingUp,
  Settings2,
  Info,
  BookOpen,
  Sliders,
  CheckCircle2,
  Sparkles
} from 'lucide-react';

export default function App() {
  // Data State
  const [scenario, setScenario] = useState<DatasetScenario>('housing');
  const [seed, setSeed] = useState<number>(42);
  const [dataset, setDataset] = useState<DataSample[]>(() =>
    generateDataset('housing', 100, 42)
  );
  const [trainRatio, setTrainRatio] = useState<number>(0.8);

  // Hyperparameters
  const [learningRate, setLearningRate] = useState<number>(0.1);
  const [epochs, setEpochs] = useState<number>(100);
  const [maxDepth, setMaxDepth] = useState<number>(3);

  // Model & Metrics state
  const [selectedModelType, setSelectedModelType] = useState<'linear' | 'logistic' | 'tree'>('linear');
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [trainedModel, setTrainedModel] = useState<TrainedModel | null>(null);
  const [scaler, setScaler] = useState<Scaler | null>(null);
  const [targetScaler, setTargetScaler] = useState<{ min: number; max: number } | null>(null);
  const [trainingHistory, setTrainingHistory] = useState<TrainingProgress[]>([]);
  const [classificationThreshold, setClassificationThreshold] = useState<number>(0.5);

  // Auto-switch default model type when scenario changes to respect task applicability
  useEffect(() => {
    const spec = DATA_SCENARIOS[scenario];
    if (spec.targetType === 'regression') {
      setSelectedModelType('linear');
    } else {
      setSelectedModelType('logistic'); // default classification model
    }
    // Clear out trained models from different scenario
    setTrainedModel(null);
    setScaler(null);
    setTargetScaler(null);
    setTrainingHistory([]);
  }, [scenario]);

  const activeSpec = DATA_SCENARIOS[scenario];
  const featureKeys = useMemo(() => activeSpec.features.map(f => f.key), [activeSpec]);
  const targetKey = activeSpec.targetKey;

  // Split datasets deterministically
  const splitData = useMemo(() => {
    return splitDataset(dataset, trainRatio);
  }, [dataset, trainRatio]);

  // Train the chosen model with step-by-step SVG loss updates
  const handleTrainModel = () => {
    if (isTraining) return;

    setTrainedModel(null);
    setTrainingHistory([]);

    const trainSet = splitData.train;
    const n = featureKeys.length;

    if (selectedModelType === 'tree') {
      // Decision Tree trains instantly structure-wise
      setIsTraining(true);
      setTimeout(() => {
        const { model, tree } = trainDecisionTree(trainSet, featureKeys, targetKey, maxDepth, 4);
        setTrainedModel(model);
        setScaler(null);
        setTargetScaler(null);
        setIsTraining(false);
      }, 550); // Small aesthetic interval to convey mathematical execution
      return;
    }

    // Gradient descent algorithms (Linear & Logistic)
    setIsTraining(true);

    let currentWeights = new Array(n).fill(0);
    let currentIntercept = 0;
    const historyLog: TrainingProgress[] = [];

    // Min-max scalers
    const featureScaler = createScaler(trainSet, featureKeys);
    setScaler(featureScaler);

    const targetMin = Math.min(...trainSet.map(d => d[targetKey]));
    const targetMax = Math.max(...trainSet.map(d => d[targetKey]));
    const targetRange = targetMax - targetMin || 1;

    if (selectedModelType === 'linear') {
      setTargetScaler({ min: targetMin, max: targetMax });
    } else {
      setTargetScaler(null); // classification
    }

    // Pre-scale train samples for raw iteration matrix
    const X = trainSet.map(d => {
      const scaled = scaleSample(d, featureScaler, featureKeys);
      return featureKeys.map(k => scaled[k]);
    });

    const isLog = selectedModelType === 'logistic';
    const y = trainSet.map(d => {
      if (isLog) {
        return d[targetKey] > 0.5 ? 1 : 0;
      } else {
        return (d[targetKey] - targetMin) / targetRange;
      }
    });

    let currentEpoch = 1;

    const runTrainingStep = () => {
      if (currentEpoch > epochs) {
        setIsTraining(false);
        return;
      }

      // Process in small batches of 3 epochs per frame to create high-accuracy visual scaling
      const stepsInThisFrame = 3;
      for (let s = 0; s < stepsInThisFrame && currentEpoch <= epochs; s++) {
        let logLossSum = 0;
        let totalError = 0;
        const dw = new Array(n).fill(0);
        let db = 0;

        for (let i = 0; i < trainSet.length; i++) {
          let pred = currentIntercept;
          for (let j = 0; j < n; j++) {
            pred += currentWeights[j] * X[i][j];
          }

          if (isLog) {
            const prob = 1 / (1 + Math.exp(-Math.max(-15, Math.min(15, pred))));
            const error = prob - y[i];

            const epsilon = 1e-15;
            const safeProb = Math.max(epsilon, Math.min(1 - epsilon, prob));
            // Log Loss: - [ y*log(p) + (1-y)*log(1-p) ]
            logLossSum += -(y[i] * Math.log(safeProb) + (1 - y[i]) * Math.log(1 - safeProb));

            for (let j = 0; j < n; j++) {
              dw[j] += error * X[i][j];
            }
            db += error;
          } else {
            const error = pred - y[i];
            totalError += error * error;

            for (let j = 0; j < n; j++) {
              dw[j] += error * X[i][j];
            }
            db += error;
          }
        }

        // Apply parameter updates
        for (let j = 0; j < n; j++) {
          currentWeights[j] -= (learningRate * dw[j]) / trainSet.length;
        }
        currentIntercept -= (learningRate * db) / trainSet.length;

        const evaluatedLoss = isLog
          ? logLossSum / trainSet.length
          : (totalError / trainSet.length) * targetRange * targetRange;

        historyLog.push({ epoch: currentEpoch, loss: evaluatedLoss });
        currentEpoch++;
      }

      setTrainingHistory([...historyLog]);

      // Update intermediate model coefficients so UI plots change smoothly!
      const weightRecord: Record<string, number> = {};
      featureKeys.forEach((k, idx) => {
        weightRecord[k] = currentWeights[idx];
      });

      setTrainedModel({
        type: selectedModelType,
        weights: weightRecord,
        intercept: currentIntercept,
        featureKeys,
        targetKey
      });

      requestAnimationFrame(runTrainingStep);
    };

    requestAnimationFrame(runTrainingStep);
  };

  // Compute evaluation stats in real-time based on active model and split ratios
  const testMetrics = useMemo(() => {
    if (!trainedModel) return null;

    if (activeSpec.targetType === 'regression') {
      if (scaler && targetScaler) {
        return evaluateRegression(splitData.test, trainedModel, scaler, targetScaler);
      }
    } else {
      // Classification (logistic or tree)
      return evaluateClassification(splitData.test, trainedModel, scaler || undefined, classificationThreshold);
    }
    return null;
  }, [trainedModel, splitData.test, scaler, targetScaler, activeSpec, classificationThreshold]);

  // Actual vs Predicted pairings for regression visualizers
  const actualVsPredictedPairings = useMemo(() => {
    if (!trainedModel || activeSpec.targetType !== 'regression' || !scaler || !targetScaler) return [];
    return splitData.test.map(sample => {
      const actual = sample[targetKey];
      const pred = predictLinear(sample, trainedModel, scaler, targetScaler);
      return { actual, pred };
    });
  }, [trainedModel, splitData.test, scaler, targetScaler, activeSpec, targetKey]);

  return (
    <div className="min-h-screen bg-slate-50/50 pb-16 font-sans antialiased text-slate-800">
      {/* Title Header banner */}
      <header className="bg-slate-900 text-white py-8 px-6 md:px-12 relative overflow-hidden border-b border-indigo-900/40">
        <div className="absolute top-0 right-0 w-80 h-80 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute -bottom-10 left-10 w-96 h-90 bg-purple-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-start md:items-center justify-between gap-6 relative z-10">
          <div className="space-y-2">
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-indigo-500/20 text-indigo-300 text-xs font-semibold uppercase tracking-wider">
              <BrainCircuit className="w-3.5 h-3.5" />
              Supervised Learning Lab
            </div>
            <h1 className="text-3xl md:text-4xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white via-slate-100 to-indigo-200">
              Predictive Modeling Laboratory
            </h1>
            <p className="text-sm text-slate-400 font-medium max-w-2xl">
              Train, trace, and evaluate analytical algorithms directly in your browser. Configure training splits, modify parameters, and inspect fit diagnostics dynamically.
            </p>
          </div>

          <div className="flex flex-wrap gap-2 text-xs">
            <span className="flex items-center gap-1 bg-slate-800/80 border border-slate-700/50 text-slate-300 px-3 py-1.5 rounded-lg font-mono">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              In-Browser Solver Runtime Active
            </span>
          </div>
        </div>
      </header>

      {/* Main Workspace Layout */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 md:px-8 mt-8 grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left column: Dataset Config, Algorithms settings (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          {/* Section 1: Data Manager */}
          <DatasetManager
            scenario={scenario}
            onScenarioChange={setScenario}
            dataset={dataset}
            setDataset={setDataset}
            trainRatio={trainRatio}
            setTrainRatio={setTrainRatio}
            seed={seed}
            setSeed={setSeed}
          />

          {/* Section 2: Model Configuration & Training Block */}
          <div id="model-training-configuration" className="bg-white rounded-2xl border border-slate-100 shadow-xs p-6 space-y-6">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 rounded-xl bg-amber-50 flex items-center justify-center text-amber-600">
                <Cpu className="w-5 h-5 animate-pulse" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-slate-800">2. Model Selection & Solver Parameters</h2>
                <p className="text-xs text-slate-500">Choose a fitting approach, configure learning depth, and trigger backprop minimization</p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-1">
              {/* Selector */}
              <div className="space-y-4">
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1">
                  <Sliders className="w-3.5 h-3.5" />
                  Select Supervised Algorithm
                </h3>

                {activeSpec.targetType === 'regression' ? (
                  <div className="space-y-2">
                    <button
                      id="algo-select-linear"
                      onClick={() => setSelectedModelType('linear')}
                      className="w-full p-4 rounded-xl border border-indigo-500 bg-indigo-50/20 text-left font-medium text-xs text-indigo-900 flex justify-between items-center"
                    >
                      <div>
                        <div className="font-bold text-sm text-slate-800">Linear Regression</div>
                        <div className="text-[11px] text-slate-500 font-normal mt-0.5">Fits multivariate linear hyperplane functions</div>
                      </div>
                      <span className="w-1.5 h-1.5 bg-indigo-600 rounded-full" />
                    </button>
                    <p className="text-[11px] text-slate-400 leading-relaxed italic">
                      *Classification algorithms (Logistic Regression, Decision Trees) are disabled for continuous numerical Real Estate targets.
                    </p>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 gap-2.5">
                    <button
                      id="algo-select-logistic"
                      onClick={() => setSelectedModelType('logistic')}
                      className={`p-4 rounded-xl border text-left font-medium text-xs transition-colors ${
                        selectedModelType === 'logistic'
                          ? 'border-indigo-500 bg-indigo-50/20 text-indigo-900'
                          : 'border-slate-100 hover:border-slate-200'
                      }`}
                    >
                      <div className="font-bold text-sm text-slate-800">Logistic Regression</div>
                      <div className="text-[11px] text-slate-500 font-normal mt-0.5">S-shaped Sigmoid probability classifier</div>
                    </button>

                    <button
                      id="algo-select-tree"
                      onClick={() => setSelectedModelType('tree')}
                      className={`p-4 rounded-xl border text-left font-medium text-xs transition-colors ${
                        selectedModelType === 'tree'
                          ? 'border-indigo-500 bg-indigo-50/20 text-indigo-900'
                          : 'border-slate-100 hover:border-slate-200'
                      }`}
                    >
                      <div className="font-bold text-sm text-slate-800">Decision Tree Classifier</div>
                      <div className="text-[11px] text-slate-500 font-normal mt-0.5">Recursive Gini impurity binary partitions</div>
                    </button>
                  </div>
                )}
              </div>

              {/* Hyperparameters Slider Block */}
              <div className="space-y-4 bg-slate-50 p-4 rounded-xl border border-slate-100">
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500 flex items-center gap-1">
                  <Settings2 className="w-3.5 h-3.5" />
                  Hyperparameters Tuning
                </h3>

                {selectedModelType === 'tree' ? (
                  // Tree splits parameters
                  <div className="space-y-4">
                    <div className="space-y-1.5">
                      <div className="flex justify-between text-xs">
                        <span className="text-slate-600 font-medium">Max Depth Limit:</span>
                        <strong className="text-slate-800 font-mono">{maxDepth} levels</strong>
                      </div>
                      <input
                        type="range"
                        min="1"
                        max="4"
                        step="1"
                        value={maxDepth}
                        onChange={e => setMaxDepth(parseInt(e.target.value))}
                        className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                      />
                      <div className="text-[9px] text-slate-400">Restricting depth limits structural overfitting</div>
                    </div>
                  </div>
                ) : (
                  // Gradient descent parameters
                  <div className="space-y-4">
                    <div className="space-y-1.5">
                      <div className="flex justify-between text-xs">
                        <span className="text-slate-600 font-semibold">Learning Rate (&alpha;):</span>
                        <strong className="text-slate-800 font-mono">{learningRate}</strong>
                      </div>
                      <input
                        type="range"
                        min="0.01"
                        max="0.30"
                        step="0.01"
                        value={learningRate}
                        onChange={e => setLearningRate(parseFloat(e.target.value))}
                        className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                      />
                    </div>

                    <div className="space-y-1.5">
                      <div className="flex justify-between text-xs">
                        <span className="text-slate-600 font-semibold">Max Epochs:</span>
                        <strong className="text-slate-800 font-mono">{epochs} iterations</strong>
                      </div>
                      <input
                        type="range"
                        min="50"
                        max="300"
                        step="10"
                        value={epochs}
                        onChange={e => setEpochs(parseInt(e.target.value))}
                        className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                      />
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Train Button */}
            <button
              onClick={handleTrainModel}
              id="btn-trigger-model-training"
              disabled={isTraining || dataset.length === 0}
              className={`w-full py-4 text-sm font-black uppercase tracking-wider rounded-xl transition-all active:scale-[0.99] flex items-center justify-center gap-2 ${
                isTraining
                  ? 'bg-indigo-100 text-indigo-500 cursor-not-allowed'
                  : 'bg-indigo-600 hover:bg-slate-900 text-white shadow-sm hover:shadow-md'
              }`}
            >
              {isTraining ? (
                <>
                  <div className="w-5 h-5 border-2 border-indigo-600 border-t-transparent rounded-full animate-spin" />
                  <span>Iterating over cost thresholds...</span>
                </>
              ) : (
                <>
                  <BrainCircuit className="w-5 h-5 text-indigo-200 animate-pulse" />
                  <span>Train {selectedModelType === 'linear' ? 'Linear Model' : selectedModelType === 'logistic' ? 'Logistic Model' : 'Decision Tree'}</span>
                </>
              )}
            </button>
          </div>

          {/* Model Interactive Sample Play Predictor */}
          <InteractivePredictor
            scenario={scenario}
            model={trainedModel}
            scaler={scaler}
            targetScaler={targetScaler}
            classificationThreshold={classificationThreshold}
          />
        </div>

        {/* Right column: Training Logs, Diagnostics metrics, ROCs / Tree Map (5 cols) */}
        <div id="diagnostics-panel" className="lg:col-span-5 space-y-6">
          <div className="bg-slate-900 hover:bg-slate-950 text-white rounded-2xl p-6 border border-slate-800 flex items-center justify-between">
            <div className="space-y-1.5">
              <span className="text-[10px] font-bold tracking-widest text-indigo-400 uppercase flex items-center gap-1 select-none">
                <CheckCircle2 className="w-3.5 h-3.5 text-indigo-400" />
                Solver Diagnostics Panel
              </span>
              <h3 className="text-xl font-bold">Model Performance</h3>
            </div>
            {trainedModel && (
              <span className="text-xs bg-indigo-500/20 border border-indigo-500/30 text-indigo-300 font-mono font-black px-2.5 py-1 rounded-md">
                {trainedModel.type.toUpperCase()} TRAINED
              </span>
            )}
          </div>

          {/* Render target metrics dashboards based on model goal type */}
          {activeSpec.targetType === 'regression' ? (
            <RegressionDashboard
              metrics={testMetrics as RegressionMetrics | null}
              history={trainingHistory}
              model={trainedModel}
              testSamplesCount={splitData.test.length}
              actualVsPred={actualVsPredictedPairings}
            />
          ) : (
            testMetrics && (
              <ClassificationDashboard
                metrics={testMetrics as ClassificationMetrics}
                threshold={classificationThreshold}
                setThreshold={setClassificationThreshold}
                testSamplesCount={splitData.test.length}
              />
            )
          )}

          {/* If there is no trained model, display placeholder */}
          {!trainedModel && (
            <div className="bg-white rounded-2xl border border-slate-100 p-8 text-center text-slate-400 space-y-3">
              <Info className="w-8 h-8 text-indigo-300 mx-auto" />
              <div>
                <p className="text-sm font-semibold text-slate-600">Model has not been trained yet</p>
                <p className="text-xs text-slate-400 max-w-sm mx-auto mt-1">
                  Adjust features in Section 1 or configure training rates and click the <strong>"Train Model"</strong> button to execute gradient learning.
                </p>
              </div>
            </div>
          )}

          {/* Educational Glossary of ML terms */}
          <div className="bg-white rounded-2xl border border-slate-100 p-6 space-y-4">
            <h3 className="text-sm font-bold text-slate-800 flex items-center gap-2 border-b border-slate-100 pb-2.5">
              <BookOpen className="w-4 h-4 text-indigo-500" />
              Machine Learning Glossary
            </h3>

            <div className="space-y-3 text-xs text-slate-600 max-h-[290px] overflow-y-auto pr-1">
              <div>
                <span className="font-extrabold text-slate-800 block">Linear Regression:</span>
                Predicts a continuous numeric outcome by finding the line (or hyperplane) of best fit that minimizes squared deviations.
              </div>
              <div>
                <span className="font-extrabold text-slate-800 block">Logistic Regression:</span>
                Classifies binary outcomes (0 or 1) by estimating probability via a Sigmoid logic function.
              </div>
              <div>
                <span className="font-extrabold text-slate-800 block">Decision Trees:</span>
                Splits data repeatedly based on feature tests that maximize node "homogeneity" or class purity.
              </div>
              <div>
                <span className="font-extrabold text-slate-800 block">Gini Impurity index:</span>
                A measure from 0.0 (pure node) to 0.5 (random split) checking probability of misclassification.
              </div>
              <div>
                <span className="font-extrabold text-slate-800 block">R-squared (R²):</span>
                Indicates the fraction of target variable variance explained by predictors. 1.0 represents a perfect fit.
              </div>
              <div>
                <span className="font-extrabold text-slate-800 block">Confusion Matrix:</span>
                Tabular grid pairing true outcomes (events) against model classifications. Highlights errors (FP / FN).
              </div>
              <div>
                <span className="font-extrabold text-slate-800 block">ROC Path / AUC Area:</span>
                Receiver Operating Characteristic evaluates sensitivity across threshold splits. Higher AUC (close to 1.0) means stronger diagnostic separation.
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Decision Tree Diagram shown spanning bottom if decision tree model trained */}
      {trainedModel && trainedModel.type === 'tree' && (
        <div id="tree-map-visualization-area" className="max-w-7xl mx-auto px-4 sm:px-6 md:px-8 mt-8">
          <DecisionTreeVisualizer rootNode={trainedModel.tree || null} />
        </div>
      )}
    </div>
  );
}
