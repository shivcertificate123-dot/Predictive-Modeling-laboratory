import {
  DatasetScenario,
  FeatureSchema,
  DataSample,
  TrainingProgress,
  RegressionMetrics,
  ClassificationMetrics,
  DecisionTreeNode,
  TrainedModel,
  Scaler
} from './types';

// Simple seedable pseudorandom generator for reproducible datasets
export function createRandom(seed: number) {
  let s = seed;
  return function () {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
}

// Scenarios Definitions
export const DATA_SCENARIOS: Record<DatasetScenario, {
  name: string;
  targetKey: string;
  targetName: string;
  targetType: 'regression' | 'classification';
  features: FeatureSchema[];
}> = {
  housing: {
    name: 'Real Estate Valuation',
    targetKey: 'Price',
    targetName: 'House Price',
    targetType: 'regression',
    features: [
      { name: 'Living Area', key: 'Area', type: 'numerical', min: 800, max: 4000, unit: 'sq ft' },
      { name: 'BedroomsCount', key: 'Bedrooms', type: 'numerical', min: 1, max: 5 },
      { name: 'Local School Rating', key: 'School', type: 'numerical', min: 1, max: 10 }
    ]
  },
  churn: {
    name: 'Customer Churn Prevention',
    targetKey: 'Churn',
    targetName: 'Churn Status',
    targetType: 'classification',
    features: [
      { name: 'Monthly Charges', key: 'Charges', type: 'numerical', min: 20, max: 150, unit: '$' },
      { name: 'Account Tenure', key: 'Tenure', type: 'numerical', min: 1, max: 72, unit: 'mon' },
      { name: 'Open Support Tickets', key: 'Tickets', type: 'numerical', min: 0, max: 5 }
    ]
  },
  medical: {
    name: 'Heart Risk Diagnosis',
    targetKey: 'HeartRisk',
    targetName: 'Cardiac Angina Risk',
    targetType: 'classification',
    features: [
      { name: 'Age', key: 'Age', type: 'numerical', min: 30, max: 80, unit: 'yrs' },
      { name: 'Max Heart Rate', key: 'MaxHeartRate', type: 'numerical', min: 90, max: 200, unit: 'bpm' },
      { name: 'Chest Angina History', key: 'AnginaHistory', type: 'binary', min: 0, max: 1 }
    ]
  },
  custom: {
    name: 'Custom Dataset',
    targetKey: 'Target',
    targetName: 'Outcome Label',
    targetType: 'classification',
    features: [
      { name: 'Feature A', key: 'FeatureA', type: 'numerical', min: 0, max: 100 },
      { name: 'Feature B', key: 'FeatureB', type: 'numerical', min: 0, max: 100 }
    ]
  }
};

// Generate Scenario Data
export function generateDataset(scenario: DatasetScenario, count: number = 100, seed: number = 42): DataSample[] {
  const rand = createRandom(seed);
  const data: DataSample[] = [];
  const spec = DATA_SCENARIOS[scenario];

  for (let i = 0; i < count; i++) {
    const row: DataSample = {};

    if (scenario === 'housing') {
      const area = Math.floor(rand() * (4000 - 800) + 800);
      const bedrooms = Math.floor(rand() * 5) + 1;
      const school = Math.floor(rand() * 10) + 1;

      // Base target formula with realistic coefficients + random error
      const noise = (rand() - 0.5) * 45000;
      const basePrice = 80000;
      const price = basePrice + 160 * area + 22000 * bedrooms + 10500 * school + noise;

      row['Area'] = area;
      row['Bedrooms'] = bedrooms;
      row['School'] = school;
      row['Price'] = Math.round(price / 1000) * 1000; // round to nearest thousand
    } else if (scenario === 'churn') {
      const charges = Math.floor(rand() * (150 - 20) + 20);
      const tenure = Math.floor(rand() * (72 - 1) + 1);
      const tickets = Math.floor(rand() * 6);

      // Probability logit
      // Positive coefficient on Charges and Tickets (increases Churn risk)
      // Negative coefficient on Tenure (longer retention, lower Churn risk)
      const xLinear = 0.04 * charges - 0.07 * tenure + 0.65 * tickets - 1.2;
      const prob = 1 / (1 + Math.exp(-xLinear));
      const churn = rand() < prob ? 1 : 0;

      row['Charges'] = charges;
      row['Tenure'] = tenure;
      row['Tickets'] = tickets;
      row['Churn'] = churn;
    } else if (scenario === 'medical') {
      const age = Math.floor(rand() * (80 - 30) + 30);
      const maxHr = Math.floor(rand() * (200 - 90) + 90);
      const angina = rand() < 0.35 ? 1 : 0;

      // Logit function:
      // Risk increases with age, higher angina history, lower max heart rate (often associated with underperformance/disease)
      const xLinear = 0.06 * age - 0.03 * (maxHr - 120) + 1.4 * angina - 2.5;
      const prob = 1 / (1 + Math.exp(-xLinear));
      const heartRisk = rand() < prob ? 1 : 0;

      row['Age'] = age;
      row['MaxHeartRate'] = maxHr;
      row['AnginaHistory'] = angina;
      row['HeartRisk'] = heartRisk;
    } else {
      // custom
      const featureA = Math.floor(rand() * 100);
      const featureB = Math.floor(rand() * 100);
      const prob = featureA + featureB > 100 ? 0.85 : 0.15;
      const outcome = rand() < prob ? 1 : 0;

      row['FeatureA'] = featureA;
      row['FeatureB'] = featureB;
      row['Target'] = outcome;
    }

    data.push(row);
  }

  return data;
}

// Train/Test Splitting Utility
export function splitDataset(data: DataSample[], trainRatio: number = 0.8) {
  // Shuffled copy
  const shuffled = [...data];
  // Simple deterministic shuffle using LCG to make split stable
  const rand = createRandom(12345);
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const temp = shuffled[i];
    shuffled[i] = shuffled[j];
    shuffled[j] = temp;
  }

  const splitIndex = Math.floor(shuffled.length * trainRatio);
  return {
    train: shuffled.slice(0, splitIndex),
    test: shuffled.slice(splitIndex)
  };
}

// Min-Max Scaling helper

export function createScaler(data: DataSample[], keys: string[]): Scaler {
  const min: Record<string, number> = {};
  const max: Record<string, number> = {};

  keys.forEach(key => {
    const vals = data.map(d => d[key]);
    min[key] = Math.min(...vals);
    max[key] = Math.max(...vals);
  });

  return { min, max };
}

export function scaleSample(sample: DataSample, scaler: Scaler, keys: string[]): DataSample {
  const scaled: DataSample = {};
  keys.forEach(key => {
    const num = sample[key];
    const mn = scaler.min[key];
    const mx = scaler.max[key];
    if (mx === mn) {
      scaled[key] = 0.5;
    } else {
      scaled[key] = (num - mn) / (mx - mn);
    }
  });
  return scaled;
}

// -----------------------------------------------------------------------------
// MODEL: LINEAR REGRESSION (Gradient Descent)
// -----------------------------------------------------------------------------
export function trainLinearRegression(
  trainData: DataSample[],
  featureKeys: string[],
  targetKey: string,
  learningRate: number = 0.1,
  epochs: number = 100
): {
  model: TrainedModel;
  scaler: Scaler;
  history: TrainingProgress[];
  targetScaler: { min: number; max: number };
} {
  const m = trainData.length;
  const n = featureKeys.length;

  // Scale features
  const scaler = createScaler(trainData, featureKeys);
  const targetMin = Math.min(...trainData.map(d => d[targetKey]));
  const targetMax = Math.max(...trainData.map(d => d[targetKey]));
  const targetRange = targetMax - targetMin || 1;

  // Normalized train representations
  const X = trainData.map(d => {
    const scaled = scaleSample(d, scaler, featureKeys);
    return featureKeys.map(k => scaled[k]);
  });

  const y = trainData.map(d => {
    // Normalise target as well to prevent divergence
    return (d[targetKey] - targetMin) / targetRange;
  });

  // Initialize weights: zeros
  const weights = new Array(n).fill(0);
  let intercept = 0;

  const history: TrainingProgress[] = [];

  for (let epoch = 1; epoch <= epochs; epoch++) {
    let totalError = 0;
    const dw = new Array(n).fill(0);
    let db = 0;

    for (let i = 0; i < m; i++) {
      let pred = intercept;
      for (let j = 0; j < n; j++) {
        pred += weights[j] * X[i][j];
      }

      const error = pred - y[i];
      totalError += error * error;

      for (let j = 0; j < n; j++) {
        dw[j] += error * X[i][j];
      }
      db += error;
    }

    // Average gradient and update
    for (let j = 0; j < n; j++) {
      weights[j] -= (learningRate * dw[j]) / m;
    }
    intercept -= (learningRate * db) / m;

    const mse = totalError / m;
    // Map back normalized loss to original scale MSE
    const originalMse = mse * targetRange * targetRange;

    history.push({ epoch, loss: originalMse });
  }

  // Convert normalized weights back to raw scale ratios for interpretation if needed, or predict scale-ready
  // We keep weights scaled and pack weights & intercept into our custom TrainedModel structure
  const weightRecord: Record<string, number> = {};
  featureKeys.forEach((k, idx) => {
    weightRecord[k] = weights[idx];
  });

  const model: TrainedModel = {
    type: 'linear',
    weights: weightRecord,
    // Store critical unscaling numbers as extras or pack them in weights
    intercept: intercept,
    featureKeys,
    targetKey
  };

  return {
    model,
    scaler,
    history,
    targetScaler: { min: targetMin, max: targetMax }
  };
}

export function predictLinear(
  sample: DataSample,
  model: TrainedModel,
  scaler: Scaler,
  targetScaler: { min: number; max: number }
): number {
  if (!model.weights || model.intercept === undefined) return 0;

  const scaled = scaleSample(sample, scaler, model.featureKeys);
  let pred = model.intercept;

  model.featureKeys.forEach(k => {
    pred += (model.weights?.[k] || 0) * (scaled[k] || 0);
  });

  // Un-scale target back to original level
  const targetRange = targetScaler.max - targetScaler.min;
  return pred * targetRange + targetScaler.min;
}

// -----------------------------------------------------------------------------
// MODEL: LOGISTIC REGRESSION (Gradient Descent)
// -----------------------------------------------------------------------------
export function trainLogisticRegression(
  trainData: DataSample[],
  featureKeys: string[],
  targetKey: string,
  learningRate: number = 0.1,
  epochs: number = 100
): {
  model: TrainedModel;
  scaler: Scaler;
  history: TrainingProgress[];
} {
  const m = trainData.length;
  const n = featureKeys.length;

  const scaler = createScaler(trainData, featureKeys);

  const X = trainData.map(d => {
    const scaled = scaleSample(d, scaler, featureKeys);
    return featureKeys.map(k => scaled[k]);
  });

  const y = trainData.map(d => (d[targetKey] > 0.5 ? 1 : 0));

  const weights = new Array(n).fill(0);
  let intercept = 0;

  const history: TrainingProgress[] = [];

  for (let epoch = 1; epoch <= epochs; epoch++) {
    let logLossSum = 0;
    const dw = new Array(n).fill(0);
    let db = 0;

    for (let i = 0; i < m; i++) {
      let z = intercept;
      for (let j = 0; j < n; j++) {
        z += weights[j] * X[i][j];
      }

      // Sigmoid
      const prob = 1 / (1 + Math.exp(-Math.max(-15, Math.min(15, z))));
      const error = prob - y[i];

      // Binary Cross Entropy Loss
      const epsilon = 1e-15;
      const safeProb = Math.max(epsilon, Math.min(1 - epsilon, prob));
      logLossSum += -(y[i] * Math.log(safeProb) + (1 - y[i]) * Math.log(1 - safeProb));

      for (let j = 0; j < n; j++) {
        dw[j] += error * X[i][j];
      }
      db += error;
    }

    // Update
    for (let j = 0; j < n; j++) {
      weights[j] -= (learningRate * dw[j]) / m;
    }
    intercept -= (learningRate * db) / m;

    const loss = logLossSum / m;
    history.push({ epoch, loss });
  }

  const weightRecord: Record<string, number> = {};
  featureKeys.forEach((k, idx) => {
    weightRecord[k] = weights[idx];
  });

  const model: TrainedModel = {
    type: 'logistic',
    weights: weightRecord,
    intercept,
    featureKeys,
    targetKey
  };

  return {
    model,
    scaler,
    history
  };
}

export function predictLogisticProb(
  sample: DataSample,
  model: TrainedModel,
  scaler: Scaler
): number {
  if (!model.weights || model.intercept === undefined) return 0;

  const scaled = scaleSample(sample, scaler, model.featureKeys);
  let z = model.intercept;

  model.featureKeys.forEach(k => {
    z += (model.weights?.[k] || 0) * (scaled[k] || 0);
  });

  return 1 / (1 + Math.exp(-Math.max(-15, Math.min(15, z))));
}

// -----------------------------------------------------------------------------
// MODEL: DECISION TREE CLASSIFIER
// -----------------------------------------------------------------------------
function calculateGini(y: number[]): number {
  if (y.length === 0) return 0;
  const counts: Record<number, number> = {};
  y.forEach(val => {
    counts[val] = (counts[val] || 0) + 1;
  });

  let sumSquaredProb = 0;
  const total = y.length;
  for (const c in counts) {
    const p = counts[c] / total;
    sumSquaredProb += p * p;
  }
  return 1 - sumSquaredProb;
}

export function trainDecisionTree(
  trainData: DataSample[],
  featureKeys: string[],
  targetKey: string,
  maxDepth: number = 3,
  minSamplesSplit: number = 4
): {
  model: TrainedModel;
  tree: DecisionTreeNode;
} {
  // Generate names library mapping for display
  const featureNames: Record<string, string> = {};
  Object.values(DATA_SCENARIOS).forEach(s => {
    s.features.forEach(f => {
      featureNames[f.key] = f.name;
    });
  });

  let nodeCounter = 0;

  function buildTree(
    data: DataSample[],
    depth: number
  ): DecisionTreeNode {
    const cellId = `node_${++nodeCounter}`;
    const targetValues = data.map(d => (d[targetKey] > 0.5 ? 1 : 0));
    const ones = targetValues.filter(v => v === 1).length;
    const zeros = targetValues.length - ones;
    const probOne = targetValues.length ? ones / targetValues.length : 0;
    const impurity = calculateGini(targetValues);

    const isFinished =
      depth >= maxDepth ||
      data.length < minSamplesSplit ||
      impurity === 0 ||
      featureKeys.length === 0;

    if (isFinished) {
      return {
        id: cellId,
        isLeaf: true,
        samples: data.length,
        impurity,
        value: probOne // leaf value represents probability of class 1
      };
    }

    // Try finding the best split
    let bestGiniGain = -1;
    let bestFeatureKey = '';
    let bestThreshold = 0;
    let bestLeftData: DataSample[] = [];
    let bestRightData: DataSample[] = [];

    featureKeys.forEach(featKey => {
      // Collect unique values for this feature in numerical form
      const sortedVals = Array.from(new Set(data.map(d => d[featKey] || 0))).sort((a, b) => a - b);
      if (sortedVals.length <= 1) return;

      // Check splits at midpoints
      for (let i = 0; i < sortedVals.length - 1; i++) {
        const threshold = (sortedVals[i] + sortedVals[i + 1]) / 2;

        const left = data.filter(d => (d[featKey] || 0) <= threshold);
        const right = data.filter(d => (d[featKey] || 0) > threshold);

        if (left.length === 0 || right.length === 0) continue;

        const leftY = left.map(d => (d[targetKey] > 0.5 ? 1 : 0));
        const rightY = right.map(d => (d[targetKey] > 0.5 ? 1 : 0));

        const leftGini = calculateGini(leftY);
        const rightGini = calculateGini(rightY);

        const weightedGini = (left.length * leftGini + right.length * rightGini) / data.length;
        const giniGain = impurity - weightedGini;

        if (giniGain > bestGiniGain) {
          bestGiniGain = giniGain;
          bestFeatureKey = featKey;
          bestThreshold = threshold;
          bestLeftData = left;
          bestRightData = right;
        }
      }
    });

    // If no meaningful gain was found, make leaf
    if (bestGiniGain <= 1e-6 || bestLeftData.length === 0 || bestRightData.length === 0) {
      return {
        id: cellId,
        isLeaf: true,
        samples: data.length,
        impurity,
        value: probOne
      };
    }

    // Recurse build
    const leftNode = buildTree(bestLeftData, depth + 1);
    const rightNode = buildTree(bestRightData, depth + 1);

    return {
      id: cellId,
      featureKey: bestFeatureKey,
      featureName: featureNames[bestFeatureKey] || bestFeatureKey,
      threshold: bestThreshold,
      isLeaf: false,
      samples: data.length,
      impurity,
      left: leftNode,
      right: rightNode
    };
  }

  const treeRoot = buildTree(trainData, 0);

  const model: TrainedModel = {
    type: 'tree',
    tree: treeRoot,
    featureKeys,
    targetKey
  };

  return {
    model,
    tree: treeRoot
  };
}

export function predictDecisionTreeProb(
  sample: DataSample,
  node: DecisionTreeNode | undefined
): number {
  if (!node) return 0;
  if (node.isLeaf) {
    return node.value ?? 0;
  }

  const key = node.featureKey;
  const threshold = node.threshold;

  if (key && threshold !== undefined) {
    const val = sample[key] ?? 0;
    if (val <= threshold) {
      return predictDecisionTreeProb(sample, node.left);
    } else {
      return predictDecisionTreeProb(sample, node.right);
    }
  }

  return node.value ?? 0;
}

// -----------------------------------------------------------------------------
// METRICS EVALUATION MATH
// -----------------------------------------------------------------------------
export function evaluateRegression(
  testData: DataSample[],
  model: TrainedModel,
  scaler: Scaler,
  targetScaler: { min: number; max: number }
): RegressionMetrics {
  const m = testData.length;
  if (m === 0) return { mse: 0, rmse: 0, mae: 0, r2: 0 };

  const yTrue = testData.map(d => d[model.targetKey]);
  const yPred = testData.map(d => predictLinear(d, model, scaler, targetScaler));

  let sumSquaredError = 0;
  let sumAbsoluteError = 0;
  let sumTrue = 0;

  for (let i = 0; i < m; i++) {
    const err = yTrue[i] - yPred[i];
    sumSquaredError += err * err;
    sumAbsoluteError += Math.abs(err);
    sumTrue += yTrue[i];
  }

  const yMean = sumTrue / m;
  let sumTotalSquaredError = 0;

  for (let i = 0; i < m; i++) {
    const diff = yTrue[i] - yMean;
    sumTotalSquaredError += diff * diff;
  }

  const mse = sumSquaredError / m;
  const rmse = Math.sqrt(mse);
  const mae = sumAbsoluteError / m;
  const r2 = sumTotalSquaredError === 0 ? 1 : 1 - sumSquaredError / sumTotalSquaredError;

  return { mse, rmse, mae, r2 };
}

export function evaluateClassification(
  testData: DataSample[],
  model: TrainedModel,
  scaler?: Scaler,
  threshold: number = 0.5
): ClassificationMetrics {
  const m = testData.length;
  if (m === 0) {
    return {
      accuracy: 0,
      precision: 0,
      recall: 0,
      f1: 0,
      confusionMatrix: { tp: 0, fn: 0, fp: 0, tn: 0 },
      auc: 0,
      rocCurve: []
    };
  }

  const yTrue = testData.map(d => (d[model.targetKey] > 0.5 ? 1 : 0));

  // Get probability score predictions
  const yProbs = testData.map(d => {
    if (model.type === 'logistic' && scaler) {
      return predictLogisticProb(d, model, scaler);
    } else if (model.type === 'tree') {
      return predictDecisionTreeProb(d, model.tree);
    }
    return 0; // fallback
  });

  // Calculate binary counts for chosen threshold
  let tp = 0, fn = 0, fp = 0, tn = 0;
  for (let i = 0; i < m; i++) {
    const predLabel = yProbs[i] >= threshold ? 1 : 0;
    const trueLabel = yTrue[i];

    if (trueLabel === 1) {
      if (predLabel === 1) tp++;
      else fn++;
    } else {
      if (predLabel === 1) fp++;
      else tn++;
    }
  }

  const accuracy = (tp + tn) / m;
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

  // Calculate standard ROC Curve by checking multiple threshold steps from 1 down to 0
  const steps = 40;
  const rocPoints: Array<{ fpr: number; tpr: number; threshold: number }> = [];

  for (let s = 0; s <= steps; s++) {
    const t = 1 - s / steps; // from 1.0 down to 0.0
    let localTp = 0, localFn = 0, localFp = 0, localTn = 0;

    for (let i = 0; i < m; i++) {
      const predLabel = yProbs[i] >= t ? 1 : 0;
      const trueLabel = yTrue[i];

      if (trueLabel === 1) {
        if (predLabel === 1) localTp++;
        else localFn++;
      } else {
        if (predLabel === 1) localFp++;
        else localTn++;
      }
    }

    const tpr = localTp + localFn > 0 ? localTp / (localTp + localFn) : 0;
    const fpr = localFp + localTn > 0 ? localFp / (localFp + localTn) : 0;

    // To prevent spikes and keep it monotonic, push
    rocPoints.push({ fpr, tpr, threshold: t });
  }

  // Sort ROC curve points by FPR ascending to compute trapezoidal AUC safely
  const sortedRoc = [...rocPoints].sort((a, b) => a.fpr - b.fpr);

  // Calculate AUC (Area under Curve)
  let auc = 0;
  for (let i = 1; i < sortedRoc.length; i++) {
    const xDiff = sortedRoc[i].fpr - sortedRoc[i - 1].fpr;
    const yAvg = (sortedRoc[i].tpr + sortedRoc[i - 1].tpr) / 2;
    auc += xDiff * yAvg;
  }

  // Ensure bounds
  auc = Math.max(0, Math.min(1, auc));

  return {
    accuracy,
    precision,
    recall,
    f1,
    confusionMatrix: { tp, fn, fp, tn },
    auc,
    rocCurve: sortedRoc
  };
}
