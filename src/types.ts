export type DatasetScenario = 'housing' | 'churn' | 'medical' | 'custom';

export interface Scaler {
  min: Record<string, number>;
  max: Record<string, number>;
}

export interface FeatureSchema {
  name: string;
  key: string;
  type: 'numerical' | 'binary';
  min: number;
  max: number;
  unit?: string;
}

export type DataSample = Record<string, number>;

export type ModelType = 'linear' | 'logistic' | 'tree';

export interface TrainingProgress {
  epoch: number;
  loss: number;
}

export interface RegressionMetrics {
  mse: number;
  rmse: number;
  mae: number;
  r2: number;
}

export interface ClassificationMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  confusionMatrix: {
    tp: number;
    fn: number;
    fp: number;
    tn: number;
  };
  auc: number;
  rocCurve: Array<{ fpr: number; tpr: number; threshold: number }>;
}

export interface DecisionTreeNode {
  id: string;
  featureKey?: string;
  featureName?: string;
  threshold?: number;
  value?: number; // predicted class probability/label
  isLeaf: boolean;
  samples: number;
  impurity: number; // Gini impurity
  left?: DecisionTreeNode;
  right?: DecisionTreeNode;
}

// Full specifications of a Trained Model
export interface TrainedModel {
  type: ModelType;
  weights?: Record<string, number>; // for regression
  intercept?: number; // for regression
  tree?: DecisionTreeNode; // for decision tree
  featureKeys: string[];
  targetKey: string;
}
