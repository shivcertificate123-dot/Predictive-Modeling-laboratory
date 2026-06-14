import { DecisionTreeNode } from '../types';
import { GitCommit, GitMerge, FileText, HelpCircle } from 'lucide-react';

interface DecisionTreeVisualizerProps {
  rootNode: DecisionTreeNode | null;
}

export default function DecisionTreeVisualizer({ rootNode }: DecisionTreeVisualizerProps) {
  if (!rootNode) {
    return (
      <div className="bg-slate-50 border border-dashed border-slate-200 rounded-2xl p-8 text-center text-slate-400">
        Train a Decision Tree model to inspect split criterion branches!
      </div>
    );
  }

  // Recursive component to draw tree node blocks
  function RenderNode({ node, pathCondition }: { node: DecisionTreeNode; pathCondition?: string }) {
    return (
      <div className="flex flex-col items-center space-y-4 w-full">
        {/* If path condition exists, show bridging badge */}
        {pathCondition && (
          <div className="px-2 py-0.5 bg-slate-100 border border-slate-200 text-[10px] text-slate-600 font-mono font-bold rounded-full shadow-xxs">
            {pathCondition}
          </div>
        )}

        {/* Tree Node Card */}
        <div
          id={`dt-${node.id}`}
          className={`p-4 rounded-xl border text-left min-w-[210px] max-w-[260px] transition-all bg-white relative hover:scale-[1.01] hover:shadow-md ${
            node.isLeaf
              ? 'border-indigo-200 shadow-xxs bg-gradient-to-br from-white to-indigo-50/20'
              : 'border-slate-200 shadow-sm'
          }`}
        >
          {/* Node Header */}
          <div className="flex items-start justify-between">
            <span
              className={`p-1.5 rounded-lg flex items-center justify-center ${
                node.isLeaf ? 'bg-indigo-100 text-indigo-700' : 'bg-slate-100 text-slate-700'
              }`}
            >
              {node.isLeaf ? <FileText className="w-3.5 h-3.5" /> : <GitMerge className="w-3.5 h-3.5" />}
            </span>
            <span className="text-[9px] font-mono font-bold text-slate-400 self-center">ID: {node.id.toUpperCase()}</span>
          </div>

          <div className="mt-2.5 space-y-1">
            {/* Split Criterion / Leaf Output */}
            {node.isLeaf ? (
              <div className="space-y-1">
                <div className="text-xs text-slate-400 font-semibold uppercase tracking-wider">Predict Probability</div>
                <div className="text-lg font-black font-mono text-indigo-700 flex items-baseline">
                  {(node.value! * 100).toFixed(0)}%
                  <span className="text-[10px] font-normal text-slate-500 ml-1">Class One</span>
                </div>
                <div className="text-[10px] text-slate-500 font-semibold">
                  Majority Class:{' '}
                  <span className={`px-1 rounded-sm ${node.value! >= 0.5 ? 'bg-indigo-100 text-indigo-800' : 'bg-slate-200 text-slate-800'}`}>
                    {node.value! >= 0.5 ? '1 (Positive)' : '0 (Negative)'}
                  </span>
                </div>
              </div>
            ) : (
              <div className="space-y-1">
                <div className="text-xs text-slate-400 font-semibold uppercase tracking-wider">Test Split</div>
                <div className="text-xs text-slate-800 font-bold bg-slate-50 border border-slate-100 p-1.5 rounded-md font-mono break-all leading-tight">
                  {node.featureName} <span className="text-pink-600 font-semibold">&le;</span> {node.threshold?.toFixed(1)}
                </div>
              </div>
            )}

            {/* Diagnostic stats */}
            <div className="pt-2 border-t border-slate-100 grid grid-cols-2 text-[10px] text-slate-500 font-mono">
              <div>
                Gini: <strong className="text-slate-700">{node.impurity.toFixed(3)}</strong>
              </div>
              <div className="text-right">
                Samples: <strong className="text-slate-700">{node.samples}</strong>
              </div>
            </div>
          </div>
        </div>

        {/* Draw children branches side by side if not leaf */}
        {!node.isLeaf && (node.left || node.right) && (
          <div className="relative flex flex-col sm:flex-row gap-6 w-full items-start justify-center pt-2">
            {/* Connecting Visual Lines (Symmetrical Branches) */}
            <div className="absolute top-0 left-1/2 w-0.5 h-4 bg-slate-200 -translate-x-1/2 hidden sm:block" />

            {node.left && (
              <div className="flex-1 flex flex-col items-center w-full">
                <RenderNode node={node.left} pathCondition={`Yes (≤ ${node.threshold?.toFixed(1)})`} />
              </div>
            )}
            {node.right && (
              <div className="flex-1 flex flex-col items-center w-full">
                <RenderNode node={node.right} pathCondition={`No (&gt; ${node.threshold?.toFixed(1)})`} />
              </div>
            )}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="bg-white rounded-2xl border border-slate-100 p-6 space-y-6 shadow-xs overflow-x-auto">
      <div className="flex items-center justify-between min-w-[500px]">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-orange-50 flex items-center justify-center text-orange-600">
            <GitCommit className="w-5 h-5 rotate-90" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-slate-800">Hierarchical Classification Tree Diagram</h3>
            <p className="text-xs text-slate-500">Each branch checks features to classify samples recursively</p>
          </div>
        </div>

        <div className="flex gap-2 text-[10px] font-medium font-mono">
          <span className="flex items-center gap-1 bg-indigo-50 text-indigo-700 px-2 py-1 border border-indigo-100 rounded-md">
            <span className="w-1.5 h-1.5 bg-indigo-600 rounded-full" />
            Class probability leafs
          </span>
          <span className="flex items-center gap-1 bg-slate-50 text-slate-600 px-2 py-1 border border-slate-100 rounded-md">
            Gini Impurity = Splitting entropy
            <HelpCircle className="w-3 h-3 text-slate-400" title="0.0 means perfect classification purity; 0.5 means perfect random overlap" />
          </span>
        </div>
      </div>

      {/* Recursive tree roots */}
      <div className="flex justify-center py-6 min-w-[600px] border-t border-slate-50">
        <RenderNode node={rootNode} />
      </div>
    </div>
  );
}
