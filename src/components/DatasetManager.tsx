import React, { useState } from 'react';
import { DATA_SCENARIOS, generateDataset } from '../ml';
import { DatasetScenario, DataSample } from '../types';
import { Database, Plus, Trash2, ListFilter, RefreshCw, HelpCircle } from 'lucide-react';

interface DatasetManagerProps {
  scenario: DatasetScenario;
  onScenarioChange: (s: DatasetScenario) => void;
  dataset: DataSample[];
  setDataset: React.Dispatch<React.SetStateAction<DataSample[]>>;
  trainRatio: number;
  setTrainRatio: (ratio: number) => void;
  seed: number;
  setSeed: (s: number) => void;
}

export default function DatasetManager({
  scenario,
  onScenarioChange,
  dataset,
  setDataset,
  trainRatio,
  setTrainRatio,
  seed,
  setSeed
}: DatasetManagerProps) {
  const [currentPage, setCurrentPage] = useState(1);
  const [newRow, setNewRow] = useState<Record<string, number>>({});
  const itemsPerPage = 8;

  const activeSpec = DATA_SCENARIOS[scenario];
  const features = activeSpec.features;
  const targetKey = activeSpec.targetKey;

  // Pagination bounds
  const totalPages = Math.ceil(dataset.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const pageData = dataset.slice(startIndex, startIndex + itemsPerPage);

  // Trigger regeneration of dataset with new seed or scenario swap
  const regenerate = (newScenario: DatasetScenario) => {
    const nextSeed = Math.floor(Math.random() * 1000);
    setSeed(nextSeed);
    const raw = generateDataset(newScenario, 100, nextSeed);
    setDataset(raw);
    setCurrentPage(1);
  };

  const handleScenarioSelect = (s: DatasetScenario) => {
    onScenarioChange(s);
    regenerate(s);
  };

  const handleDeleteRow = (indexInGlobal: number) => {
    const updated = dataset.filter((_, idx) => idx !== indexInGlobal);
    setDataset(updated);
    // Check if page needs to adjust
    const newTotalPages = Math.ceil(updated.length / itemsPerPage);
    if (currentPage > newTotalPages && currentPage > 1) {
      setCurrentPage(newTotalPages);
    }
  };

  const handleAddNewRow = (e: React.FormEvent) => {
    e.preventDefault();
    const entry: Record<string, number> = {};
    features.forEach(feat => {
      entry[feat.key] = newRow[feat.key] ?? (feat.min + feat.max) / 2;
    });
    entry[targetKey] = newRow[targetKey] ?? (activeSpec.targetType === 'classification' ? 0 : 250000);

    setDataset([entry, ...dataset]);
    setCurrentPage(1);
    setNewRow({});
  };

  return (
    <div id="dataset-manager-panel" className="bg-white rounded-2xl border border-slate-100 shadow-xs p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-indigo-50 flex items-center justify-center text-indigo-600">
            <Database className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-slate-800">1. Dataset Selection & Configuration</h2>
            <p className="text-xs text-slate-500">Choose or edit data to train supervised learning algorithms</p>
          </div>
        </div>

        <button
          onClick={() => regenerate(scenario)}
          id="btn-regenerate-data"
          title="Regenerate dataset with random seed"
          className="p-2 text-slate-400 hover:text-indigo-600 hover:bg-slate-50 rounded-lg Transition-all active:scale-95 flex items-center space-x-1 border border-slate-100"
        >
          <RefreshCw className="w-4 h-4 cursor-pointer" />
          <span className="text-xs font-semibold px-1">Reroll seed: {seed}</span>
        </button>
      </div>

      {/* Scenario Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {(Object.keys(DATA_SCENARIOS) as DatasetScenario[]).map(key => {
          const spec = DATA_SCENARIOS[key];
          const isActive = scenario === key;
          return (
            <button
              key={key}
              id={`scenario-tab-${key}`}
              onClick={() => handleScenarioSelect(key)}
              className={`p-4 rounded-xl border text-left transition-all ${
                isActive
                  ? 'border-indigo-500 bg-indigo-50/40 shadow-xs ring-1 ring-indigo-500/10'
                  : 'border-slate-100 hover:border-slate-200 bg-slate-50/20'
              }`}
            >
              <div className="font-semibold text-slate-800 text-sm flex items-center justify-between">
                {spec.name}
                <span
                  className={`text-[10px] font-bold px-1.5 py-0.5 rounded-full ${
                    spec.targetType === 'classification'
                      ? 'bg-amber-100 text-amber-800'
                      : 'bg-emerald-100 text-emerald-800'
                  }`}
                >
                  {spec.targetType === 'classification' ? 'Classification' : 'Regression'}
                </span>
              </div>
              <div className="text-xs text-slate-500 mt-1 block">
                Target: <span className="font-medium text-slate-700">{spec.targetName}</span>
              </div>
              <div className="text-[11px] text-slate-400 mt-2">
                Features: {spec.features.map(f => f.name).join(', ')}
              </div>
            </button>
          );
        })}
      </div>

      {/* Split Ratios */}
      <div className="space-y-3 bg-slate-50/50 p-4 rounded-xl border border-slate-100">
        <div className="flex justify-between items-center text-xs">
          <span className="font-medium text-slate-700 flex items-center gap-1">
            Train / Test Partition Ratio:
            <HelpCircle className="w-3.5 h-3.5 text-slate-400" title="Fraction of data used for training vs validation" />
          </span>
          <span className="font-mono text-indigo-700 font-bold bg-indigo-50 px-2 py-0.5 rounded-md">
            {Math.round(trainRatio * 100)}% Train / {Math.round((1 - trainRatio) * 100)}% Test
          </span>
        </div>
        <input
          type="range"
          id="train-split-slider"
          min="0.5"
          max="0.9"
          step="0.05"
          value={trainRatio}
          onChange={e => setTrainRatio(parseFloat(e.target.value))}
          className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
        />
        <div className="grid grid-cols-2 text-[11px] text-slate-500 font-mono">
          <div className="text-left border-r border-slate-100 pr-2">
            Training set size: <span className="font-semibold text-slate-700">{Math.floor(dataset.length * trainRatio)} samples</span>
          </div>
          <div className="text-right pl-2">
            Testing set size: <span className="font-semibold text-slate-700">{dataset.length - Math.floor(dataset.length * trainRatio)} samples</span>
          </div>
        </div>
      </div>

      {/* Dataset Preview with local inserts */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-slate-700 flex items-center gap-2">
            <ListFilter className="w-4 h-4 text-slate-500" />
            Interactive Data Records ({dataset.length} loaded)
          </h3>
          <span className="text-xs text-slate-400">Showing rows {startIndex + 1}-{Math.min(startIndex + itemsPerPage, dataset.length)}</span>
        </div>

        <div className="overflow-x-auto border border-slate-100 rounded-xl">
          <table id="dataset-preview-table" className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="bg-slate-50 border-b border-slate-100 text-slate-500 font-medium">
                <th className="p-2.5 font-semibold text-center w-12">#</th>
                {features.map(feat => (
                  <th key={feat.key} className="p-2.5 font-semibold text-slate-600">
                    {feat.name} {feat.unit ? `(${feat.unit})` : ''}
                  </th>
                ))}
                <th className="p-2.5 font-semibold text-indigo-700 bg-indigo-50/40">
                  {activeSpec.targetName}
                </th>
                <th className="p-2.5 font-semibold w-12 text-center text-slate-400">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-50 text-slate-700 font-mono">
              {pageData.map((row, index) => {
                const globalIndex = startIndex + index;
                return (
                  <tr key={globalIndex} className="hover:bg-slate-50/40 transition-colors">
                    <td className="p-2 text-center text-slate-400 text-[10px]">{dataset.length - globalIndex}</td>
                    {features.map(feat => (
                      <td key={feat.key} className="p-2">
                        {row[feat.key] ?? '-'}
                      </td>
                    ))}
                    <td className="p-2 font-semibold text-indigo-600 bg-indigo-50/10">
                      {row[targetKey]}
                    </td>
                    <td className="p-2 text-center">
                      <button
                        onClick={() => handleDeleteRow(globalIndex)}
                        title="Delete this row"
                        className="p-1 text-slate-300 hover:text-red-500 hover:bg-red-50 rounded-md transition-colors"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Pagination & Add Row Form */}
        <div className="flex flex-col sm:flex-row justify-between items-center gap-3 pt-1">
          <div className="flex space-x-1">
            <button
              disabled={currentPage === 1}
              onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
              className="px-2.5 py-1 text-xs border border-slate-100 rounded-lg hover:bg-slate-50 disabled:opacity-40"
            >
              Prev
            </button>
            <span className="px-3 py-1 text-xs text-slate-500 font-medium">Page {currentPage} of {totalPages}</span>
            <button
              disabled={currentPage === totalPages}
              onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
              className="px-2.5 py-1 text-xs border border-slate-100 rounded-lg hover:bg-slate-50 disabled:opacity-40"
            >
              Next
            </button>
          </div>

          <form onSubmit={handleAddNewRow} className="flex flex-wrap items-center gap-2 self-end">
            <div className="text-[11px] text-slate-500 font-sans hidden md:block">Insert custom row:</div>
            {features.map(feat => (
              <input
                key={feat.key}
                type="number"
                placeholder={feat.name}
                step={feat.type === 'binary' ? '1' : 'any'}
                min={feat.min}
                max={feat.max}
                value={newRow[feat.key] ?? ''}
                onChange={e => setNewRow({ ...newRow, [feat.key]: parseFloat(e.target.value) })}
                className="w-20 px-2 py-1 text-xs border border-slate-200 rounded-lg focus:border-indigo-400 focus:outline-hidden"
                required
              />
            ))}
            <input
              type="number"
              step={activeSpec.targetType === 'classification' ? '1' : 'any'}
              min={activeSpec.targetType === 'classification' ? 0 : 0}
              max={activeSpec.targetType === 'classification' ? 1 : undefined}
              placeholder={activeSpec.targetName}
              value={newRow[targetKey] ?? ''}
              onChange={e => setNewRow({ ...newRow, [targetKey]: parseFloat(e.target.value) })}
              className="w-24 px-2 py-1 text-xs border border-indigo-200 rounded-lg bg-indigo-50/20 focus:border-indigo-400 focus:outline-hidden font-bold"
              required
            />
            <button
              type="submit"
              className="px-2.5 py-1 text-xs bg-slate-800 text-white hover:bg-indigo-600 rounded-lg flex items-center gap-1 transition-all"
            >
              <Plus className="w-3.5 h-3.5" />
              <span>Add</span>
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
