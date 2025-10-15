import React, { useState } from "react";
import TaskInput from "./components/TaskInput";
import ConstraintsPanel from "./components/ConstraintsPanel";
import ResultsTable from "./components/ResultsTable";
import { getRecommendations, setBaseline, getModelsForTask } from "./api";

function App() {
  const [taskCategory, setTaskCategory] = useState(null);
  const [filters, setFilters] = useState({ minAcc: 70, maxLat: 500, priority: "balanced" });
  const [recs, setRecs] = useState([]);
  const [recContext, setRecContext] = useState(null);
  const [baselineModel, setBaselineModel] = useState("");
  const [modelsForTask, setModelsForTask] = useState([]);

  async function handleRecommend() {
    if (!taskCategory) {
      alert("Map the task first (press Map task).");
      return;
    }
    try {
      const payload = {
        task: taskCategory,
        min_acc: filters.minAcc,
        max_lat: filters.maxLat,
        priority: filters.priority,
        topk: 5,
        baseline_model: baselineModel || undefined
      };
      const res = await getRecommendations(payload);
        setRecs(res.recommendations);
        setRecContext(res.context || null);
    } catch (err) {
      console.error(err);
      alert("Recommendation error: " + (err?.response?.data?.detail || err.message));
    }
  }

  // fetch models list when taskCategory changes
  React.useEffect(()=>{
    async function load() {
      if (!taskCategory) return setModelsForTask([]);
      try {
        const res = await getModelsForTask(taskCategory);
        setModelsForTask(res.models || []);
      } catch(e){
        console.error(e);
        setModelsForTask([]);
      }
    }
    load();
  }, [taskCategory]);

  return (
    <div style={{ padding: 20 }}>
      <h2>Sustainable AI Model Recommender</h2>
      <TaskInput onTaskConfirmed={(cat) => setTaskCategory(cat)} />
      <ConstraintsPanel onChange={(c)=>{ setFilters(c); }} />
      {taskCategory && (
        <div style={{ marginTop: 12 }}>
          <label>Choose an optional go-to model for this task (optional):</label>
          <div>
            <select value={baselineModel} onChange={(e)=>setBaselineModel(e.target.value)}>
              <option value="">-- no baseline --</option>
              {modelsForTask.map((m)=> (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
            <button onClick={async ()=>{
              if (!baselineModel) return alert('Select a model or leave empty');
              try {
                await setBaseline(taskCategory, baselineModel);
                alert('Baseline saved');
              } catch(e){ console.error(e); alert('Could not save baseline') }
            }}>Save baseline</button>
          </div>
        </div>
      )}
      <div style={{ marginTop: 12 }}>
        <button onClick={handleRecommend}>Get Recommendations</button>
      </div>
      <ResultsTable recs={recs} context={recContext} />
      {recContext && (
        <div style={{ marginTop: 12, padding: 8, border: '1px solid #ddd' }}>
          <h4>Baseline diagnostics</h4>
          <div><strong>Baseline input:</strong> {recContext.baseline_input ?? 'none'}</div>
          <div><strong>Baseline matched model:</strong> {recContext.baseline_matched_model ?? 'none'}</div>
          <div><strong>Baseline energy (Wh/1k):</strong> {recContext.baseline_energy_wh_per_1k ?? 'N/A'}</div>
          <div><strong>Baseline in task:</strong> {String(recContext.baseline_in_task)}</div>
          <div><strong>Baseline passed filters:</strong> {String(recContext.baseline_passed_filters)}</div>
          <div><strong>Baseline comparable score:</strong> {recContext.baseline_comparable_score ?? 'N/A'}</div>
          {recContext.baseline_note && (<div style={{ color: 'orange' }}><strong>Note:</strong> {recContext.baseline_note}</div>)}
        </div>
      )}
    </div>
  );
}

export default App;
