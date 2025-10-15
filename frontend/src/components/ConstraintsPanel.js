import React, { useState } from "react";

export default function ConstraintsPanel({ onChange }) {
  const [minAcc, setMinAcc] = useState(70);
  const [maxLat, setMaxLat] = useState(500);
  const [priority, setPriority] = useState("balanced");

  function apply() {
    onChange({ minAcc, maxLat, priority });
  }

  return (
    <div style={{ marginTop: 12 }}>
      <div>
        <label>Minimum Accuracy: {minAcc}%</label>
        <input type="range" min="50" max="100" value={minAcc} onChange={(e) => setMinAcc(+e.target.value)} />
      </div>
      <div>
        <label>Maximum Latency: {maxLat} ms</label>
        <input type="range" min="50" max="2000" step="10" value={maxLat} onChange={(e) => setMaxLat(+e.target.value)} />
      </div>
      <div>
        <label>Priority:</label>
        <div>
          <label><input type="radio" name="priority" value="accuracy-first" checked={priority==="accuracy-first"} onChange={(e)=>setPriority(e.target.value)} /> Accuracy-first</label>
          <label style={{ marginLeft: 8 }}><input type="radio" name="priority" value="balanced" checked={priority==="balanced"} onChange={(e)=>setPriority(e.target.value)} /> Balanced</label>
          <label style={{ marginLeft: 8 }}><input type="radio" name="priority" value="green-first" checked={priority==="green-first"} onChange={(e)=>setPriority(e.target.value)} /> Green-first</label>
        </div>
      </div>
      <div style={{ marginTop: 8 }}>
        <button onClick={apply}>Apply filters</button>
      </div>
    </div>
  );
}
