import React, { useState } from "react";
import { classifyTask } from "../api";

export default function TaskInput({ onTaskConfirmed }) {
  const [text, setText] = useState("");
  const [mapping, setMapping] = useState(null);

  async function handleMap() {
    if (!text) return;
    try {
      const res = await classifyTask(text);
      setMapping(res);
      onTaskConfirmed(res.category);
    } catch (err) {
      console.error(err);
      alert("Could not map task: " + (err?.response?.data?.detail || err.message));
    }
  }

  return (
    <div>
      <label>Describe your task</label>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="e.g., Summarize legal contracts in plain English"
        rows={3}
        style={{ width: "100%" }}
      />
      <div style={{ marginTop: 8 }}>
        <button onClick={handleMap}>Map task → category</button>
        {mapping && (
          <div style={{ marginTop: 8 }}>
            <strong>Mapped to:</strong> {mapping.category} (score: {mapping.score.toFixed(3)})
          </div>
        )}
      </div>
    </div>
  );
}
