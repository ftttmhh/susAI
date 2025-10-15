import React from "react";

export default function ResultsTable({ recs }) {
  if (!recs || recs.length === 0) return null;
  return (
    <div style={{ marginTop: 16 }}>
      <h3>Top Recommendations</h3>
      <table border="1" cellPadding="6" cellSpacing="0">
        <thead>
          <tr>
            <th>Model</th><th>Provider</th><th>Accuracy</th><th>Latency (ms)</th><th>Energy Wh/1k</th><th>CO₂ kg/1k</th><th>Energy saved (Wh/1k)</th>
          </tr>
        </thead>
        <tbody>
          {recs.map((r, i) => (
            <tr key={i}>
              <td>{r.model}</td>
              <td>{r.provider}</td>
              <td>{r.accuracy}</td>
              <td>{r.latency_ms}</td>
              <td>{(r.gpu_energy ?? r['gpu_energy']) ?? 'N/A'}</td>
              <td>{(r.co2_kg_per_1k ?? r['co2_kg_per_1k']) ?? 'N/A'}</td>
              <td title={typeof (r.energy_delta_wh_per_1k) !== 'undefined' ? `Delta (candidate - baseline): ${r.energy_delta_wh_per_1k}` : ''}>{(r.energy_saved_wh_per_1k ?? r['energy_saved_wh_per_1k'] ?? 0)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
