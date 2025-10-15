import axios from "axios";

const API_BASE = "http://localhost:8000";

export async function classifyTask(text) {
  const resp = await axios.post(`${API_BASE}/classify`, { text, top_k: 1 });
  return resp.data;
}

export async function getRecommendations(payload) {
  const resp = await axios.post(`${API_BASE}/recommend`, payload);
  return resp.data;
}

export async function setBaseline(task, model) {
  const resp = await axios.post(`${API_BASE}/baseline`, { task, model });
  return resp.data;
}

export async function getBaseline(task) {
  const resp = await axios.get(`${API_BASE}/baseline`, { params: { task } });
  return resp.data;
}

export async function getModelsForTask(task) {
  const resp = await axios.get(`${API_BASE}/models`, { params: { task } });
  return resp.data;
}
