const API_BASE =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export async function getPatients() {
  const response = await fetch(`${API_BASE}/patients`);
  if (!response.ok) throw new Error("Could not load patients");
  return response.json();
}

export async function getPatient(patientId) {
  const response = await fetch(`${API_BASE}/patients/${patientId}`);
  if (!response.ok) throw new Error("Could not load patient");
  return response.json();
}

export async function getPrediction(patientId) {
  const response = await fetch(`${API_BASE}/patients/${patientId}/predict`);
  if (!response.ok) throw new Error("Could not load prediction");
  return response.json();
}

export async function exportRiskAssessment(patientId) {
  const response = await fetch(`${API_BASE}/patients/${patientId}/risk-assessment`, {
    method: "POST",
  });
  if (!response.ok) throw new Error("Could not export RiskAssessment");
  return response.json();
}

export async function getMetrics() {
  const response = await fetch(`${API_BASE}/metrics`);
  if (!response.ok) throw new Error("Could not load metrics");
  return response.json();
}

export async function getModelComparison() {
  const response = await fetch(`${API_BASE}/models/comparison`);
  if (!response.ok) throw new Error("Could not load model comparison");
  return response.json();
}

export async function getModelDetails() {
  const response = await fetch(`${API_BASE}/models/details`);
  if (!response.ok) throw new Error("Could not load model details");
  return response.json();
}

export async function getRiskAssessments() {
  const response = await fetch(`${API_BASE}/risk-assessments`);
  if (!response.ok) throw new Error("Could not load risk assessments");
  return response.json();
}

export function getRiskAssessmentsDownloadUrl() {
  return `${API_BASE}/risk-assessments/download`;
}
