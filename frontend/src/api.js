const API_BASE = "http://127.0.0.1:8000";

export async function getPatients() {
  const response = await fetch(`${API_BASE}/patients`);

  if (!response.ok) {
    throw new Error("Could not load patients");
  }

  return response.json();
}

export async function getPatient(patientId) {
  const response = await fetch(`${API_BASE}/patients/${patientId}`);

  if (!response.ok) {
    throw new Error("Could not load patient");
  }

  return response.json();
}

export async function getPrediction(patientId) {
  const response = await fetch(`${API_BASE}/patients/${patientId}/predict`);

  if (!response.ok) {
    throw new Error("Could not load prediction");
  }

  return response.json();
}

export async function exportRiskAssessment(patientId) {
  const response = await fetch(
    `${API_BASE}/patients/${patientId}/risk-assessment`,
    {
      method: "POST",
    }
  );

  if (!response.ok) {
    throw new Error("Could not export RiskAssessment");
  }

  return response.json();
}

export async function getMetrics() {
  const response = await fetch(`${API_BASE}/metrics`);

  if (!response.ok) {
    throw new Error("Could not load metrics");
  }

  return response.json();
}