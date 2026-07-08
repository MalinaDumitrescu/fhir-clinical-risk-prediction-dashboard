import { useEffect, useState } from "react";
import {
  getPatients,
  getPatient,
  getPrediction,
  exportRiskAssessment,
  getMetrics,
  getModelComparison,
} from "./api";

function App() {
  const [patients, setPatients] = useState([]);
  const [selectedPatientId, setSelectedPatientId] = useState(null);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [modelComparison, setModelComparison] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    loadPatients();
    loadMetrics();
    loadModelComparison();
  }, []);

  async function loadPatients() {
    try {
      const data = await getPatients();
      setPatients(data);

      if (data.length > 0) {
        setSelectedPatientId(data[0].patient_id);
      }
    } catch (err) {
      setError(err.message);
    }
  }

  async function loadMetrics() {
    try {
      const data = await getMetrics();
      setMetrics(data);
    } catch (err) {
      console.log("Metrics not available yet.");
    }
  }

  async function loadModelComparison() {
    try {
      const data = await getModelComparison();
      setModelComparison(data);
    } catch (err) {
      console.log("Model comparison not available yet.");
    }
  }

  useEffect(() => {
    if (selectedPatientId) {
      loadSelectedPatient(selectedPatientId);
      loadPrediction(selectedPatientId);
    }
  }, [selectedPatientId]);

  async function loadSelectedPatient(patientId) {
    try {
      const data = await getPatient(patientId);
      setSelectedPatient(data);
    } catch (err) {
      setError(err.message);
    }
  }

  async function loadPrediction(patientId) {
    try {
      const data = await getPrediction(patientId);
      setPrediction(data);
    } catch (err) {
      setError(err.message);
    }
  }

  async function handleExportRiskAssessment() {
    try {
      const data = await exportRiskAssessment(selectedPatientId);
      alert(`RiskAssessment exported: ${data.id}`);
    } catch (err) {
      setError(err.message);
    }
  }

  function getRiskClass(level) {
    if (level === "high") {
      return "risk-high";
    }

    if (level === "medium") {
      return "risk-medium";
    }

    return "risk-low";
  }

  return (
    <div className="app">
      <header>
        <h1>FHIR Clinical Risk Dashboard</h1>
        <p>
          Educational prototype using MIMIC-IV-on-FHIR data, Optuna, boosting,
          and ensemble models.
        </p>
      </header>

      {error && <div className="error">{error}</div>}

      <main>
        <section className="panel patient-list">
          <h2>Patients</h2>

          <select
            value={selectedPatientId || ""}
            onChange={(event) => setSelectedPatientId(event.target.value)}
          >
            {patients.map((patient) => (
              <option key={patient.patient_id} value={patient.patient_id}>
                {patient.patient_id} | age {patient.age || "unknown"} |{" "}
                {patient.gender}
              </option>
            ))}
          </select>

          <div className="small-info">Loaded patients: {patients.length}</div>
        </section>

        <section className="panel">
          <h2>Patient Summary</h2>

          {!selectedPatient && <p>No patient selected.</p>}

          {selectedPatient && (
            <div className="grid">
              <Info label="Patient ID" value={selectedPatient.patient_id} />
              <Info label="Gender" value={selectedPatient.gender} />
              <Info label="Age" value={selectedPatient.age} />
              <Info label="Conditions" value={selectedPatient.condition_count} />
              <Info
                label="Medication events"
                value={selectedPatient.medication_event_count}
              />
              <Info label="Procedures" value={selectedPatient.procedure_count} />
              <Info
                label="ICU LOS days"
                value={Number(selectedPatient.icu_los_days || 0).toFixed(2)}
              />
              <Info
                label="Target: long ICU stay"
                value={selectedPatient.target_long_icu_stay}
              />
            </div>
          )}
        </section>

        <section className="panel">
          <h2>Risk Prediction</h2>

          {!prediction && <p>No prediction available.</p>}

          {prediction && (
            <>
              <div className={`risk-box ${getRiskClass(prediction.risk_level)}`}>
                <div className="risk-percent">{prediction.risk_percent}%</div>
                <div className="risk-label">
                  {prediction.risk_level.toUpperCase()} RISK
                </div>
              </div>

              <p className="warning">{prediction.warning}</p>

              <button onClick={handleExportRiskAssessment}>
                Export FHIR RiskAssessment
              </button>
            </>
          )}
        </section>

        <section className="panel">
          <h2>Top Explanation Signals</h2>

          {!prediction && <p>No explanation available.</p>}

          {prediction && prediction.explanation.length === 0 && (
            <p>No explanation signals available.</p>
          )}

          {prediction && prediction.explanation.length > 0 && (
            <ol>
              {prediction.explanation.map((item, index) => (
                <li key={index}>
                  <strong>{item.feature}</strong>
                  <br />
                  value: {Number(item.value).toFixed(3)} | importance:{" "}
                  {Number(item.importance).toFixed(4)}
                </li>
              ))}
            </ol>
          )}
        </section>

        <section className="panel">
          <h2>Model Metrics</h2>

          {!metrics && <p>Train the model to see metrics.</p>}

          {metrics && (
            <div className="grid">
              <Info label="Best model" value={metrics.model_name || "unknown"} />
              <Info label="Accuracy" value={round(metrics.accuracy)} />
              <Info label="ROC-AUC" value={round(metrics.roc_auc)} />
              <Info
                label="Average precision"
                value={round(metrics.average_precision)}
              />
              <Info
                label="Balanced accuracy"
                value={round(metrics.balanced_accuracy)}
              />
              <Info label="Recall" value={round(metrics.recall)} />
              <Info label="F1" value={round(metrics.f1)} />
            </div>
          )}
        </section>

        <section className="panel patient-list">
          <h2>Model Comparison</h2>

          {!modelComparison && (
            <p>Run advanced training to see model comparison.</p>
          )}

          {modelComparison && (
            <>
              <p>
                <strong>Best model:</strong> {modelComparison.best_model_name}
              </p>

              <p className="small-info">
                Selection rule: {modelComparison.selection_metric}
              </p>

              <div className="table-wrapper">
                <table>
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th>ROC-AUC</th>
                      <th>Avg Precision</th>
                      <th>Balanced Acc.</th>
                      <th>Recall</th>
                      <th>F1</th>
                    </tr>
                  </thead>

                  <tbody>
                    {modelComparison.models.map((model) => (
                      <tr key={model.model_name}>
                        <td>{model.model_name}</td>
                        <td>{round(model.roc_auc)}</td>
                        <td>{round(model.average_precision)}</td>
                        <td>{round(model.balanced_accuracy)}</td>
                        <td>{round(model.recall)}</td>
                        <td>{round(model.f1)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </section>
      </main>
    </div>
  );
}

function Info({ label, value }) {
  return (
    <div className="info-card">
      <span>{label}</span>
      <strong>{value === null || value === undefined ? "unknown" : value}</strong>
    </div>
  );
}

function round(value) {
  if (value === null || value === undefined) {
    return "N/A";
  }

  return Number(value).toFixed(3);
}

export default App;