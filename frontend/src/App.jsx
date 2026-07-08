import { useEffect, useState } from "react";
import {
  getPatients,
  getPatient,
  getPrediction,
  exportRiskAssessment,
  getMetrics,
  getModelComparison,
  getModelDetails,
} from "./api";

function App() {
  const [patients, setPatients] = useState([]);
  const [selectedPatientId, setSelectedPatientId] = useState(null);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [modelComparison, setModelComparison] = useState(null);
  const [modelDetails, setModelDetails] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    loadPatients();
    loadMetrics();
    loadModelComparison();
    loadModelDetails();
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

  async function loadModelDetails() {
    try {
      const data = await getModelDetails();
      setModelDetails(data);
    } catch (err) {
      console.log("Model details not available yet.");
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
    if (level === "high") return "risk-high";
    if (level === "medium") return "risk-medium";
    return "risk-low";
  }

  return (
    <div className="app">
      <header>
        <h1>FHIR Clinical Risk Dashboard</h1>
        <p>
          Leakage-safe first-24h FHIR features, Optuna tuning, calibrated risk
          scores, SHAP explanations, and FHIR RiskAssessment export.
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
              <Info label="Conditions 24h" value={selectedPatient.condition_count} />
              <Info
                label="Medication events 24h"
                value={selectedPatient.medication_event_count}
              />
              <Info label="Procedures 24h" value={selectedPatient.procedure_count} />
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

              <div className="small-info">
                Model: {prediction.model_name} | calibrated:{" "}
                {String(prediction.calibrated)}
              </div>

              <p className="warning">{prediction.warning}</p>

              <button onClick={handleExportRiskAssessment}>
                Export FHIR RiskAssessment
              </button>
            </>
          )}
        </section>

        <section className="panel">
          <h2>SHAP Explanation</h2>

          {!prediction && <p>No explanation available.</p>}

          {prediction && prediction.explanation.length > 0 && (
            <ol>
              {prediction.explanation.map((item, index) => (
                <li key={index}>
                  <strong>{item.feature}</strong>
                  <br />
                  value: {round(item.value)} | SHAP: {round(item.shap_value)} |{" "}
                  {item.impact}
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
              <Info label="Calibrated" value={String(metrics.calibrated)} />
              <Info label="Accuracy" value={round(metrics.accuracy)} />
              <Info label="ROC-AUC" value={round(metrics.roc_auc)} />
              <Info label="Average precision" value={round(metrics.average_precision)} />
              <Info label="Brier score" value={round(metrics.brier_score)} />
              <Info label="Balanced accuracy" value={round(metrics.balanced_accuracy)} />
              <Info label="Recall" value={round(metrics.recall)} />
              <Info label="F1" value={round(metrics.f1)} />
            </div>
          )}
        </section>

        <section className="panel">
          <h2>Model Details</h2>

          {!modelDetails && <p>No model details available.</p>}

          {modelDetails && (
            <div className="details-list">
              <p><strong>Target:</strong> {modelDetails.target}</p>
              <p><strong>Target definition:</strong> {modelDetails.target_definition}</p>
              <p><strong>Prediction window:</strong> first {modelDetails.prediction_window_hours}h</p>
              <p><strong>Leakage policy:</strong> {modelDetails.leakage_policy}</p>
              <p><strong>Patients:</strong> {modelDetails.n_patients}</p>
              <p><strong>Features:</strong> {modelDetails.n_features}</p>
              <p><strong>Training date:</strong> {modelDetails.training_date_utc}</p>
            </div>
          )}
        </section>

        <section className="panel patient-list">
          <h2>Calibration and Performance Curves</h2>

          {!metrics && <p>Train the model to see curves.</p>}

          {metrics && (
            <div className="charts-grid">
              <CurveChart
                title="ROC Curve"
                data={metrics.roc_curve || []}
                xKey="fpr"
                yKey="tpr"
                xLabel="False positive rate"
                yLabel="True positive rate"
              />

              <CurveChart
                title="Precision-Recall Curve"
                data={metrics.pr_curve || []}
                xKey="recall"
                yKey="precision"
                xLabel="Recall"
                yLabel="Precision"
              />

              <CurveChart
                title="Calibration Curve"
                data={metrics.calibration_curve || []}
                xKey="mean_predicted_probability"
                yKey="fraction_of_positives"
                xLabel="Predicted probability"
                yLabel="Observed frequency"
                diagonal
              />
            </div>
          )}
        </section>

        <section className="panel patient-list">
          <h2>Model Comparison</h2>

          {!modelComparison && (
            <p>Run production training to see model comparison.</p>
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
                      <th>Brier</th>
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
                        <td>{round(model.brier_score)}</td>
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

function CurveChart({ title, data, xKey, yKey, xLabel, yLabel, diagonal }) {
  const width = 360;
  const height = 240;
  const padding = 38;

  const safeData = data.filter(
    (point) =>
      point[xKey] !== null &&
      point[yKey] !== null &&
      point[xKey] !== undefined &&
      point[yKey] !== undefined
  );

  function scaleX(value) {
    return padding + Number(value) * (width - 2 * padding);
  }

  function scaleY(value) {
    return height - padding - Number(value) * (height - 2 * padding);
  }

  const points = safeData
    .map((point) => `${scaleX(point[xKey])},${scaleY(point[yKey])}`)
    .join(" ");

  return (
    <div className="curve-card">
      <h3>{title}</h3>

      {safeData.length === 0 && <p>No curve data available.</p>}

      {safeData.length > 0 && (
        <svg viewBox={`0 0 ${width} ${height}`} className="curve-svg">
          <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} className="axis" />
          <line x1={padding} y1={padding} x2={padding} y2={height - padding} className="axis" />

          {diagonal && (
            <line
              x1={padding}
              y1={height - padding}
              x2={width - padding}
              y2={padding}
              className="diagonal"
            />
          )}

          <polyline points={points} className="curve-line" />

          <text x={width / 2} y={height - 6} textAnchor="middle" className="axis-label">
            {xLabel}
          </text>

          <text
            x="12"
            y={height / 2}
            textAnchor="middle"
            className="axis-label"
            transform={`rotate(-90 12 ${height / 2})`}
          >
            {yLabel}
          </text>
        </svg>
      )}
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