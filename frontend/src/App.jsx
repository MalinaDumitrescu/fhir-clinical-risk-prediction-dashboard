import { useEffect, useState } from "react";
import {
  getPatients,
  getPatient,
  getPrediction,
  exportRiskAssessment,
  getMetrics,
  getModelComparison,
  getModelDetails,
  getRiskAssessments,
  getRiskAssessmentsDownloadUrl,
} from "./api";

function App() {
  const [patients, setPatients] = useState([]);
  const [selectedPatientId, setSelectedPatientId] = useState(null);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [modelComparison, setModelComparison] = useState(null);
  const [modelDetails, setModelDetails] = useState(null);
  const [riskAssessments, setRiskAssessments] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    loadInitialData();
  }, []);

  async function loadInitialData() {
    loadPatients();
    loadMetrics();
    loadModelComparison();
    loadModelDetails();
    loadRiskAssessments();
  }

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

  async function loadRiskAssessments() {
    try {
      const data = await getRiskAssessments();
      setRiskAssessments(data);
    } catch (err) {
      console.log("Risk assessments not available yet.");
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
      loadRiskAssessments();
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
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">+</div>
          <div>
           <div className="brand-name">FHIR RISK</div>
           <div className="brand-subtitle">Clinical ML Prototype</div>
          </div>
        </div>

        <nav className="sidebar-nav" aria-label="Dashboard sections">
          <a className="nav-item active" href="#overview">
            <span className="nav-icon">⌂</span>
            Overview
          </a>
          <a className="nav-item" href="#patient">
            <span className="nav-icon">✚</span>
            Patient
          </a>
          <a className="nav-item" href="#risk">
            <span className="nav-icon">◒</span>
            Risk Analysis
          </a>
          <a className="nav-item" href="#performance">
            <span className="nav-icon">⌁</span>
            Model Performance
          </a>
          <a className="nav-item" href="#exports">
            <span className="nav-icon">⇩</span>
            FHIR Exports
          </a>
        </nav>

        <div className="sidebar-footer">
          <div className="system-status">
            <span className="status-dot"></span>
             Prototype API online
          </div>
          <div className="sidebar-note">
            Decision-support prototype
          </div>
        </div>
      </aside>

      <div className="workspace">
        <header className="topbar">
          <div>
            <div className="eyebrow">Clinical Decision Support</div>
            <h1>FHIR Clinical Risk Dashboard</h1>
            <p>
              First-24h clinical risk estimation with calibrated predictions,
              explainability, and interoperable FHIR export.
            </p>
          </div>

          <div className="topbar-status">
            <span className="status-dot"></span>
            Prototype operational
          </div>
        </header>

        {error && (
          <div className="error" role="alert">
            <strong>System message:</strong> {error}
          </div>
        )}

        <main id="overview" className="dashboard">
          <section className="panel patient-selector" id="patient">
            <div className="panel-heading">
              <div>
                <div className="section-kicker">Patient workspace</div>
                <h2>Select Patient</h2>
              </div>
              <div className="patient-count">
                <span>{patients.length}</span>
                patients loaded
              </div>
            </div>

            <label className="field-label" htmlFor="patient-select">
              Patient record
            </label>
            <select
              id="patient-select"
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

            <div className="small-info">
              Choose a record to refresh the patient summary and clinical risk
              estimate.
            </div>
          </section>

          <section className="panel patient-summary">
            <div className="panel-heading compact">
              <div>
                <div className="section-kicker">Patient record</div>
                <h2>Clinical Summary</h2>
              </div>
              {selectedPatient && (
                <span className="record-badge">FHIR-linked record</span>
              )}
            </div>

            {!selectedPatient && <p>No patient selected.</p>}

            {selectedPatient && (
              <>
                <div className="patient-identity">
                  <div className="avatar">
                    {String(selectedPatient.gender || "?").charAt(0).toUpperCase()}
                  </div>
                  <div>
                    <span className="identity-label">Patient ID</span>
                    <strong>{selectedPatient.patient_id}</strong>
                    <span className="identity-meta">
                      {selectedPatient.age === null ||
                      selectedPatient.age === undefined
                        ? "Age unknown"
                        : `${selectedPatient.age} years`}{" "}
                      · {selectedPatient.gender || "gender unknown"}
                    </span>
                  </div>
                </div>

                <div className="grid clinical-grid">
                  <Info
                    label="Medication requests 24h"
                    value={selectedPatient.medication_request_count}
                  />
                  <Info
                    label="Medication administrations 24h"
                    value={selectedPatient.medication_administration_count}
                  />
                  <Info
                    label="Procedures 24h"
                    value={selectedPatient.procedure_count}
                  />
                  <Info
                    label="ICU LOS days"
                    value={Number(selectedPatient.icu_los_days || 0).toFixed(2)}
                  />
                  <Info
                    label="Target: long ICU stay"
                    value={selectedPatient.target_long_icu_stay}
                  />
                </div>
              </>
            )}
          </section>

          <section className="panel risk-panel" id="risk">
            <div className="panel-heading compact">
              <div>
                <div className="section-kicker">Clinical prediction</div>
                <h2>Risk Assessment</h2>
              </div>
              <span className="clinical-tag">CDS</span>
            </div>

            {!prediction && <p>No prediction available.</p>}

            {prediction && (
              <>
  <div className={`risk-box ${getRiskClass(prediction.risk_level)}`}>
    <div className="risk-box-copy">
      <span className="risk-caption">
        Predicted probability of prolonged ICU stay
      </span>

      <div className="risk-percent">
        {prediction.risk_percent}%
      </div>

      <div className="risk-label">
        {prediction.predicted_long_icu_stay
          ? "PROLONGED ICU STAY PREDICTED"
          : "PROLONGED ICU STAY NOT PREDICTED"}
      </div>
    </div>

    <div className="risk-gauge" aria-hidden="true">
      <div className="risk-gauge-inner">
        {prediction.risk_percent}%
      </div>
    </div>
  </div>

  <div className="grid metrics-grid prediction-details-grid">
    <Info
      label="Probability"
      value={`${prediction.risk_percent}%`}
    />

    <Info
      label="Decision threshold"
      value={`${prediction.decision_threshold_percent}%`}
    />

    <Info
      label="Binary prediction"
      value={
        prediction.predicted_long_icu_stay
          ? "Positive"
          : "Negative"
      }
    />

    <Info
      label="Display probability band"
      value={prediction.risk_level.toUpperCase()}
    />
  </div>

  <p className="small-info">
    The positive/negative prediction is determined using the
    development-derived threshold saved with the trained model.
    Low/medium/high is only a descriptive probability band and is
    not a clinically validated decision category.
  </p>

  <div className="model-strip">
    <div>
      <span>Model</span>
      <strong>{prediction.model_name}</strong>
    </div>

    <div>
      <span>Calibrated</span>
      <strong>
        {prediction.calibrated ? "Yes" : "No"}
      </strong>
    </div>
  </div>

  {prediction.warning && (
    <p className="warning">
      <span>!</span>
      {prediction.warning}
    </p>
  )}

  <button
    className="primary-button"
    onClick={handleExportRiskAssessment}
  >
    Export FHIR RiskAssessment
  </button>
</>
            )}
          </section>

          <section className="panel explanation-panel">
            <div className="panel-heading compact">
              <div>
                <div className="section-kicker">Explainable AI</div>
                <h2>SHAP Explanation</h2>
              </div>
            </div>

            {!prediction && <p>No explanation available.</p>}

            {prediction &&
              Array.isArray(prediction.explanation) &&
              prediction.explanation.length > 0 && (
                <ol className="shap-list">
                  {prediction.explanation.map((item, index) => (
                    <li key={index} className="shap-item">
                      <div className="shap-rank">{index + 1}</div>
                      <div className="shap-content">
                        <strong>{item.feature}</strong>
                        <div className="shap-values">
                          <span>Value {round(item.value)}</span>
                          <span>SHAP {round(item.shap_value)}</span>
                          <span className="impact-pill">{item.impact}</span>
                        </div>
                      </div>
                    </li>
                  ))}
                </ol>
              )}

            {prediction && typeof prediction.explanation === "string" && (
              <p>{prediction.explanation}</p>
            )}
          </section>

          <section className="panel metrics-panel" id="performance">
            <div className="panel-heading compact">
              <div>
                <div className="section-kicker">Held-out test evaluation</div>
                <h2>Final Model Metrics</h2>
              </div>
              {metrics && (
                <span className="model-badge">
                  {metrics.model_name || "unknown"}
                </span>
              )}
            </div>

            {!metrics && <p>Train the model to see metrics.</p>}

            {metrics && (
              <>
                <div className="grid metrics-grid">
                  <Info label="Model" value={metrics.model_name} />
                  <Info
                    label="Calibrated"
                    value={metrics.calibrated ? "Yes" : "No"}
                  />
                  <Info label="ROC-AUC" value={round(metrics.roc_auc)} />
                  <Info
                    label="Average precision"
                    value={round(metrics.average_precision)}
                  />
                  <Info label="Brier score" value={round(metrics.brier_score)} />
                  <Info label="Accuracy" value={round(metrics.accuracy)} />
                  <Info
                    label="Balanced accuracy"
                    value={round(metrics.balanced_accuracy)}
                  />
                  <Info label="Precision" value={round(metrics.precision)} />
                  <Info label="Recall" value={round(metrics.recall)} />
                  <Info label="F1" value={round(metrics.f1)} />
                  <Info
                    label="Decision threshold"
                    value={round(metrics.decision_threshold)}
                  />
                  <Info
                    label="Evaluation split"
                    value={metrics.evaluation_split || "test"}
                  />
                </div>

                <p className="small-info metrics-note">
                  Final performance is reported on a held-out 20-patient test
                  partition. Because the test set contains only five positive
                  outcomes, these estimates have substantial statistical
                  uncertainty and are not evidence of clinical validity.
                </p>
              </>
            )}
          </section>

          <section className="panel exports-panel" id="exports">
            <div className="panel-heading compact">
              <div>
                <div className="section-kicker">Interoperability</div>
                <h2>Exported RiskAssessments</h2>
              </div>
              <a
                className="secondary-button"
                href={getRiskAssessmentsDownloadUrl()}
                download
              >
                Download All
              </a>
            </div>

            {riskAssessments.length === 0 && (
              <p>No assessments exported yet.</p>
            )}

            {riskAssessments.length > 0 && (
              <div className="assessment-list">
                {riskAssessments.map((ra) => (
                  <div className="assessment-row" key={ra.id}>
                    <div className="document-icon">FHIR</div>
                    <div>
                      <strong>{ra.id}</strong>
                      <span>{ra.subject.reference}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>

          <section className="panel model-details-panel">
            <div className="panel-heading compact">
              <div>
                <div className="section-kicker">Governance</div>
                <h2>Model Details</h2>
              </div>
            </div>

            {!modelDetails && <p>No model details available.</p>}

            {modelDetails && (
              <div className="details-list">
                <DetailRow label="Model" value={modelDetails.model_name} />
                <DetailRow label="Target" value={modelDetails.target} />
                <DetailRow
                  label="Target definition"
                  value={modelDetails.target_definition}
                />
                <DetailRow
                  label="Index admission"
                  value={modelDetails.index_admission_definition}
                />
                <DetailRow
                  label="Prediction window"
                  value={`first ${modelDetails.prediction_window_hours}h`}
                />
                <DetailRow
                  label="Leakage policy"
                  value={modelDetails.leakage_policy}
                />
                <DetailRow
                  label="Observation mapping"
                  value={modelDetails.observation_mapping}
                />
                <DetailRow label="Patients" value={modelDetails.n_patients} />
                <DetailRow
                  label="Development cohort"
                  value={modelDetails.n_development}
                />
                <DetailRow
                  label="Train / validation / test"
                  value={`${modelDetails.n_train} / ${modelDetails.n_validation} / ${modelDetails.n_test}`}
                />
                <DetailRow label="Features" value={modelDetails.n_features} />
                <DetailRow
                  label="Model selection split"
                  value={modelDetails.selection_split}
                />
                <DetailRow
                  label="Final evaluation split"
                  value={modelDetails.final_evaluation_split}
                />
                <DetailRow
                  label="Decision threshold"
                  value={round(modelDetails.decision_threshold)}
                />
                <DetailRow
                  label="Threshold selection"
                  value={modelDetails.threshold_selection_method}
                />
                <DetailRow
                  label="Threshold objective"
                  value={modelDetails.threshold_selection_metric}
                />
                <DetailRow
                  label="OOF balanced accuracy"
                  value={round(modelDetails.oof_balanced_accuracy_at_threshold)}
                />
                <DetailRow label="Random state" value={modelDetails.random_state} />
                <DetailRow label="Optuna trials" value={modelDetails.optuna_trials} />
                <DetailRow
                  label="Training date"
                  value={modelDetails.training_date_utc}
                />
              </div>
            )}
          </section>

          <section className="panel charts-panel full-width">
            <div className="panel-heading compact">
              <div>
                <div className="section-kicker">Performance analysis</div>
                <h2>Calibration and Performance Curves</h2>
              </div>
            </div>

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

          <section className="panel model-comparison-panel full-width">
            <div className="panel-heading compact">
              <div>
                <div className="section-kicker">Validation-stage benchmarking</div>
                <h2>Candidate Model Comparison</h2>
              </div>
              {modelComparison && (
                <span className="model-badge">
                  Best: {modelComparison.best_model_name}
                </span>
              )}
            </div>

            {!modelComparison && (
              <p>Run production training to see model comparison.</p>
            )}

            {modelComparison && (
              <>
                <p className="small-info comparison-rule">
                  Candidate models were compared on the validation partition only. Selection rule: {modelComparison.selection_metric}
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
                          <td>
                            <strong>{model.model_name}</strong>
                          </td>
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

function DetailRow({ label, value }) {
  return (
    <div className="detail-row">
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
          <line
            x1={padding}
            y1={height - padding}
            x2={width - padding}
            y2={height - padding}
            className="axis"
          />
          <line
            x1={padding}
            y1={padding}
            x2={padding}
            y2={height - padding}
            className="axis"
          />

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

          <text
            x={width / 2}
            y={height - 6}
            textAnchor="middle"
            className="axis-label"
          >
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