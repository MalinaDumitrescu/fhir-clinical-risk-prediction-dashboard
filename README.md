# FHIR Clinical Risk Prediction Dashboard

An educational medical informatics prototype that predicts the risk of long ICU stays using structured FHIR resources and machine learning. This system leverages the MIMIC-IV-on-FHIR demo dataset to provide clinical risk assessments with explainability features.

## ⚠ Important Disclaimer

**This system is NOT intended for real clinical decision-making.** It is an educational prototype designed to demonstrate how machine learning can be applied to FHIR-structured healthcare data. Any clinical use requires rigorous validation, regulatory approval, and clinical oversight.

---

##  Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Model Details](#model-details)
- [Development](#development)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

---

##  Project Overview

The FHIR Clinical Risk Prediction Dashboard is a full-stack application that:

1. **Processes FHIR Data**: Ingests structured patient data in FHIR format from the MIMIC-IV-on-FHIR dataset
2. **Extracts Features**: Builds predictive features from clinical encounters, medications, procedures, and diagnoses
3. **Predicts Risk**: Uses trained ML models to assess the likelihood of long ICU stays (≥ 3 days)
4. **Explains Predictions**: Provides SHAP-based explainability to understand model decisions
5. **Visualizes Results**: Presents clinical risk assessments through an interactive React dashboard

### Prediction Target

- **Target Variable**: `target_long_icu_stay` (binary classification)
- **Definition**: ICU length of stay ≥ 3 days
- **Prediction Time**: After the first 24 hours of encounter start
- **Data Source**: MIMIC-IV-on-FHIR demo dataset

---

## ✨ Features

### Backend Features
-  **FHIR Data Processing**: Parse and extract clinical data from FHIR resources
-  **Multiple ML Models**: XGBoost, LightGBM, and ensemble models for risk prediction
-  **Advanced Feature Engineering**: Temporal features, encounter characteristics, and medication profiles
-  **SHAP Explainability**: Understand individual predictions through feature importance
-  **Model Evaluation**: Comprehensive metrics, ROC curves, and model comparison
-  **Risk Assessment Tracking**: Save and retrieve patient risk assessments
-  **Fast API**: Modern async Python API with automatic OpenAPI documentation

### Frontend Features
-  **Interactive Dashboard**: React-based UI for risk assessment visualization
-  **Responsive Design**: Works on desktop and mobile devices
-  **Real-time Integration**: Communicates with backend API via REST endpoints
- ️ **Modern Build Tools**: Vite for fast development and optimized builds
---

## 🛠 Technology Stack

### Backend
- **Framework**: FastAPI 1.0+ with Uvicorn
- **ML Libraries**: scikit-learn, XGBoost, LightGBM, SHAP
- **Data Processing**: pandas, numpy
- **Experiment Tracking**: MLflow
- **Testing**: pytest, httpx

### Frontend
- **Framework**: React 19.2+
- **Build Tool**: Vite 8.1+
- **Linting**: Oxlint 1.71+
- **Development**: React with JSX

### Data
- **Dataset**: MIMIC-IV-on-FHIR demo
- **Format**: FHIR JSON resources

---

##  Prerequisites

- **Python**: 3.8+ (recommended 3.10+)
- **Node.js**: 16+ (for frontend development)
- **npm**: 8+ (for package management)
- **Git**: For version control

### Optional
- **Docker**: For containerized deployment
- **Jupyter**: For notebook-based exploration (optional)

---

##  Installation

### 1. Clone the Repository

```bash
git clone https://github.com/MalinaDumitrescu/healthcare-related-projects.git
cd clinical-risk-fhir-dashboard
```

### 2. Create Python Virtual Environment

```bash
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Backend Dependencies

```bash
pip install -r requirements.txt
```

**Note**: If `jupyter` package fails to install due to network issues, you can install without it:
```bash
pip install fastapi uvicorn pandas numpy scikit-learn joblib python-dateutil matplotlib optuna xgboost lightgbm shap mlflow pytest httpx
```

### 4. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

---

## ▶️ Running the Project

### Option 1: Run Backend and Frontend Separately

#### Start Backend Server

```bash
python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`
- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

#### Start Frontend Development Server

In a new terminal (with the same virtual environment activated):

```bash
cd frontend
npm run dev
```

The dashboard will be available at `http://localhost:5173`

### Option 2: Build Frontend for Production

```bash
cd frontend
npm run build
# Outputs to frontend/dist/
```

---

##  Project Structure

```
clinical-risk-fhir-dashboard/
├── backend/
│   └── app/
│       ├── main.py                    # FastAPI application entry point
│       ├── config.py                  # Configuration and paths
│       ├── features.py                # Feature engineering from FHIR data
│       ├── risk_assessment.py         # Risk assessment logic
│       ├── explainability.py          # SHAP explainability functions
│       ├── fhir_utils.py              # FHIR data parsing utilities
│       ├── evaluation_utils.py        # Model evaluation metrics
│       ├── train_model.py             # Model training pipeline
│       ├── train_advanced_models.py   # Advanced model configurations
│       └── train_production_models.py # Production model training
├── frontend/
│   ├── src/
│   │   ├── components/                # React components
│   │   ├── App.jsx                    # Main app component
│   │   └── main.jsx                   # Vite entry point
│   ├── index.html                     # HTML template
│   ├── package.json                   # Frontend dependencies
│   ├── vite.config.js                 # Vite configuration
│   └── README.md                      # Frontend-specific docs
├── notebooks/                         # Jupyter notebooks for exploration
├── data/                              # Data storage
├── artifacts/                         # Generated artifacts
├── mlruns/                            # MLflow experiment tracking
├── tests/                             # Test suite
├── requirements.txt                   # Python dependencies
├── model_card.md                      # Detailed model documentation
├── clinical_limitations.md            # Clinical considerations
└── README.md                          # This file
```

### Key Files

- **`backend/app/main.py`**: FastAPI application with API endpoints
- **`backend/app/features.py`**: Feature engineering from FHIR encounters
- **`backend/app/risk_assessment.py`**: Risk prediction logic
- **`backend/app/explainability.py`**: SHAP-based model explanations
- **`frontend/src/App.jsx`**: Main React dashboard component
- **`model_card.md`**: Comprehensive model documentation
- **`clinical_limitations.md`**: Clinical and ethical considerations

---

##  API Documentation

### Main Endpoints

#### Health Check
```http
GET /health
```
Returns API status and version.

#### Get Risk Assessment
```http
POST /risk-assess
Content-Type: application/json

{
  "patient_id": "string",
  "encounter_id": "string",
  "features": {...}
}
```

#### Get Model Explanation
```http
POST /explain
Content-Type: application/json

{
  "features": {...}
}
```

#### Get Model Metrics
```http
GET /metrics
```
Returns model performance metrics and statistics.

#### Get Model Comparison
```http
GET /model-comparison
```
Returns comparison of different models.

### Interactive API Docs

Once the backend is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

Both provide interactive API exploration and testing.

---

##  Model Details

### Models Trained

The system trains and compares multiple models:

1. **XGBoost**: Gradient boosting with optimized hyperparameters
2. **LightGBM**: Lightweight gradient boosting framework
3. **Ensemble**: Combined predictions from multiple models

### Features Used

Features are extracted from FHIR resources in the first 24 hours:

- **Patient Demographics**: Age, gender
- **Encounter Characteristics**: ICU admission details, admission type
- **Clinical History**: Conditions, diagnoses within 24 hours
- **Medications**: Count and types of medications administered
- **Procedures**: Procedure codes and counts
- **Vital Signs & Labs**: Aggregated values within first 24 hours

### Model Performance

See `model_card.md` for:
- Detailed performance metrics
- Feature importance analysis
- Model validation approach
- Limitations and considerations

---

## Development

### Code Style

The project follows:
- **Python**: PEP 8 style guide
- **JavaScript**: Oxlint rules (see `.oxlintrc.json`)
- **Components**: Reusable, well-documented modules

### Running Linter

#### Backend
```bash
# Install flake8 or similar (optional)
pip install flake8
flake8 backend/
```

#### Frontend
```bash
cd frontend
npm run lint
```

### Building for Production

#### Backend
No special build required; deploy the Python environment and code.

#### Frontend
```bash
cd frontend
npm run build
# Outputs optimized build to frontend/dist/
```

---

##  Testing

### Run Tests

```bash
pytest tests/ -v
```

### Run Specific Test File

```bash
pytest tests/test_features.py -v
```

### Test Coverage

```bash
pytest tests/ --cov=backend --cov-report=html
```

### Test Structure

Tests are organized by module:
- `tests/test_features.py`: Feature engineering tests
- `tests/test_models.py`: Model training and prediction
- `tests/test_fhir_utils.py`: FHIR data processing
- `tests/test_api.py`: API endpoint tests

---

##  Machine Learning Workflow

### 1. Data Preparation
```bash
python backend/app/features.py
# Extracts features from FHIR data and saves to CSV
```

### 2. Model Training
```bash
python backend/app/train_model.py
# Trains baseline model with default hyperparameters
```

### 3. Advanced Training
```bash
python backend/app/train_advanced_models.py
# Hyperparameter tuning with Optuna
```

### 4. Production Models
```bash
python backend/app/train_production_models.py
# Trains final production-ready models
```

### 5. Model Evaluation
```bash
python check_metrics.py
# Displays model performance metrics
```

---

##  Configuration

Configuration is managed in `backend/app/config.py`:

```python
# Model path
MODEL_PATH = Path("artifacts/model.joblib")

# Features CSV path
FEATURES_CSV = Path("data/features.csv")

# Metrics and comparison paths
METRICS_PATH = Path("artifacts/metrics.json")
MODEL_COMPARISON_PATH = Path("artifacts/model_comparison.json")

# Risk assessments storage
RISK_ASSESSMENTS_PATH = Path("data/risk_assessments")
```

Modify these paths as needed for your deployment environment.

---

##  Docker Support (Optional)

To containerize the application:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY backend/ backend/
COPY frontend/dist/ frontend/dist/

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t fhir-dashboard .
docker run -p 8000:8000 fhir-dashboard
```

---

##  Additional Resources

### Documentation
- **Model Card**: See `model_card.md` for comprehensive model documentation
- **Clinical Limitations**: See `clinical_limitations.md` for clinical considerations
- **Dataset Info**: See `dataset_explained.txt` for data structure details

### External References
- [FHIR Standard](https://www.hl7.org/fhir/)
- [MIMIC-IV Database](https://mimic.physionet.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [SHAP Documentation](https://shap.readthedocs.io/)

---

##  Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes and test thoroughly
4. Commit with clear messages: `git commit -m "Add feature description"`
5. Push to your branch and open a Pull Request

### Development Guidelines
- Write tests for new features
- Update documentation as needed
- Follow existing code style
- Test API endpoints with provided tools
- Ensure no breaking changes to existing APIs

---

##  License

This project is provided as an educational prototype. See the `licenses/` directory for details on dependencies and licensing.

---

##  Author

**Malina Dumitrescu**

This project is part of the healthcare-related-projects repository. For questions or issues, please open an issue on the repository.

---

## ⚠️ Clinical and Ethical Considerations

**This system is for educational purposes only.**

### Important Limitations:
- Not validated for clinical decision-making
- MIMIC data contains de-identified historical records
- Model may contain biases present in training data
- Predictions should never replace clinical judgment
- Regulatory approval required for any real-world use

See `clinical_limitations.md` for detailed clinical and ethical considerations.

---

##  Version History

- **v2.0.0** (Current): Full-stack dashboard with explainability
- **v1.0**: Initial ML model prototype

---

##  Support

For issues or questions:
1. Check existing issues on GitHub
2. Review documentation files
3. Consult the model card for technical details
4. Open a new issue with detailed description

---

**Last Updated**: August 2026
**Status**: Educational/Prototype
