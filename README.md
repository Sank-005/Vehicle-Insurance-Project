# 🚗 Vehicle Insurance Prediction — End-to-End MLOps Pipeline

> **An end-to-end production-oriented Machine Learning project that automates the complete lifecycle of a Vehicle Insurance Prediction model — from data ingestion and validation to model training, evaluation, versioning, deployment, and prediction through a web application.**

---

## 🎯 Project Overview

This project implements a complete **MLOps pipeline** for predicting whether a vehicle insurance customer is likely to show interest in purchasing vehicle insurance.

Rather than building only a machine learning model, the project focuses on the **entire ML lifecycle**:

**Data → Validation → Transformation → Training → Evaluation → Model Registry → Deployment → Prediction**

The system uses **MongoDB Atlas** for data storage, **AWS S3** for model storage/versioning, **FastAPI** for serving predictions, and **Docker + AWS EC2 + ECR + GitHub Actions** for deployment and CI/CD.

---

## ✨ Key Highlights

* 🧩 **Modular MLOps architecture** with separate components for every ML lifecycle stage
* 🍃 **MongoDB Atlas** integration for cloud-based dataset storage and ingestion
* 🔍 Automated **data validation** using schema configuration
* ⚙️ Dedicated **data transformation & preprocessing** pipeline
* 🤖 Automated **model training and evaluation**
* ☁️ **AWS S3 Model Registry** for storing trained models
* 📊 Configurable model evaluation threshold for deciding whether a new model should replace the existing model
* 🚀 **FastAPI prediction service** with an interactive web interface
* 🐳 **Dockerized application**
* 🔄 **CI/CD using GitHub Actions**
* 📦 **AWS ECR** for Docker image storage
* 💻 **AWS EC2** for application deployment
* 🏃 **Self-hosted GitHub Actions runner** on EC2
* 📝 Centralized logging and custom exception handling
* 🔐 Environment-based configuration for database and cloud credentials

---

# 🏗️ System Architecture

The overall architecture of the project can be summarized as:

```mermaid
flowchart LR
    A[Raw Dataset] --> B[MongoDB Atlas]

    B --> C[Data Ingestion]
    C --> D[Data Validation]
    D --> E[Data Transformation]
    E --> F[Model Training]
    F --> G[Model Evaluation]

    G --> H{Model Better?}

    H -->|Yes| I[AWS S3<br/>Model Registry]
    H -->|No| J[Keep Existing Model]

    I --> K[Prediction Pipeline]

    K --> L[FastAPI Application]
    L --> M[Web Interface]
    M --> N[Prediction]

    O[GitHub Repository] --> P[GitHub Actions]
    P --> Q[Self Hosted EC2 Runner]
    Q --> R[Docker Build]
    R --> S[AWS ECR]
    S --> T[AWS EC2]
    T --> L
```

---

# 🔄 End-to-End ML Pipeline

The core machine learning workflow follows a modular pipeline:

```mermaid
flowchart TD
    A[MongoDB Atlas] --> B[Data Ingestion]

    B --> C[Data Validation]
    C --> D[Data Transformation]
    D --> E[Model Trainer]

    E --> F[Model Evaluation]

    F --> G{Performance Threshold}

    G -->|Pass| H[Model Pusher]
    G -->|Fail| I[Reject Model]

    H --> J[AWS S3 Model Registry]
    J --> K[Prediction Pipeline]

    K --> L[Vehicle Insurance Prediction]
```

Each stage produces structured **configuration objects and artifacts**, making the pipeline easier to debug, maintain, and extend.

---

# 🗂️ Project Structure

```text
Vehicle-Insurance-Project/
│
├── .github/
│   └── workflows/
│       └── aws.yaml
│
├── artifacts/
│
├── notebook/
│   ├── EDA/
│   ├── Feature Engineering/
│   └── mongoDB_demo.ipynb
│
├── src/
│   │
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   └── model_pusher.py
│   │
│   ├── configuration/
│   │   ├── mongo_db_connection.py
│   │   └── aws_connection.py
│   │
│   ├── data_access/
│   │   └── proj1_data.py
│   │
│   ├── entity/
│   │   ├── config_entity.py
│   │   ├── artifact_entity.py
│   │   ├── estimator.py
│   │   └── s3_estimator.py
│   │
│   ├── pipline/
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   │
│   ├── aws_storage/
│   │
│   ├── constants/
│   │   └── __init__.py
│   │
│   ├── exception.py
│   ├── logger.py
│   └── utils/
│       └── main_utils.py
│
├── static/
│   └── css/
│
├── templates/
│   └── vehicledata.html
│
├── app.py
├── demo.py
├── config/
│   └── schema.yaml
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
├── setup.py
├── pyproject.toml
├── template.py
└── README.md
```

---

# 🧠 ML Pipeline Components

## 1. Data Ingestion

The pipeline retrieves the dataset stored in **MongoDB Atlas**.

```mermaid
flowchart LR
    A[MongoDB Atlas] --> B[MongoDB Connection]
    B --> C[Fetch Documents]
    C --> D[Convert Key-Value Data]
    D --> E[Pandas DataFrame]
    E --> F[Data Ingestion Artifact]
```

The ingestion component:

* Establishes the MongoDB connection
* Fetches documents from the collection
* Converts MongoDB records into a Pandas DataFrame
* Stores the resulting artifact for downstream components

---

## 2. Data Validation

Before training, the dataset is validated against a predefined schema.

```mermaid
flowchart TD
    A[Ingested Dataset] --> B[Schema Configuration]
    B --> C{Validation}
    C -->|Valid| D[Validated Dataset]
    C -->|Invalid| E[Validation Error]
```

The schema configuration contains information about the expected dataset structure and features.

This prevents unexpected changes in the incoming data from silently propagating into the training pipeline.

---

## 3. Data Transformation

The validated dataset is transformed into a format suitable for model training.

```mermaid
flowchart LR
    A[Validated Data] --> B[Preprocessing]
    B --> C[Feature Transformation]
    C --> D[Transformed Train Data]
    C --> E[Transformed Test Data]
```

The transformation stage encapsulates preprocessing logic and generates the transformed datasets required by the model trainer.

---

## 4. Model Training

The transformed dataset is passed to the model training component.

```mermaid
flowchart TD
    A[Train Dataset] --> B[Model Trainer]
    B --> C[Trained Model]
    C --> D[Model Artifact]
```

The training component is isolated from the rest of the pipeline, allowing the underlying estimator/model to be modified without changing the overall pipeline architecture.

---

## 5. Model Evaluation

A trained model is evaluated before being pushed to the model registry.

```mermaid
flowchart TD
    A[Trained Model] --> C[Model Evaluation]
    B[Evaluation Dataset] --> C

    C --> D{Performance Comparison}

    D -->|Improvement ≥ Threshold| E[Accept Model]
    D -->|Below Threshold| F[Reject Model]
```

The project defines a configurable model evaluation threshold:

```text
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE = 0.02
```

This allows the pipeline to determine whether a newly trained model provides sufficient improvement before replacing the existing model.

---

# ☁️ AWS S3 Model Registry

Accepted models are stored in an **AWS S3 bucket**.

```mermaid
flowchart LR
    A[Model Evaluation] --> B{Model Accepted?}
    B -->|Yes| C[Model Pusher]
    C --> D[AWS S3]
    D --> E[Model Registry]

    E --> F[Prediction Pipeline]
```

The S3 integration provides centralized storage for trained models and allows the prediction pipeline to retrieve the appropriate model.

The project uses:

```text
MODEL_BUCKET_NAME = my-model-mlopsproj
MODEL_PUSHER_S3_KEY = model-registry
```

---

# 🔮 Prediction Pipeline

Once the model has been trained and stored, the prediction pipeline loads the model and performs inference on user-provided vehicle information.

```mermaid
sequenceDiagram
    participant U as User
    participant F as FastAPI
    participant P as Prediction Pipeline
    participant M as Model
    participant R as S3 Model Registry

    U->>F: Submit vehicle details
    F->>P: Create prediction input
    P->>R: Retrieve trained model
    R-->>P: Model artifact
    P->>M: Perform inference
    M-->>P: Prediction
    P-->>F: Response-Yes / Response-No
    F-->>U: Display prediction
```

---

# 🌐 FastAPI Web Application

The trained model is exposed through a **FastAPI application**.

The application provides:

* Vehicle information input form
* Prediction endpoint
* Model training endpoint
* Prediction result rendering
* Static CSS assets
* Jinja2 HTML templates

Example application flow:

```text
User
  ↓
Vehicle Information Form
  ↓
FastAPI
  ↓
Prediction Pipeline
  ↓
Trained Model
  ↓
Prediction
  ↓
Response-Yes / Response-No
```

The application runs locally on port `5000` during development.

---

# 🐳 Dockerization

The application is containerized using Docker.

```mermaid
flowchart LR
    A[Source Code] --> B[Dockerfile]
    B --> C[Docker Image]
    C --> D[AWS ECR]
    D --> E[AWS EC2]
    E --> F[Running Container]
    F --> G[FastAPI Application]
```

Docker provides a consistent runtime environment between development and deployment.

---

# 🔄 CI/CD Pipeline

The project implements an automated CI/CD workflow using **GitHub Actions**.

```mermaid
flowchart LR
    A[Developer] --> B[Git Push]
    B --> C[GitHub Repository]
    C --> D[GitHub Actions]
    D --> E[Self Hosted Runner]
    E --> F[Docker Build]
    F --> G[AWS ECR]
    G --> H[AWS EC2]
    H --> I[Deploy Application]
```

### Deployment Infrastructure

| Component          | Purpose                      |
| ------------------ | ---------------------------- |
| GitHub             | Source code management       |
| GitHub Actions     | CI/CD automation             |
| Self-hosted Runner | Executes deployment workflow |
| Docker             | Application containerization |
| AWS ECR            | Docker image registry        |
| AWS EC2            | Application hosting          |
| AWS S3             | ML model registry            |
| MongoDB Atlas      | Dataset storage              |

---

# 🚀 Deployment Workflow

The deployment architecture is:

```mermaid
flowchart TD
    A[GitHub Repository] --> B[GitHub Actions Workflow]
    B --> C[Self Hosted Runner]
    C --> D[Build Docker Image]
    D --> E[Push Image to ECR]
    E --> F[EC2 Instance]
    F --> G[Pull / Run Container]
    G --> H[FastAPI Server]
    H --> I[Vehicle Insurance Prediction]
```

A new commit pushed to the repository triggers the CI/CD workflow.

The workflow builds the Docker image and deploys the application to the AWS infrastructure.

---

# 🛠️ Tech Stack

### Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn

### MLOps

* Modular pipeline architecture
* Data validation
* Model evaluation
* Model artifacts
* Model registry
* Logging
* Exception handling

### Backend

* FastAPI
* Uvicorn
* Jinja2

### Data

* MongoDB Atlas
* Pandas DataFrame

### Cloud & DevOps

* AWS S3
* AWS ECR
* AWS EC2
* Docker
* GitHub Actions

### Development

* Conda
* Git
* GitHub
* `setup.py`
* `pyproject.toml`

---

# ⚙️ Local Setup

## 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd Vehicle-Insurance-Project
```

## 2. Create the Conda environment

```bash
conda create -n vehicle python=3.10 -y
conda activate vehicle
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Install the local package

```bash
pip install -e .
```

Verify the installed packages:

```bash
pip list
```

---

# 🍃 MongoDB Configuration

Create a MongoDB Atlas cluster and database user.

Store the MongoDB connection string as an environment variable rather than hardcoding credentials in the source code.

### Windows CMD

```cmd
set "MONGODB_URL=mongodb+srv://<username>:<password>@<cluster>/<database>"
```

Verify:

```cmd
echo %MONGODB_URL%
```

### PowerShell

```powershell
$env:MONGODB_URL="mongodb+srv://<username>:<password>@<cluster>/<database>"
```

### Bash

```bash
export MONGODB_URL="mongodb+srv://<username>:<password>@<cluster>/<database>"
```

> ⚠️ **Security:** Never commit MongoDB credentials, AWS access keys, `.env` files, or other secrets to GitHub.

---

# ☁️ AWS Configuration

The project uses AWS S3 for model storage and AWS ECR/EC2 for deployment.

Required environment variables include:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION
```

For production deployments, credentials should be stored using a secure secret-management mechanism such as GitHub Actions Secrets or AWS IAM roles rather than hardcoded in source code.

---

# ▶️ Running the Training Pipeline

After configuring MongoDB:

```bash
python demo.py
```

The training pipeline follows:

```text
MongoDB
   ↓
Data Ingestion
   ↓
Data Validation
   ↓
Data Transformation
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Pusher
   ↓
AWS S3
```

---

# 🌐 Running the Web Application

Start the FastAPI application:

```bash
python app.py
```

The application will be available locally at:

```text
http://127.0.0.1:5000
```

The web interface allows users to enter vehicle information and obtain an insurance response prediction.

---

# 📊 Example Prediction Workflow

```text
                    Vehicle Information
                            │
                            ▼
                    ┌───────────────┐
                    │    FastAPI    │
                    └───────┬───────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ Prediction Pipeline│
                  └─────────┬─────────┘
                            │
                            ▼
                     ┌─────────────┐
                     │ Trained ML  │
                     │    Model    │
                     └──────┬──────┘
                            │
                    ┌───────┴────────┐
                    ▼                ▼
              Response-Yes     Response-No
```

---

# 📈 MLOps Lifecycle

The complete lifecycle implemented in this project:

```mermaid
flowchart LR
    A[Data Collection] --> B[Data Ingestion]
    B --> C[Data Validation]
    C --> D[Data Transformation]
    D --> E[Model Training]
    E --> F[Model Evaluation]
    F --> G[Model Registry]
    G --> H[Deployment]
    H --> I[Prediction]
    I --> J[Continuous Improvement]
    J --> A
```

This transforms the project from a standalone ML experiment into a **repeatable ML production workflow**.

---

# 🔐 Configuration & Security

Sensitive configuration is handled through environment variables and CI/CD secrets.

The following should **never** be committed to the repository:

```text
.env
AWS credentials
MongoDB passwords
Private keys
Cloud access tokens
```

The `artifacts/` directory is also excluded from version control where appropriate.

---

# 🧪 Development Workflow

The project was developed incrementally:

```text
Project Template
      ↓
Environment & Packaging
      ↓
MongoDB Integration
      ↓
Logging & Exception Handling
      ↓
EDA & Feature Engineering
      ↓
Data Ingestion
      ↓
Data Validation
      ↓
Data Transformation
      ↓
Model Training
      ↓
Model Evaluation
      ↓
AWS S3 Model Registry
      ↓
Prediction Pipeline
      ↓
FastAPI Application
      ↓
Dockerization
      ↓
GitHub Actions
      ↓
AWS ECR
      ↓
AWS EC2 Deployment
```

---

# 💡 What This Project Demonstrates

This project demonstrates practical experience with the **engineering side of Machine Learning**, including:

* Designing maintainable ML pipelines
* Building reusable Python packages
* Connecting ML systems to cloud databases
* Automating data validation and preprocessing
* Managing model artifacts
* Evaluating and promoting models
* Building inference APIs
* Containerizing ML applications
* Implementing CI/CD for ML systems
* Deploying applications on AWS
* Separating configuration from application code
* Implementing logging and exception handling

---

# 🚧 Future Improvements

Potential extensions include:

* [ ] Add experiment tracking using MLflow
* [ ] Add automated unit and integration tests
* [ ] Add data drift monitoring
* [ ] Add model performance monitoring
* [ ] Add automated model retraining
* [ ] Introduce AWS IAM roles instead of long-lived access keys
* [ ] Add a dedicated secrets manager
* [ ] Add automated API testing
* [ ] Add model versioning and rollback
* [ ] Add monitoring and alerting
* [ ] Improve frontend UX and visualization

---

# 👨‍💻 Author

**Sankalp Singh**

M.Tech — Computer Science & Engineering
IIT Guwahati

Interested in **Machine Learning, MLOps, Deep Learning, and scalable ML systems**.

---

## ⭐ If you found this project useful

Feel free to explore the repository, raise an issue, or contribute to improving the pipeline.

**Built with Python, Machine Learning, FastAPI, MongoDB, AWS, Docker, and GitHub Actions.**
