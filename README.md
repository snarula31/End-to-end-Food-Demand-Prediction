# End-to-End Food Demand Forecasting

## 📌 Project Overview
The **Food Demand Forecasting** project aims to predict the demand for food items in various fulfillment centers for the upcoming weeks. Accurate forecasting enables efficient inventory management, reduces food wastage, and ensures optimal stock levels. This solution implements an end-to-end pipeline ranging from data ingestion and preprocessing to model training and deployment.

The project utilizes both **Traditional Machine Learning** algorithms (XGBoost, CatBoost, LightGBM) and **Deep Learning** techniques (LSTM with Entity Embeddings) to achieve high-accuracy predictions. It serves the final model via a **FastAPI** web application and is containerized using **Docker** for easy deployment.

## 🚀 Key Features
- **Data Ingestion & Processing**: Automated pipeline to merge, clean, and transform raw CSV data.
- **Advanced Feature Engineering**: Creation of time-series features and handling of categorical variables.
- **Model Training**: 
  - **Ensemble Methods**: XGBoost, LightGBM, CatBoost tuned with **Optuna**.
  - **Deep Learning**: LSTM networks with embedding layers for categorical features to capture temporal dependencies.
- **Evaluation Metrics**: Models are evaluated using **RMSLE**, **RMSE**, **MAE**, and **R2 Score**.
- **Deployment**: Interactive web application built with **FastAPI** and **Jinja2** templates.
- **Containerization**: Fully dockerized application for consistent environments.

## 🛠️ Tech Stack
- **Languages**: Python 3.8+
- **Web Framework**: FastAPI, Uvicorn
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn, XGBoost, CatBoost, LightGBM, Optuna
- **Deep Learning**: TensorFlow/Keras (LSTM)
- **Containerization**: Docker
- **Visualization**: Matplotlib, Seaborn

## 📂 Project Structure
```
├── artifacts/             # Stores trained models and processed data
├── notebook/              # Jupyter notebooks for EDA and experiments
├── src/                   # Source code for the application
│   ├── components/        # Core modules (Ingestion, Transformation, Training)
│   ├── pipeline/          # Prediction pipeline logic
│   ├── app.py             # FastAPI application entry point
│   ├── utils.py           # Utility functions
│   ├── logger.py          # Logging configuration
│   └── exception.py       # Custom exception handling
├── templates/             # HTML templates for the web app
├── static/                # Static files (CSS/JS)
├── Dockerfile             # Docker configuration
├── requirements.txt       # Project dependencies
├── setup.py               # Package setup script
└── README.md              # Project documentation
```

## 📊 Dataset Details
The dataset consists of the following files:
- **`train.csv`**: Historical demand data for product-center combinations.
- **`fulfilment_center_info.csv`**: Information about fulfillment centers (e.g., center type, city, region).
- **`meal_info.csv`**: Product information (e.g., category, cuisine).

## ⚙️ Installation
### Prerequisites
- Python 3.8 or higher
- Git

### Steps
1. **Clone the Repository**
   ```bash
   git clone https://github.com/snarula31/End-to-end-Food-Demand-Prediction.git
   cd End-to-end-Food-Demand-Prediction
   ```

2. **Create a Virtual Environment**
   ```bash
   conda create -p venv python=3.8 -y
   conda activate venv/
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🏃 Usage
### Running the Web Application
To start the FastAPI server:
```bash
python src/app.py
```
Open your browser and navigate to `http://localhost:8000` to access the prediction interface.

### Running via Docker
1. **Build the Docker Image**
   ```bash
   docker build -t food-demand-app .
   ```

2. **Run the Container**
   ```bash
   docker run -p 8000:8000 food-demand-app
   ```

## 📈 Model Performance
The models were rigorously evaluated using multiple metrics.

### Key Metrics:
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Percentage Error (MAPE)**
- **R2 Score**

