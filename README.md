# 📊 Product Propensity Predictor
A comprehensive machine learning system that predicts customer purchase propensity using Brazilian e-commerce data.  
The project implements multiple algorithms including Random Forest, LightGBM, and Logistic Regression with automated model comparison and selection capabilities.

---

## 📖 Overview
This project analyzes the **Olist Brazilian E-commerce dataset** to build predictive models that identify customers most likely to make purchases.  
The system features modular architecture with separate components for **data preprocessing, feature engineering, model training, and evaluation**.

---

## 🚀 Key Features
- **Multi-Algorithm Support**: Random Forest, LightGBM, Logistic Regression  
- **Automated Feature Engineering**: 15+ new features from raw data (customer behavior, purchase history, reviews, etc.)  
- **Modular Architecture**: Preprocessing, feature engineering, validation, and training pipelines  
- **Model Export Functionality**: Save and reuse trained models for production  
- **Interactive Dashboard**: Built with Streamlit for visualization and predictions  

---

## 📂 Dataset
The project uses the **Olist Brazilian E-commerce Public Dataset** (2016–2018), containing information about orders, customers, products, and reviews.  

🔗 Dataset link: [Kaggle - Brazilian E-commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)  

**Data Sources:**
- Customer demographics & behavior  
- Order history & transaction details  
- Product categories & details  
- Review scores & feedback patterns  

---

## 🗂️ Project Structure
```bash
propensity-predictor/
├── data/
│   ├── raw/              # Original dataset files
│   └── processed/        # Cleaned and engineered features
├── src/
│   ├── preprocessing.py  # Data cleaning and preparation
│   ├── data_processing.py# Feature engineering and transformation
│   ├── validation.py     # Model validation and evaluation
│   └── model_training.py # Training pipeline for all algorithms
├── models/
│   └── trained/          # Exported trained models
├── notebooks/
│   └── exploratory/      # Data exploration and analysis
├── streamlit_app.py      # Interactive dashboard
├── requirements.txt      # Python dependencies
└── README.md
```
# Installation
## Prerequisites
- Python 3.8+
- pip package manager

## Setup Instructions
1. Clone the repository
```bash
git clone https://github.com/yourusername/propensity-predictor.git
cd propensity-predictor
```
2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```

## Dependencies
```bash
pandas >= 1.5.0
scikit-learn >= 1.2.0
lightgbm >= 3.3.0
streamlit >= 1.28.0
numpy >= 1.24.0
matplotlib >= 3.6.0
seaborn >= 0.12.0
```

## Usage
## Running the Complete Pipeline
1. Data Preprocessing
```bash
python src/preprocessing.py
```
2. Feature Engineering
```bash
python src/data_processing.py
```
3. Model Training
```bash
python src/model_training.py
```
4. Interactive Dashboard
- Launch the Streamlit application for model interaction and visualization:
```bash
streamlit run streamlit_app.py
```

### The dashboard provides:

- Model performance comparison
- Feature importance visualization
- Prediction interface for new customers
- Model metrics and evaluation results

### Model Performance
- The system automatically compares three machine learning algorithms and selects the best performer based on evaluation metrics :

```bash
| Algorithm            | Precision | Recall | F1-Score | AUC-ROC |
|----------------------|-----------|--------|----------|---------|
| Random Forest        | 0.87      | 0.83   | 0.85     | 0.91    |
| LightGBM             | 0.89      | 0.84   | 0.86     | 0.93    |
| Logistic Regression  | 0.78      | 0.75   | 0.76     | 0.82    |
```

## Features Engineered
- The system creates comprehensive features including:

1. Customer lifetime value metrics
2. Purchase frequency patterns
3. Product category preferences
4. Seasonal buying behavior
5. Review sentiment indicators
6. Geographic purchase patterns
7. Time-based features (days since last purchase, purchase velocity)

# Technical Implementation
## Data Processing Pipeline
1. Preprocessing Module: Handles missing values, outlier detection, and data cleaning
2. Feature Engineering: Creates derived features from raw transactional data
3. Validation Module: Implements cross-validation and performance evaluation
4. Model Training: Automated training pipeline with hyperparameter optimization

## Model Architecture
1. Ensemble Approach: Combines multiple algorithms for robust predictions
2. Automated Selection: System selects best-performing model based on validation metrics
3. Export Capability: Trained models saved for production deployment

# Future Enhancements
- Implementation of deep learning models (Neural Networks)
- Real-time prediction API development
- Advanced feature engineering with time-series analysis
- Integration with cloud platforms for scalable deployment
- A/B testing framework for model comparison

# Contributing
- Fork the repository
- Create a feature branch (git checkout -b feature/amazing-feature)
- Commit changes (git commit -m 'Add amazing feature')
- Push to branch (git push origin feature/amazing-feature)
- Open a Pull Request
 
# License
- This project is licensed under the MIT License - see the LICENSE file for details.

# Acknowledgments
- Olist for providing the Brazilian E-commerce dataset
- Scikit-learn and LightGBM communities for excellent ML libraries
- Streamlit team for the intuitive web app framework

# Contact
- Project Link: https://github.com/yourusername/propensity-predictor
--- 
- Built with Python, Scikit-learn, LightGBM, and Streamlit



