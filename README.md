# Car Price Regression using Tensorflow Demo
Car Price Regression using Tensorflow


This Jupyter notebook demonstrates how to manage and test GPU/CPU device placement in TensorFlow 2.x, including device assignment, verification, and basic tensor operations.

## 📋 System Requirements

- **OS**: Ubuntu 22.04 LTS (or compatible Linux distribution)
- **Python**: 3.10.x (3.10.12 specifically tested)
- **TensorFlow**: 2.21.0
- **CUDA**: Required for GPU support (optional, falls back to CPU)
- **Jupyter**: For running the notebook

## 📑 Contents

### `car_price_regression_using_tensorflow.ipynb`
**Car Price Prediction using Neural Network Regression** - A complete machine learning pipeline for predicting vehicle prices using TensorFlow/Keras.

---

### 1. Environment Setup & Configuration
- **Library Imports**: 
  - Core: `tensorflow`, `numpy`, `pandas`
  - Visualization: `seaborn`, `matplotlib.pyplot`
  - Keras modules: `Normalization`, `Dense`, `InputLayer`
  - Loss functions: `MeanAbsoluteError`, `MeanSquaredError`, `Huber`
  - Metrics: `RootMeanSquaredError`
  - Optimizer: `Adam`
- **Environment Variables**: 
  - `TF_CPP_MIN_LOG_LEVEL = '3'` - Suppresses TensorFlow info/warning messages
  - `TF_ENABLE_ONEDNN_OPTS = '0'` - Disables oneDNN optimizations

### 2. Hardware Detection
- **GPU Verification**:
  - Lists available physical GPU devices
  - Counts number of GPUs detected
  - Checks if TensorFlow was built with CUDA support
- Output includes device name(s) and CUDA availability status

### 3. Data Loading & Exploration
- **Data Import**: 
  - Reads `carsData/train.csv` using pandas
  - Uses comma as separator
- **Initial Data Inspection**:
  - `data.head()` - Displays first 5 rows to understand data structure
  - Shows 12 columns including features and target variable
- **Statistical Summary**:
  - `data.describe().transpose()` - Provides comprehensive statistics:
    - Count, mean, standard deviation
    - Minimum, maximum values
    - Quartile distributions (25%, 50%, 75%)
- **Missing Values Check**:
  - `data.isna().sum()` - Verifies data completeness
  - Returns zero missing values for all columns

### 4. Data Preprocessing (Setup)
- **Normalization Layer**: Imported for feature scaling
- **Data splitting**: Framework prepared for train/test separation

### 5. Model Architecture Components
- **Sequential Model Structure**:
  - `InputLayer` - Defines input shape
  - `Dense` layers - Fully connected hidden layers
  - Output layer - Single neuron for price prediction

### 6. Model Compilation Components
- **Optimizers**: 
  - `Adam` - Adaptive Moment Estimation optimizer
- **Loss Functions**:
  - `MeanAbsoluteError` (MAE) - Robust to outliers
  - `MeanSquaredError` (MSE) - Penalizes larger errors
  - `Huber` - Combines MAE and MSE advantages
- **Evaluation Metrics**:
  - `RootMeanSquaredError` (RMSE) - Error in original price units

---

## 📊 Dataset Features

| Feature Name | Description | Data Type |
|--------------|-------------|-----------|
| v.id | Vehicle identifier | Integer |
| on road old | Previous on-road price | Integer |
| on road now | Current on-road price | Integer |
| years | Age of vehicle | Integer |
| km | Kilometers driven | Integer |
| rating | User satisfaction rating | Integer |
| condition | Vehicle condition score | Integer |
| economy | Fuel economy rating | Integer |
| top speed | Maximum speed capability | Integer |
| hp | Horsepower | Integer |
| torque | Engine torque | Integer |
| **current price** | **Target variable** | **Float** |

---

## 📈 Performance Metrics Tracked

| Metric | Purpose |
|--------|---------|
| **Mean Absolute Error (MAE)** | Average absolute prediction error |
| **Mean Squared Error (MSE)** | Penalizes large errors quadratically |
| **Huber Loss** | Combines MAE and MSE characteristics |
| **Root Mean Squared Error (RMSE)** | Error in same units as target variable |

---

## 🎯 Key Learning Outcomes

| Section | Key Concepts Covered |
|---------|---------------------|
| **Environment Setup** | TensorFlow configuration for optimal performance |
| **Hardware Detection** | GPU verification for accelerated training |
| **Data Exploration** | Dataset structure and statistical analysis |
| **Model Architecture** | Neural networks for regression tasks |
| **Model Compilation** | Loss functions and optimizer selection |
| **Training & Evaluation** | Regression metrics and model assessment |

## 🚀 Quick Setup Guide

### 1. Clone/Download the Repository
```bash
git clone https://github.com/mohammed3120/Car_Price_Regression_using_Tensorflow.git

cd Car_Price_Regression_using_Tensorflow
```
### 2. Create Virtual Environment

##### Create venv with Python 3

```bash
python3 -m venv venv
```
#### Activate the virtual environment

```bash
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Configure CUDA Library Paths
1. After installing dependencies, add the following CUDA library paths to your virtual environment activation script:
```bash
cat >> venv/bin/activate << 'EOF'
```
2. paste:
```bash
if [ -n "$VIRTUAL_ENV" ]; then
    VENV_SITE_PACKAGES="$VIRTUAL_ENV/lib/python3.10/site-packages"
    export LD_LIBRARY_PATH="$VENV_SITE_PACKAGES/nvidia/cuda_runtime/lib:$VENV_SITE_PACKAGES/nvidia/cudnn/lib:$VENV_SITE_PACKAGES/nvidia/cublas/lib:$VENV_SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$VENV_SITE_PACKAGES/nvidia/cuda_cupti/lib:$VENV_SITE_PACKAGES/nvidia/cusolver/lib:${LD_LIBRARY_PATH:-}"
fi
EOF
```

3. Re-activate the virtual environment
```bash
deactivate

source venv/bin/activate
```
### 5. Launch Jupyter Notebook
```bash
jupyter notebook
```
