# 🚗 Vehicle Price Prediction

## 📌 Project Overview

This project predicts the **price of a vehicle** based on its specifications using a regression model. It processes structured and textual vehicle data, extracts meaningful features, and builds a predictive model. A **Streamlit web app** allows users to input vehicle attributes and instantly receive a predicted price.

---

## 🎯 Objective

Build a machine learning system to **predict vehicle price** accurately from its features.

The model output is a **continuous value** representing the vehicle’s estimated **price in USD**.

---

## 📊 Dataset Description

The dataset includes structured information and descriptive text about vehicles.

### Raw Dataset Features

| Column           | Description                             |
| ---------------- | --------------------------------------- |
| `name`           | Full vehicle name (make + model + trim) |
| `description`    | Text description of vehicle features    |
| `make`           | Manufacturer (e.g., Ford, Toyota)       |
| `model`          | Model name                              |
| `year`           | Year of manufacture                     |
| `price`          | Price in USD (target variable)          |
| `engine`         | Engine details (type/specs)             |
| `cylinders`      | Engine cylinder count                   |
| `fuel`           | Fuel type (e.g., Gasoline, Electric)    |
| `mileage`        | Mileage in miles                        |
| `transmission`   | Transmission type (e.g., Automatic)     |
| `trim`           | Trim level                              |
| `body`           | Body style (e.g., SUV, Sedan)           |
| `doors`          | Number of doors                         |
| `exterior_color` | Vehicle exterior color                  |
| `interior_color` | Vehicle interior color                  |
| `drivetrain`     | Drivetrain type (e.g., AWD, FWD)        |

---

## 🧹 Data Cleaning & Feature Engineering

The [`data_cleaning.ipynb`](./data_cleaning.ipynb) notebook includes:

- **Parsing textual fields** (e.g., `engine`, `transmission`) into structured features.
- Extracted attributes:
  - `fuel_system`, `valve_train`, `aspiration`, `transmission_type`, etc.
- Unique values for all columns are stored in the lookups/ folder, which were used to explore the data, identify anomalies, extract new features, and reduce categorical cardinality.
  - `unique_engines.txt`, `unique_trim.txt`, `unique_exterior_colour.txt`, etc. 


---

## 🧠 Machine Learning Pipeline

The final model uses the **XGBoost Regressor** wrapped in a pipeline.

### 🔧 Preprocessing

| Step           | Transformer         | Details                                                               |
| -------------- | ------------------- | --------------------------------------------------------------------- |
| `preprocessor` | `ColumnTransformer` | OneHotEncoder for categorical columns, passthrough for numerical      |
| `model`        | `XGBRegressor`      | `objective='reg:squarederror'`, `n_estimators=200`, `random_state=42` |

**Categorical columns encoded:**

```python
['make', 'model', 'fuel', 'trim', 'body', 'drivetrain',
 'fuel_system', 'valve_train', 'aspiration', 'transmission_type',
 'ext_color', 'int_color']
```

### 🧚‍♀️ Model Training and Performance

Trained on cleaned and feature-enriched data from `data_processed.csv`.

- **Train-Test Evaluation**:

  - **RMSE**: 7123.86
  - **R² Score**: 0.8595

- **Cross-Validation (5-fold)**:

  - **Average R²**: 0.8110



---

## 🌐 Web Application

A **Streamlit web app** allows users to select or input vehicle features and view price predictions.

🎥 **Demo:** 

![Demo](assets/application.gif)
---

## ⚙️ Technologies Used

- **Python**: Main programming language
- **Pandas, NumPy**: Data processing
- **scikit-learn, XGBoost**: Machine learning & preprocessing
- **Streamlit**: Web interface for model deployment

---

## 🗂️ Project Structure

| File / Folder         | Description                                    |
| --------------------- | ---------------------------------------------- |
| `app.py`              | Streamlit web app                              |
| `dataset.csv`         | Original raw dataset                           |
| `data_processed.csv`  | Cleaned & feature-engineered dataset           |
| `data_cleaning.ipynb` | Notebook for cleaning and feature extraction   |
| `train_model.ipynb`   | Model training & evaluation                    |
| `model.pkl`           | Serialized trained pipeline                    |
| `lookups/`            | Text files of unique values for raw textual columns |
| `assets/`             | Images, videos, and visual assets              |

---

## Installation & Setup

Follow these steps to run locally:

1. **Download or Clone the Repository**\
   → Use GitHub's "Code → Download ZIP" or `git clone`

2. **Create and Activate Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate       # Linux/Mac
   venv\Scripts\activate          # Windows
   ```

3. **Install Required Packages**

   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Streamlit App**

   ```bash
   streamlit run app.py
   ```

5. **Access the App in Browser**

   - Local: [http://localhost:8501](http://localhost:8501)
   - Network: [http://your-ip:8501](http://your-ip:8501)

---

## 📈 Summary

This end-to-end system for **vehicle price prediction** provides:

- Clean and structured input from raw textual specs
- A powerful XGBoost regression model
- A responsive and intuitive web interface for real-time prediction

