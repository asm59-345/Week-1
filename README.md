
         ## EVision — Intelligent Electric Vehicle Performance & Sales Predictor

### 📅 **Week 1 Report**

##  Problem Statement

With the rapid adoption of electric vehicles (EVs), both manufacturers and consumers face challenges in:

* Accurately estimating **vehicle range** and **price** based on performance specifications.
* Predicting **future EV sales trends** given changing market dynamics.

Currently, there is no unified intelligent platform that predicts these metrics while also offering insights through AI-powered interaction.

## 💡 Proposed Solution

**EVision** is an AI-powered platform that combines **machine learning** and **generative AI** to:

1. Predict EV **range and cost** from specifications.
2. Forecast **future EV sales** using historical data.
3. Provide **chatbot-based insights** using OpenAI / Gemini APIs.
4. Offer an interactive **Streamlit dashboard** for visualization and real-time predictions.

---

## ⚙️ Tech Stack

| Component                   | Technology Used                                   |
| --------------------------- | ------------------------------------------------- |
| **Programming Language**    | Python                                            |
| **Data Analysis**           | Pandas, NumPy                                     |
| **Visualization**           | Matplotlib, Seaborn                               |
| **Machine Learning**        | Scikit-learn (Regression, Random Forest, K-Means) |
| **Frontend (Dashboard)**    | Streamlit                                         |
| **Generative AI APIs**      | OpenAI, Gemini, DeepSeek                          |
| **Development Environment** | Jupyter Notebook                                  |

---

## 🗂️ Week 1 Tasks

### ✅ **1. Define the Problem**

* Researched the challenges in **EV range and sales forecasting**.
* Formulated the problem statement and set clear project goals.

### ✅ **2. Dataset Collection**

* Collected datasets from **Kaggle** and **EV databases** including:

  * EV specifications (battery capacity, motor power, charging time, etc.)
  * Historical EV sales data across countries.
* Imported datasets into **Jupyter Notebook**.

### ✅ **3. Understanding the Dataset**

* Used `head()`, `info()`, `describe()` to explore data structure and key features.
* Identified null values, outliers, and correlations between EV attributes.

### ✅ **4. Data Preprocessing**

* Handled missing data using mean/median imputation.
* Normalized and standardized numeric columns.
* Merged specification and sales datasets on `Model` and `Manufacturer`.
* Encoded categorical features (Brand, Fuel Type, etc.).

### ✅ **5. Exploratory Data Analysis (EDA)**

Visualized relationships between parameters using:

* **Histograms** (distribution of EV range, battery capacity)
* **Scatter Plots** (Battery Capacity vs. Range, Price vs. Efficiency)
* **Heatmap** (Feature correlation)
* **Pairplots** (to detect hidden patterns and relationships)

📊 *Key Insights:*

* Higher battery capacity strongly correlates with EV range.
* Price increases significantly with range and motor power.
* Seasonal trends visible in sales data (higher Q1 & Q3 sales).

### ✅ **6. Model Selection & Training**

#### a) **Supervised Learning**

* **Regression Models:**

  * Simple Linear Regression → Predict EV Range
  * Multiple Linear Regression → Predict EV Cost
* **Classification Models (for future EV type prediction):**

  * Decision Tree, Random Forest

#### b) **Unsupervised Learning**

* **K-Means Clustering** to identify EV segments (e.g., Budget, Mid-range, Premium).

#### c) **Reinforcement Learning (Exploratory)**

* Investigating model optimization using reward-based evaluation.

---

## 🧠 Model Evaluation Metrics

| Metric                            | Description                                                           | Goal                     |
| --------------------------------- | --------------------------------------------------------------------- | ------------------------ |
| **MSE (Mean Squared Error)**      | Measures average squared difference between predicted & actual values | Lower is better          |
| **MAE (Mean Absolute Error)**     | Measures average magnitude of errors                                  | Lower is better          |
| **R² Score**                      | Indicates model accuracy                                              | Above **0.80** preferred |
| **Accuracy / Recall / Precision** | For classification models                                             | > 80% desirable          |

---

## 🎨 Streamlit Frontend (Prototype - Week 1 Setup)

**Objective:** Prepare Streamlit base structure for next week integration.

**Features Implemented:**

* Basic UI Layout: Title, Sidebar, Dataset Upload Section.
* Buttons for EDA and Model Prediction execution.
* Placeholder for chatbot integration (OpenAI API).

**Example Code Snippet:**

```python
import streamlit as st
import pandas as pd

st.title("🚗 EVision — Intelligent EV Predictor")
st.sidebar.header("Upload EV Dataset")

uploaded_file = st.sidebar.file_uploader("Upload your CSV dataset", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.dataframe(data.head())
```

---

## 📈 Outputs (Week 1 Progress)

* ✅ Dataset collected and preprocessed.
* ✅ Completed data exploration and visualization.
* ✅ Implemented initial ML models in Jupyter Notebook.
* ✅ Created basic Streamlit structure for future deployment.

## 📚 Learning Outcomes

* Understood EV datasets and preprocessing techniques.
* Learned regression and clustering algorithms in practice.
* Gained experience integrating data analytics with Streamlit.
* Prepared foundation for chatbot integration using OpenAI API.
* 
## 👨‍💻 Author

**Ashmit Gautam**
AI | ML | Web Developer | Innovator
[LinkedIn Profile](https://www.linkedin.com/in/ashmit-gautam-8a5aa3269/)
