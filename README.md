📦 Supply Chain Delay Prediction ML App
A Data-Driven Model for Predicting Delivery Delays in Supply Chains
📘 Project Overview

In today’s global economy, efficient supply chain management is critical for customer satisfaction, cost control, and timely delivery of goods. However, delivery delays remain a persistent challenge that negatively impact operational performance and customer trust.

This project, “A Data-Driven Model for Predicting Delivery Delays in Supply Chains,” focuses on identifying and predicting late deliveries using machine learning techniques. By analyzing real-world transactional supply chain data, the project aims to help organizations proactively manage logistics, optimize operations, and reduce delivery risks.

The solution is implemented as an interactive Streamlit web application that allows users to explore data insights and predict delivery delays using a trained machine learning model.

🎯 Objectives

The key objectives of this project are:

Analyze supply chain transactional data to identify factors contributing to delivery delays

Perform detailed Exploratory Data Analysis (EDA) across markets, regions, product categories, and delivery outcomes

Identify high-risk regions, markets, and products associated with late deliveries

Develop and evaluate machine learning classification models for predicting delivery delays

Visualize insights using Python libraries to support data-driven decision-making

📊 Dataset Description

This project uses the DataCo Smart Supply Chain Dataset (Kaggle), which contains real-world business-to-business supply chain data.

Key Dataset Features

Order status and shipment schedules

Delivery dates and delivery performance

Late delivery risk indicator

Market and regional information

Product category and subcategory details

Sales, cost, profit, and financial losses

Customer demographic attributes

The dataset includes both categorical and numerical features, representing end-to-end supply chain operations.

🧪 Methodology

The project follows a complete data science lifecycle, implemented using Python.

1️⃣ Data Loading & Exploration

Data loaded using Pandas

Dataset structure, distributions, and target variable (Late_delivery_risk) examined

2️⃣ Data Cleaning & Preprocessing

Missing values handling

Mean imputation for Logistic Regression

Median imputation for Random Forest

Categorical encoding

Ordinal Encoding for ordered features

One-Hot Encoding for nominal features

Feature scaling

MinMaxScaler applied (required for Logistic Regression)

Train–Test Split

80% training, 20% testing

📈 Exploratory Data Analysis (EDA)

EDA was performed to uncover patterns and insights, including:

Sales comparison between on-time and late deliveries

Revenue distribution across markets

Identification of high-risk product categories

Regional performance and financial loss analysis

Distribution of late delivery risk

Visualizations were created using:

Bar charts

Pie charts

Horizontal bar plots

🤖 Machine Learning Models
🔹 Logistic Regression (Baseline Model)

Used as a benchmark classifier

Pipeline included:

Mean imputation

One-Hot Encoding

Min-Max Scaling

Performance

Train Accuracy: 54.82%

Test Accuracy: 54.83%

This model underperformed due to the inability to capture complex, non-linear patterns.

🔹 Random Forest Classifier (Final Model)

Selected as the primary model

Pipeline included:

Median imputation

One-Hot Encoding

No scaling required

Reasons for selection

Handles mixed data types effectively

Captures non-linear relationships

Robust to overfitting due to ensemble learning

Performance

Accuracy: 98.12%

Near-perfect recall for late deliveries

Only 2 late deliveries misclassified

This model significantly outperformed Logistic Regression and demonstrated excellent predictive power.

🧾 Model Evaluation Metrics

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

Classification Report

These metrics ensured reliable evaluation, especially for imbalanced delivery classes.

🖥️ Streamlit Application

The trained Random Forest model is deployed using Streamlit, providing:

Interactive user interface

Real-time delay prediction

Data-driven insights for decision-makers

🚀 How to Run the Project Locally
Step 1: Clone the Repository
git clone https://github.com/burrapriyanka85-pixel/supply_chain_ml_app.git
cd supply_chain_ml_app

Step 2: Install Dependencies
pip install -r requirements.txt

Step 3: Run the Streamlit App
streamlit run app.py

📦 Technologies Used

Python

Pandas & NumPy

Matplotlib & Seaborn

scikit-learn

Streamlit

📌 Conclusion

This project successfully demonstrates how machine learning can be used to predict delivery delays in supply chain operations. While Logistic Regression served as a baseline, the Random Forest model delivered outstanding performance with high accuracy and recall.

The insights generated can help organizations:

Identify delay-prone regions and products

Optimize logistics planning

Improve operational efficiency

Enhance customer satisfaction

📜 License

This project is licensed under the Apache 2.0 License.
