# Churn Prediction with Budget-Constrained Retention Optimization

## Overview

Most churn projects stop at one question: which customers are likely to leave. That question alone has limited business value, because a company can almost never afford to intervene on every customer flagged as high-risk — retention offers such as discounts, free upgrades, or account manager calls cost money. This project goes further: it predicts churn, converts that prediction into an estimated dollar value per customer, and then decides who to actually target given a fixed monthly retention budget, using a constrained optimization method rather than a simple ranked list. The result is a full pipeline from raw customer data to a concrete, budget-respecting business decision, plus a head-to-head comparison proving that the optimization step adds measurable value over the naive approach most fresher projects stop at.

## Dataset

IBM Telco Customer Churn dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`, publicly available on Kaggle). Each row is one customer with demographics (gender, senior citizen status, partner, dependents), account information (tenure, contract type, paperless billing, payment method), services subscribed (phone, multiple lines, internet service type, online security, online backup, device protection, tech support, streaming TV, streaming movies), charges (monthly charges, total charges), and the target column Churn (Yes/No).

## Pipeline

Data cleaning starts with converting `TotalCharges` from string to numeric — it contains blanks for customers with zero tenure, and these are filled with 0 since a brand-new customer genuinely has zero total charges to date. The target column is mapped to binary, and customer IDs are held aside so they can be reattached to predictions later without leaking into the model as a feature. Exploratory analysis checks the overall churn rate, which matters because roughly 1 in 4 to 5 customers churns in this dataset — a model that predicts "no churn" for everyone would already score 73 to 80 percent accuracy, which is why accuracy alone is never used to judge model quality here. Churn is visualized against contract type, monthly charges, and tenure to see which factors correlate with it before any modeling happens.

The data is split into train and test sets before any encoding or scaling is fit, to avoid leakage — a common invisible bug where encoders or scalers get fit on the full dataset before splitting. Preprocessing is wrapped in a ColumnTransformer and Pipeline: numeric columns are standardized, categorical columns are one-hot encoded. Two models are trained for comparison: a Logistic Regression baseline, which is transparent and interpretable and sets the bar that a more complex model needs to clear to justify its added complexity, and a Random Forest with 300 trees and max depth 8, trained with `class_weight='balanced'` to account for the churn class being the minority class. Models are evaluated with ROC-AUC, full classification reports covering precision, recall, and F1 for both classes, ROC curves, and a confusion matrix, and the better-performing model by ROC-AUC is carried forward through the rest of the pipeline.

Each customer's predicted churn probability is then converted into a Customer Lifetime Value estimate: expected remaining tenure is calculated as 1 divided by the churn probability (capped at 3 years), and CLV is that expected remaining tenure multiplied by monthly charges, giving expected revenue at risk as churn probability multiplied by CLV. This turns an abstract probability into a concrete number — how much revenue is genuinely at stake for this specific customer if nothing is done.

Not every at-risk customer is worth targeting, and the business only has a fixed budget to spend, so this is formulated as a 0/1 knapsack problem: maximize the sum of value times a selection variable, subject to the sum of cost times that same variable staying under the budget, with the selection variable constrained to 0 or 1. For each customer, cost is a base outreach cost plus a percentage discount applied to their monthly bill, and value is churn probability multiplied by an assumed offer success rate multiplied by CLV — the expected revenue saved if the offer actually works. Only customers above a minimum churn-probability threshold are considered candidates, since targeting a near-zero-risk customer just burns budget for no expected benefit. The knapsack is solved exactly via dynamic programming rather than a greedy approximation, and the selected customers are recovered by backtracking through the DP table.

The naive approach that most churn projects stop at — sort by churn probability, spend the budget on the top-ranked customers first — is implemented separately and compared directly against the optimized selection using the identical budget, so the value of the optimization layer is measurable rather than asserted. The notebook reports customers targeted, budget spent, and expected revenue saved for both strategies, along with the dollar and percentage improvement. Because a single comparison at one budget level could be a coincidence, the same optimized-versus-naive comparison is re-run across a range of budgets from $1,000 to $10,000 and beyond, confirming that the optimized strategy consistently beats the naive one regardless of how much money is available — the kind of robustness check that separates a demonstration from an analysis a business could actually rely on.

## Key results

Random Forest and Logistic Regression ROC-AUC scores, and the exact dollar and percentage improvement of the optimized strategy over the naive one at your chosen budget, go here once filled in from your own notebook run — these numbers are what make the results section credible rather than just descriptive, and the optimized strategy should be shown outperforming the naive one at every tested budget level from $1,000 to $10,000 and above.

## Repository contents

`Churn_Budget_Optimization.ipynb` is the full analysis notebook covering everything described above. `app.py` is an interactive Streamlit dashboard version of the same pipeline, with adjustable budget, discount rate, outreach cost, success rate, and risk threshold, live model comparison, a CLV and risk explorer, an optimized-versus-naive comparison view, a budget sensitivity chart, and a single-customer lookup tool. `requirements.txt` lists the dependencies.

## Setup

Install dependencies with `pip install -r requirements.txt`, then run `streamlit run app.py`. Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the same folder, or upload it from the app's sidebar once it's running.

## Tech stack

Python, Pandas, NumPy, scikit-learn for Logistic Regression, Random Forest, and preprocessing pipelines, dynamic programming for the 0/1 knapsack, Streamlit and Plotly for the dashboard, and Matplotlib and Seaborn for the notebook visualizations.

## What this project demonstrates

Correct handling of an imbalanced classification problem evaluated with ROC-AUC rather than accuracy, a concrete and explainable method for converting a probability into a dollar figure, a genuine constrained-optimization layer that turns predictions into a decision system respecting a real business constraint rather than just a model, and quantified proof that the optimization layer adds measurable value over the naive baseline across multiple budget levels.
