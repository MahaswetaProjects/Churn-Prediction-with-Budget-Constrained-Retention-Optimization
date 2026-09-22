import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

RANDOM_STATE = 42

st.set_page_config(
    page_title="Churn Retention Optimizer",
    layout="wide",
)

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.6rem; }
.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["TotalCharges"].isna(), "TotalCharges"] = 0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


@st.cache_resource
def train_models(df):
    customer_ids = df["customerID"]
    work = df.drop(columns=["customerID"])
    X = work.drop(columns=["Churn"])
    y = work["Churn"]

    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numeric_cols = X.select_dtypes(exclude="object").columns.tolist()

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, customer_ids, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])

    log_reg = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    log_reg.fit(X_train, y_train)

    rf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(n_estimators=300, max_depth=8,
                                          class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    rf.fit(X_train, y_train)

    models = {"Logistic Regression": log_reg, "Random Forest": rf}
    aucs = {name: roc_auc_score(y_test, m.predict_proba(X_test)[:, 1]) for name, m in models.items()}

    return {
        "models": models,
        "aucs": aucs,
        "X_test": X_test,
        "y_test": y_test,
        "id_test": id_test,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }


def solve_knapsack(values, costs, max_budget):
    n = len(values)
    dp = np.zeros(max_budget + 1)
    keep = np.zeros((n, max_budget + 1), dtype=bool)
    for i in range(n):
        cost_i = costs[i]
        val_i = values[i]
        prev = dp.copy()
        if cost_i <= max_budget:
            shifted = np.full_like(dp, -np.inf)
            shifted[cost_i:] = prev[:max_budget + 1 - cost_i] + val_i
            take = shifted > prev
            dp = np.where(take, shifted, prev)
            keep[i] = take
    return dp, keep


def recover_selection(keep, costs_int, budget):
    n = keep.shape[0]
    selected = np.zeros(n, dtype=int)
    b = budget
    for i in range(n - 1, -1, -1):
        if keep[i, b]:
            selected[i] = 1
            b -= costs_int[i]
    return selected


def naive_selection(candidates_sorted, budget):
    selected_idx = []
    running_cost = 0.0
    for i, row in candidates_sorted.iterrows():
        if running_cost + row["intervention_cost"] <= budget:
            selected_idx.append(i)
            running_cost += row["intervention_cost"]
    return selected_idx


st.title("Churn Prediction & Budget-Constrained Retention Optimizer")
st.caption(
    "Predicts which customers will churn, estimates the revenue at stake, and decides "
    "who to target with a limited retention budget — using 0/1 knapsack optimization "
    "instead of a naive top-N list."
)

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload Telco churn CSV", type="csv")
    default_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    source = uploaded if uploaded is not None else default_path

    st.header("Model")
    model_choice = st.radio("Model used for predictions", ["Random Forest", "Logistic Regression"], index=0)

    st.header("Retention offer economics")
    base_cost = st.slider("Base outreach cost per customer ($)", 0, 50, 10)
    discount_rate = st.slider("Discount offered (% of monthly bill)", 0, 30, 8) / 100
    success_rate = st.slider("Assumed offer success rate (%)", 5, 80, 35) / 100
    risk_threshold = st.slider("Minimum churn probability to consider targeting", 0.0, 0.9, 0.20, 0.05)

    st.header("Budget")
    monthly_budget = st.slider("Monthly retention budget ($)", 500, 20000, 3000, 500)

try:
    df = load_data(source)
except FileNotFoundError:
    st.warning(
        f"Couldn't find **{default_path}** next to app.py. Upload the CSV from the sidebar "
        "to run the dashboard (Kaggle: 'Telco Customer Churn' by blastchar)."
    )
    st.stop()

bundle = train_models(df)
models = bundle["models"]
X_test = bundle["X_test"]
y_test = bundle["y_test"]
id_test = bundle["id_test"]

model = models[model_choice]
proba_test = model.predict_proba(X_test)[:, 1]
preds_test = model.predict(X_test)

test_df = X_test.copy()
test_df["customerID"] = id_test.values
test_df["actual_churn"] = y_test.values
test_df["churn_probability"] = proba_test

capped_prob = test_df["churn_probability"].clip(lower=0.02)
test_df["expected_remaining_months"] = np.minimum(1 / capped_prob, 36)
test_df["CLV"] = test_df["MonthlyCharges"] * test_df["expected_remaining_months"]
test_df["expected_revenue_at_risk"] = test_df["churn_probability"] * test_df["CLV"]

opt_df = test_df.copy()
opt_df["intervention_cost"] = base_cost + opt_df["MonthlyCharges"] * discount_rate
opt_df["expected_value_if_targeted"] = opt_df["churn_probability"] * success_rate * opt_df["CLV"]

candidates = opt_df[opt_df["churn_probability"] >= risk_threshold].reset_index(drop=True)

costs_int = np.round(candidates["intervention_cost"].values).astype(int)
values_arr = candidates["expected_value_if_targeted"].values

if len(candidates) > 0:
    dp_table, keep_table = solve_knapsack(values_arr, costs_int, monthly_budget)
    candidates["selected_optimized"] = recover_selection(keep_table, costs_int, monthly_budget)
else:
    candidates["selected_optimized"] = []

optimized_value = (candidates["expected_value_if_targeted"] * candidates["selected_optimized"]).sum()
optimized_cost = (costs_int * candidates["selected_optimized"]).sum() if len(candidates) else 0
optimized_count = int(candidates["selected_optimized"].sum())

candidates_sorted = candidates.sort_values("churn_probability", ascending=False).reset_index(drop=True)
naive_idx = naive_selection(candidates_sorted, monthly_budget)
naive_value = candidates_sorted.loc[naive_idx, "expected_value_if_targeted"].sum()
naive_cost = candidates_sorted.loc[naive_idx, "intervention_cost"].sum()
naive_count = len(naive_idx)

improvement = optimized_value - naive_value
improvement_pct = (improvement / naive_value * 100) if naive_value > 0 else 0

tab_overview, tab_model, tab_clv, tab_optimize, tab_sensitivity, tab_explorer = st.tabs([
    "Overview", "Model performance", "CLV & risk", "Budget optimization", "Budget sensitivity", "Customer lookup"
])

with tab_overview:
    churn_rate = df["Churn"].mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Overall churn rate", f"{churn_rate:.1%}")
    c3.metric("Test-set candidates ≥ threshold", f"{len(candidates):,}")
    c4.metric(f"{model_choice} ROC-AUC", f"{bundle['aucs'][model_choice]:.3f}")

    left, right = st.columns(2)
    with left:
        fig = px.histogram(df, x="Contract", color=df["Churn"].map({1: "Churn", 0: "Stay"}),
                            barmode="group", title="Churn by contract type")
        fig.update_layout(legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.box(df, x=df["Churn"].map({1: "Churn", 0: "Stay"}), y="MonthlyCharges",
                     title="Monthly charges by churn outcome")
        fig.update_layout(xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    fig = px.box(df, x=df["Churn"].map({1: "Churn", 0: "Stay"}), y="tenure",
                 title="Tenure by churn outcome")
    fig.update_layout(xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

with tab_model:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("ROC curve")
        fig = go.Figure()
        for name, m in models.items():
            p = m.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, p)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                      name=f"{name} (AUC={bundle['aucs'][name]:.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                  line=dict(dash="dash", color="grey"), name="Chance"))
        fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader(f"Confusion matrix — {model_choice}")
        cm = confusion_matrix(y_test, preds_test)
        fig = px.imshow(cm, text_auto=True, x=["Pred: Stay", "Pred: Churn"], y=["Actual: Stay", "Actual: Churn"],
                         color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

    if model_choice == "Random Forest":
        st.subheader("Top feature importances")
        rf_model = models["Random Forest"].named_steps["model"]
        preprocess = models["Random Forest"].named_steps["preprocess"]
        feature_names = preprocess.get_feature_names_out()
        importances = pd.Series(rf_model.feature_importances_, index=feature_names)
        importances = importances.sort_values(ascending=False).head(15)
        fig = px.bar(x=importances.values, y=[n.split("__")[-1] for n in importances.index], orientation="h",
                     labels={"x": "Importance", "y": ""})
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

with tab_clv:
    st.subheader("Top customers by expected revenue at risk")
    st.caption("expected_revenue_at_risk = churn_probability × estimated CLV")
    top_risk = test_df[["customerID", "MonthlyCharges", "churn_probability", "CLV", "expected_revenue_at_risk"]]
    top_risk = top_risk.sort_values("expected_revenue_at_risk", ascending=False).head(25)
    st.dataframe(top_risk.style.format({
        "MonthlyCharges": "${:.2f}", "churn_probability": "{:.1%}",
        "CLV": "${:,.0f}", "expected_revenue_at_risk": "${:,.0f}"
    }), use_container_width=True)

    fig = px.scatter(test_df, x="churn_probability", y="CLV", color=test_df["actual_churn"].map({1: "Churned", 0: "Stayed"}),
                      hover_data=["customerID"], title="Predicted churn risk vs estimated CLV")
    fig.update_layout(legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

with tab_optimize:
    c1, c2, c3 = st.columns(3)
    c1.metric("Optimized value", f"${optimized_value:,.0f}", help="Expected revenue saved, knapsack-selected")
    c2.metric("Naive value (top-N)", f"${naive_value:,.0f}")
    c3.metric("Improvement", f"${improvement:,.0f}", f"{improvement_pct:.1f}%")

    comparison = pd.DataFrame({
        "Strategy": ["Naive (top churn-probability)", "Optimized (knapsack)"],
        "Customers targeted": [naive_count, optimized_count],
        "Budget spent ($)": [naive_cost, optimized_cost],
        "Expected revenue saved ($)": [naive_value, optimized_value],
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    fig = px.bar(comparison, x="Strategy", y="Expected revenue saved ($)", color="Strategy",
                 title=f"Expected revenue saved at ${monthly_budget:,.0f} budget")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Selected customer list")
    selected = candidates[candidates["selected_optimized"] == 1][
        ["customerID", "churn_probability", "MonthlyCharges", "intervention_cost", "expected_value_if_targeted"]
    ].sort_values("expected_value_if_targeted", ascending=False)
    st.dataframe(selected.style.format({
        "churn_probability": "{:.1%}", "MonthlyCharges": "${:.2f}",
        "intervention_cost": "${:.2f}", "expected_value_if_targeted": "${:,.0f}"
    }), use_container_width=True)
    st.download_button(
        "Download selected customer list (CSV)",
        selected.to_csv(index=False).encode("utf-8"),
        file_name="optimized_retention_list.csv",
        mime="text/csv",
    )

with tab_sensitivity:
    budgets = [1000, 2000, 3000, 4000, 5000, 7500, 10000, 15000]
    if len(candidates) > 0:
        max_b = max(budgets)
        dp_full, _ = solve_knapsack(values_arr, costs_int, max_b)
        rows = []
        for b in budgets:
            opt_val_b = dp_full[b]
            running_cost_b, naive_val_b = 0.0, 0.0
            for _, row in candidates_sorted.iterrows():
                if running_cost_b + row["intervention_cost"] <= b:
                    running_cost_b += row["intervention_cost"]
                    naive_val_b += row["expected_value_if_targeted"]
            rows.append({"Budget": b, "Optimized": opt_val_b, "Naive": naive_val_b})
        sens_df = pd.DataFrame(rows)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sens_df["Budget"], y=sens_df["Optimized"], mode="lines+markers", name="Optimized"))
        fig.add_trace(go.Scatter(x=sens_df["Budget"], y=sens_df["Naive"], mode="lines+markers", name="Naive"))
        fig.add_vline(x=monthly_budget, line_dash="dot", annotation_text="current budget")
        fig.update_layout(xaxis_title="Retention budget ($)", yaxis_title="Expected revenue saved ($)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(sens_df.style.format({"Budget": "${:,.0f}", "Optimized": "${:,.0f}", "Naive": "${:,.0f}"}),
                     use_container_width=True, hide_index=True)
    else:
        st.info("No candidates at the current risk threshold to run a sensitivity check on.")

with tab_explorer:
    st.subheader("Look up a single customer")
    pick = st.selectbox("Customer ID", options=sorted(test_df["customerID"].tolist()))
    row = test_df[test_df["customerID"] == pick].iloc[0]
    opt_row = opt_df[opt_df["customerID"] == pick].iloc[0]
    is_targeted = pick in candidates.loc[candidates["selected_optimized"] == 1, "customerID"].values

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Churn probability", f"{row['churn_probability']:.1%}")
    c2.metric("Estimated CLV", f"${row['CLV']:,.0f}")
    c3.metric("Revenue at risk", f"${row['expected_revenue_at_risk']:,.0f}")
    c4.metric("Retention decision", "Target" if is_targeted else "Skip")

    st.write("Raw customer record:")
    st.dataframe(row.to_frame().T, use_container_width=True)

st.divider()
st.caption(
    "Built on a Telco churn dataset. Models: Logistic Regression baseline + Random Forest "
    "(class-balanced). Retention targeting uses a 0/1 knapsack (dynamic programming) to "
    "maximize expected revenue saved under a fixed monthly budget, benchmarked against a "
    "naive top-churn-probability baseline."
)
