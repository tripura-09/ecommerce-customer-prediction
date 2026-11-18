
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
from pathlib import Path


# PAGE CONFIGURATION
st.set_page_config(
    page_title="E-Commerce Customer Prediction",
    page_icon="🛒",
    layout="wide"
)

# LOAD PROJECT FILES
# Use the folder containing this App.py file (works on Streamlit Cloud).
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "purchase_prediction_model_compressed.joblib")

@st.cache_data
def load_config():
    with open(BASE_DIR / "purchase_model_config.json") as f:
        return json.load(f)

@st.cache_data
def load_dataset():
    return pd.read_csv(BASE_DIR / "online_shoppers_intention.csv")

model = load_model()
config = load_config()
df = load_dataset()

threshold = config["threshold"]

# SIDEBAR
st.sidebar.title("🛒 E-Commerce Analytics")

page = st.sidebar.radio(
    "Select Dashboard",
    [
        "Dashboard Overview",
        "Purchase Prediction",
        "Customer Behaviour",
        "Feature Importance",
        "Model Performance"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("MCA Final Year Project")

# ---------------------------------
# 1. DASHBOARD OVERVIEW
# ---------------------------------

if page == "Dashboard Overview":

    st.title("🛒 E-Commerce Customer Behaviour")
    st.subheader("Customer Purchase Prediction Dashboard")

    total = len(df)
    purchases = int(df["Revenue"].sum())
    non_purchases = total - purchases
    purchase_rate = purchases / total * 100

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Sessions", f"{total:,}")
    c2.metric("Purchases", f"{purchases:,}")
    c3.metric("No Purchases", f"{non_purchases:,}")
    c4.metric("Purchase Rate", f"{purchase_rate:.2f}%")

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.subheader("Purchase Distribution")

        chart_data = pd.DataFrame({
            "Category": ["Purchase", "No Purchase"],
            "Sessions": [purchases, non_purchases]
        })

        fig = px.pie(
            chart_data,
            names="Category",
            values="Sessions",
            hole=0.45
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Purchase Rate by Visitor Type")

        visitor = (
            df.groupby("VisitorType")["Revenue"]
            .mean()
            .mul(100)
            .reset_index()
        )

        fig = px.bar(
            visitor,
            x="VisitorType",
            y="Revenue",
            color="VisitorType"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly Purchase Analysis")

    month_order = [
        "Feb", "Mar", "May", "June", "Jul",
        "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

    monthly = (
        df.groupby("Month")["Revenue"]
        .mean()
        .mul(100)
        .reindex(month_order)
        .dropna()
        .reset_index()
    )

    fig = px.line(
        monthly,
        x="Month",
        y="Revenue",
        markers=True,
        title="Monthly Purchase Rate"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)


# ---------------------------------
# 2. PURCHASE PREDICTION
# ---------------------------------

elif page == "Purchase Prediction":

    st.title("🔮 Customer Purchase Prediction")

    st.write(
        "Enter the customer's browsing details "
        "to predict their purchase behaviour."
    )

    with st.form("customer_form"):

        col1, col2 = st.columns(2)

        with col1:

            Administrative = st.number_input(
                "Administrative Pages", 0, value=2
            )

            Administrative_Duration = st.number_input(
                "Administrative Duration", 0.0, value=30.0
            )

            Informational = st.number_input(
                "Informational Pages", 0, value=1
            )

            Informational_Duration = st.number_input(
                "Informational Duration", 0.0, value=15.0
            )

            ProductRelated = st.number_input(
                "Product Related Pages", 0, value=20
            )

            ProductRelated_Duration = st.number_input(
                "Product Related Duration", 0.0, value=600.0
            )

            BounceRates = st.number_input(
                "Bounce Rate", 0.0, 1.0, 0.02
            )

            ExitRates = st.number_input(
                "Exit Rate", 0.0, 1.0, 0.04
            )

            PageValues = st.number_input(
                "Page Values", 0.0, value=25.0
            )

        with col2:

            SpecialDay = st.number_input(
                "Special Day", 0.0, 1.0, 0.0
            )

            Month = st.selectbox(
                "Month",
                [
                    "Feb", "Mar", "May", "June",
                    "Jul", "Aug", "Sep", "Oct",
                    "Nov", "Dec"
                ]
            )

            OperatingSystems = st.selectbox(
                "Operating System", list(range(1, 9))
            )

            Browser = st.selectbox(
                "Browser", list(range(1, 14))
            )

            Region = st.selectbox(
                "Region", list(range(1, 10))
            )

            TrafficType = st.selectbox(
                "Traffic Type", list(range(1, 21))
            )

            VisitorType = st.selectbox(
                "Visitor Type",
                [
                    "Returning_Visitor",
                    "New_Visitor",
                    "Other"
                ]
            )

            Weekend = st.selectbox(
                "Weekend Visit", [False, True],
                format_func=lambda x: "Yes" if x else "No"
            )

        submit = st.form_submit_button(
            "Predict Purchase",
            use_container_width=True
        )

    if submit:

        customer = pd.DataFrame([{
            "Administrative": Administrative,
            "Administrative_Duration": Administrative_Duration,
            "Informational": Informational,
            "Informational_Duration": Informational_Duration,
            "ProductRelated": ProductRelated,
            "ProductRelated_Duration": ProductRelated_Duration,
            "BounceRates": BounceRates,
            "ExitRates": ExitRates,
            "PageValues": PageValues,
            "SpecialDay": SpecialDay,
            "Month": Month,
            "OperatingSystems": OperatingSystems,
            "Browser": Browser,
            "Region": Region,
            "TrafficType": TrafficType,
            "VisitorType": VisitorType,
            "Weekend": Weekend
        }])

        probability = model.predict_proba(customer)[0, 1]

        prediction = int(probability >= threshold)

        st.markdown("---")
        st.subheader("Prediction Result")

        st.metric(
            "Purchase Probability",
            f"{probability * 100:.2f}%"
        )

        st.progress(float(probability))

        if prediction == 1:
            st.success(
                "Customer is likely to PURCHASE."
            )
        else:
            st.warning(
                "Customer is likely NOT to PURCHASE."
            )

        st.caption(
            f"Selected model threshold: {threshold:.2f}"
        )


# ---------------------------------
# 3. CUSTOMER BEHAVIOUR
# ---------------------------------

elif page == "Customer Behaviour":

    st.title("📊 Customer Behaviour Analysis")

    c1, c2 = st.columns(2)

    with c1:

        st.subheader("Product Pages vs Purchase")

        fig = px.box(
            df,
            x="Revenue",
            y="ProductRelated",
            color="Revenue"
        )

        st.plotly_chart(fig, use_container_width=True)

    with c2:

        st.subheader("Purchase by Visitor Type")

        visitor = (
            df.groupby("VisitorType")["Revenue"]
            .mean()
            .mul(100)
            .reset_index()
        )

        fig = px.bar(
            visitor,
            x="VisitorType",
            y="Revenue",
            color="VisitorType"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly Purchase Trend")

    month_order = [
        "Feb", "Mar", "May", "June", "Jul",
        "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

    monthly = (
        df.groupby("Month")["Revenue"]
        .mean()
        .mul(100)
        .reindex(month_order)
        .dropna()
        .reset_index()
    )

    fig = px.line(
        monthly,
        x="Month",
        y="Revenue",
        markers=True
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------
# 4. FEATURE IMPORTANCE
# ---------------------------------

elif page == "Feature Importance":

    st.title("⭐ Feature Importance Analysis")

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()

    importance = classifier.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    importance_df = importance_df.sort_values(
        "Importance", ascending=False
    ).head(15)

    fig = px.bar(
        importance_df.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 15 Important Features"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        importance_df,
        use_container_width=True
    )


# ---------------------------------
# 5. MODEL PERFORMANCE
# ---------------------------------

elif page == "Model Performance":

    st.title("📈 Model Performance")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Accuracy",
        f"{config['accuracy'] * 100:.2f}%"
    )

    c2.metric(
        "Precision",
        f"{config['purchase_precision']:.4f}"
    )

    c3.metric(
        "Recall",
        f"{config['purchase_recall']:.4f}"
    )

    c4, c5, c6 = st.columns(3)

    c4.metric(
        "F1 Score",
        f"{config['purchase_f1_score']:.4f}"
    )

    c5.metric(
        "ROC-AUC",
        f"{config['roc_auc']:.4f}"
    )

    c6.metric(
        "Average Precision",
        f"{config['average_precision']:.4f}"
    )

    st.markdown("---")

    st.subheader("Model Details")

    st.write("Algorithm: Random Forest Classifier")
    st.write("Problem Type: Binary Classification")
    st.write("Target: Revenue")
    st.write("Classes: Purchase / No Purchase")

    st.write(
        f"Selected Threshold: {threshold:.2f}"
    )

# FOOTER

st.markdown("---")

st.caption(
    "E-Commerce Customer Behaviour and Purchase Prediction"
)
