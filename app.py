"""
Customer Churn Prediction and Retention Segmentation Dashboard
Built with Streamlit for TY Computer Engineering ML Lab (Practical 10)
Corporate Navy & Grey Professional Theme
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Customer Churn Prediction & Retention Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Unified Professional Corporate CSS (Navy & Grey)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stMarkdown, .stText, p, span, div {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Main container background */
    .stApp {
        background-color: #F1F5F9;
    }

    /* Full-width Navy Header Banner (No Gradients) */
    .header-banner {
        background-color: #0B1F3A;
        border-radius: 10px;
        padding: 24px 30px;
        margin-bottom: 22px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    }
    .header-title {
        color: #FFFFFF;
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin: 0;
        line-height: 1.25;
    }
    .header-subtitle {
        color: #CBD5E1;
        font-size: 0.92rem;
        margin-top: 6px;
        margin-bottom: 14px;
        font-weight: 400;
    }
    .header-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .badge-pill {
        background-color: rgba(255, 255, 255, 0.10);
        color: #F8FAFC;
        border: 1px solid rgba(255, 255, 255, 0.20);
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    /* Sidebar Styling (Navy-Tinted) */
    section[data-testid="stSidebar"] {
        background-color: #0D1E35;
        border-right: 1px solid #1E293B;
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: #E2E8F0 !important;
    }
    .sidebar-section-title {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94A3B8 !important;
        margin-top: 18px;
        margin-bottom: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.12);
        padding-bottom: 4px;
    }

    /* Sidebar Steel Blue Button */
    section[data-testid="stSidebar"] .stButton > button {
        background-color: #1F4E79 !important;
        color: #FFFFFF !important;
        border: 1px solid #1F4E79 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 10px 18px !important;
        width: 100% !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12) !important;
        transition: background-color 0.15s ease !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #286096 !important;
        border-color: #286096 !important;
    }

    /* White Cards */
    .white-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        margin-bottom: 16px;
    }

    /* KPI Cards */
    .kpi-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 20px 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        text-align: center;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .kpi-label {
        font-size: 0.72rem;
        font-weight: 700;
        color: #64748B;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0B1F3A;
        line-height: 1.1;
        margin-bottom: 6px;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #64748B;
    }

    /* Status Badges */
    .status-badge-low {
        background-color: #DCFCE7;
        color: #15803D;
        border: 1px solid #BBF7D0;
        font-weight: 700;
        font-size: 0.95rem;
        padding: 4px 12px;
        border-radius: 6px;
        display: inline-block;
    }
    .status-badge-medium {
        background-color: #FEF3C7;
        color: #B45309;
        border: 1px solid #FDE68A;
        font-weight: 700;
        font-size: 0.95rem;
        padding: 4px 12px;
        border-radius: 6px;
        display: inline-block;
    }
    .status-badge-high {
        background-color: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FECACA;
        font-weight: 700;
        font-size: 0.95rem;
        padding: 4px 12px;
        border-radius: 6px;
        display: inline-block;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 0px;
        margin-bottom: 18px;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        color: #475569 !important;
        padding: 10px 18px !important;
        border-radius: 8px 8px 0 0 !important;
        background-color: transparent !important;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        color: #0B1F3A !important;
        border-bottom: 3px solid #1F4E79 !important;
        background-color: #FFFFFF !important;
    }

    /* Playbook Action Cards */
    .action-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        display: flex;
        align-items: flex-start;
        gap: 16px;
    }
    .action-num {
        background-color: #1F4E79;
        color: #FFFFFF;
        font-weight: 700;
        font-size: 0.88rem;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .action-body {
        flex: 1;
    }
    .action-title {
        font-size: 0.98rem;
        font-weight: 700;
        color: #0B1F3A;
        margin-bottom: 4px;
    }
    .action-reason {
        font-size: 0.87rem;
        color: #475569;
        line-height: 1.45;
    }

    /* Revenue at Risk Banner */
    .revenue-banner {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #1F4E79;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 18px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .revenue-label {
        font-size: 0.72rem;
        font-weight: 700;
        color: #64748B;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .revenue-val {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0B1F3A;
        line-height: 1.1;
    }
    .revenue-sub {
        font-size: 0.82rem;
        color: #64748B;
        margin-top: 4px;
    }

    /* Empty State Card */
    .empty-state-box {
        background-color: #FFFFFF;
        border: 1px dashed #CBD5E1;
        border-radius: 10px;
        padding: 44px 24px;
        text-align: center;
        margin-top: 10px;
    }
    .empty-state-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0B1F3A;
        margin-bottom: 6px;
    }
    .empty-state-text {
        font-size: 0.92rem;
        color: #64748B;
        max-width: 500px;
        margin: 0 auto;
        line-height: 1.45;
    }

    /* Footer */
    .footer-line {
        font-size: 0.78rem;
        color: #64748B;
        text-align: center;
        margin-top: 36px;
        padding-top: 16px;
        border-top: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    """Load pre-trained best churn model and retention cluster artifacts."""
    model_path = os.path.join("models", "best_churn_model.joblib")
    cluster_path = os.path.join("models", "retention_cluster_model.joblib")

    model = None
    cluster_artifact = None

    if os.path.exists(model_path):
        model = joblib.load(model_path)
    if os.path.exists(cluster_path):
        cluster_artifact = joblib.load(cluster_path)

    return model, cluster_artifact


def engineer_features_for_input(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering transformations matching training pipeline."""
    df_fe = df.copy()

    # Tenure groups
    bins = [-1, 12, 24, 48, 60, np.inf]
    labels = ["0-12m", "13-24m", "25-48m", "49-60m", ">60m"]
    df_fe["tenure_group"] = pd.cut(df_fe["tenure"], bins=bins, labels=labels)

    # Average monthly spend
    df_fe["avg_monthly_spend"] = (df_fe["TotalCharges"] / (df_fe["tenure"] + 1)).round(2)

    # Number of active services
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    count_services = pd.Series(0, index=df_fe.index)
    for col in service_cols:
        if col in df_fe.columns:
            count_services += (df_fe[col] == "Yes").astype(int)
    if "InternetService" in df_fe.columns:
        count_services += (df_fe["InternetService"].isin(["DSL", "Fiber optic"])).astype(int)
    df_fe["num_services"] = count_services

    # Has support services
    has_tech = (df_fe["TechSupport"] == "Yes")
    has_sec = (df_fe["OnlineSecurity"] == "Yes")
    df_fe["has_support_services"] = (has_tech | has_sec).map({True: "Yes", False: "No"})

    return df_fe


def main():
    # 1. Header Banner
    st.markdown("""
    <div class="header-banner">
        <div class="header-title">Customer Churn Prediction & Retention Dashboard</div>
        <div class="header-subtitle">Practical 10: Telecom Churn Prediction & Retention Micro-Segmentation Engine &bull; TY Computer Engineering</div>
        <div class="header-badges">
            <span class="badge-pill">Gradient Boosting</span>
            <span class="badge-pill">Threshold 0.40</span>
            <span class="badge-pill">K-Means Segmentation</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    best_model, cluster_artifact = load_artifacts()

    # 2. Sidebar: Navy-tinted background, small-caps headers, grouped sections
    st.sidebar.markdown('<div class="sidebar-section-title">Demographics</div>', unsafe_allow_html=True)
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])

    st.sidebar.markdown('<div class="sidebar-section-title">Contract and Billing</div>', unsafe_allow_html=True)
    tenure = st.sidebar.slider("Tenure (Months with company)", min_value=0, max_value=72, value=6)
    contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.sidebar.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    monthly = st.sidebar.number_input("Monthly Charges ($)", min_value=18.0, max_value=130.0, value=75.0, step=0.5)
    est_total = round(tenure * monthly, 2)
    total = st.sidebar.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(est_total), step=1.0)

    st.sidebar.markdown('<div class="sidebar-section-title">Services</div>', unsafe_allow_html=True)
    phone = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    mult_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet = st.sidebar.selectbox("Internet Service Provider", ["Fiber optic", "DSL", "No"])

    if internet != "No":
        sec = st.sidebar.selectbox("Online Security", ["No", "Yes"])
        backup = st.sidebar.selectbox("Online Backup", ["No", "Yes"])
        protect = st.sidebar.selectbox("Device Protection", ["No", "Yes"])
        tech = st.sidebar.selectbox("Tech Support", ["No", "Yes"])
        tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes"])
        movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes"])
    else:
        sec = "No internet service"
        backup = "No internet service"
        protect = "No internet service"
        tech = "No internet service"
        tv = "No internet service"
        movies = "No internet service"

    # Analyze Button in Sidebar
    st.sidebar.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    analyze_btn = st.sidebar.button("Analyze Churn Risk", type="primary", use_container_width=True)

    # State tracking so tab switching retains prediction
    if analyze_btn:
        st.session_state["has_analyzed"] = True

    # 3. Prediction & Segmentation Execution
    if not st.session_state.get("has_analyzed", False):
        st.markdown("""
        <div class="empty-state-box">
            <div class="empty-state-title">Ready for Churn Risk Diagnostics</div>
            <div class="empty-state-text">
                Enter customer details in the sidebar and click <strong>Analyze Churn Risk</strong> to generate risk probabilities, factor comparisons, and personalized retention actions.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        if best_model is None:
            st.error("Model artifact (models/best_churn_model.joblib) not found. Please ensure trained models exist.")
            return

        # Prepare input dataframe
        raw_input = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": mult_lines,
            "InternetService": internet,
            "OnlineSecurity": sec,
            "OnlineBackup": backup,
            "DeviceProtection": protect,
            "TechSupport": tech,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }])

        # Feature engineering
        processed_input = engineer_features_for_input(raw_input)

        # Predict probability
        try:
            prob = float(best_model.predict_proba(processed_input)[0, 1])
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        # Tuned decision threshold (default 0.40)
        tuned_threshold = cluster_artifact.get('tuned_threshold', 0.40) if cluster_artifact else 0.40

        # Risk Classification
        if prob < 0.25:
            risk_label = "LOW RISK"
            badge_class = "status-badge-low"
        elif prob < tuned_threshold:
            risk_label = "MEDIUM RISK"
            badge_class = "status-badge-medium"
        else:
            risk_label = "HIGH RISK"
            badge_class = "status-badge-high"

        # Persona Assignment via K-Means
        assigned_persona = "Standard Retained Profile"
        cluster_id = None
        if prob >= tuned_threshold and cluster_artifact is not None:
            try:
                kmeans = cluster_artifact['kmeans']
                scaler = cluster_artifact['scaler']
                features = cluster_artifact.get(
                    'features',
                    ['tenure', 'MonthlyCharges', 'TotalCharges', 'avg_monthly_spend', 'num_services']
                )
                persona_mapping = cluster_artifact.get('persona_mapping', {})

                input_seg = processed_input[features]
                input_seg_scaled = scaler.transform(input_seg)
                cluster_id = int(kmeans.predict(input_seg_scaled)[0])
                assigned_persona = persona_mapping.get(cluster_id, f"Cluster {cluster_id}")
            except Exception as e:
                st.error(f"Persona assignment failed: {e}")
                return
        elif prob < tuned_threshold:
            assigned_persona = "Safe Retention Cohort"

        # 4. Main Area: Three Tabs
        tab_risk, tab_profile, tab_playbook = st.tabs([
            "Risk Assessment",
            "Customer Profile",
            "Retention Playbook"
        ])

        # --- TAB 1: RISK ASSESSMENT ---
        with tab_risk:
            # Row of 3 KPI Cards
            kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
            with kpi_col1:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Churn Probability</div>
                    <div class="kpi-value">{prob * 100:.1f}%</div>
                    <div class="kpi-sub">Decision threshold: {tuned_threshold * 100:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)

            with kpi_col2:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Risk Level</div>
                    <div style="margin: 8px 0;"><span class="{badge_class}">{risk_label}</span></div>
                    <div class="kpi-sub">Classification based on tuned cutoff</div>
                </div>
                """, unsafe_allow_html=True)

            with kpi_col3:
                persona_sub = f"Cluster {cluster_id}" if cluster_id is not None else "Below threshold"
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Assigned Persona</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #0B1F3A; margin: 6px 0;">{assigned_persona}</div>
                    <div class="kpi-sub">{persona_sub}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

            # Semicircular Plotly Gauge Chart
            st.markdown("""
            <div class="white-card">
                <div style="font-size: 0.88rem; font-weight: 700; color: #0B1F3A; margin-bottom: 4px;">
                    Churn Propensity Gauge
                </div>
                <div style="font-size: 0.82rem; color: #64748B; margin-bottom: 12px;">
                    Semicircular calibrated indicator with operational zones and decision threshold line at 40%.
                </div>
            """, unsafe_allow_html=True)

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                number={
                    'suffix': "%",
                    'valueformat': ".1f",
                    'font': {'size': 42, 'family': 'Inter, sans-serif', 'color': '#0B1F3A'}
                },
                gauge={
                    'shape': 'angular',
                    'axis': {
                        'range': [0, 100],
                        'tickwidth': 1,
                        'tickcolor': '#94A3B8',
                        'tickvals': [0, 25, 40, 75, 100],
                        'ticktext': ['0%', '25% (Low)', '40% (Threshold)', '75%', '100%'],
                        'tickfont': {'size': 11, 'color': '#475569', 'family': 'Inter, sans-serif'}
                    },
                    'bar': {'color': '#0B1F3A', 'thickness': 0.22},
                    'bgcolor': '#FFFFFF',
                    'borderwidth': 1,
                    'bordercolor': '#E2E8F0',
                    'steps': [
                        {'range': [0, 25], 'color': '#DCFCE7'},    # Low risk (Soft green)
                        {'range': [25, 40], 'color': '#FEF3C7'},   # Medium risk (Soft amber)
                        {'range': [40, 100], 'color': '#FEE2E2'}   # High risk (Soft red)
                    ],
                    'threshold': {
                        'line': {'color': '#B91C1C', 'width': 4},
                        'thickness': 0.82,
                        'value': tuned_threshold * 100
                    }
                }
            ))
            fig_gauge.update_layout(
                height=260,
                margin=dict(l=30, r=30, t=25, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'family': 'Inter, sans-serif'}
            )
            st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
            st.markdown("</div>", unsafe_allow_html=True)

        # --- TAB 2: CUSTOMER PROFILE ---
        with tab_profile:
            prof_col1, prof_col2 = st.columns([1, 1.15], gap="medium")

            with prof_col1:
                st.markdown("""
                <div class="white-card">
                    <div style="font-size: 0.88rem; font-weight: 700; color: #0B1F3A; margin-bottom: 12px;">
                        Entered Customer Attributes
                    </div>
                """, unsafe_allow_html=True)

                summary_records = [
                    {"Category": "Demographics", "Attribute": "Gender / Senior", "Value": f"{gender} / {senior}"},
                    {"Category": "Demographics", "Attribute": "Partner / Dependents", "Value": f"{partner} / {dependents}"},
                    {"Category": "Contract", "Attribute": "Tenure", "Value": f"{tenure} months"},
                    {"Category": "Contract", "Attribute": "Contract Type", "Value": contract},
                    {"Category": "Billing", "Attribute": "Payment Method", "Value": payment},
                    {"Category": "Billing", "Attribute": "Paperless Billing", "Value": paperless},
                    {"Category": "Billing", "Attribute": "Monthly Charges", "Value": f"${monthly:.2f}"},
                    {"Category": "Billing", "Attribute": "Total Charges", "Value": f"${total:.2f}"},
                    {"Category": "Services", "Attribute": "Internet Service", "Value": internet},
                    {"Category": "Services", "Attribute": "Tech Support", "Value": tech},
                    {"Category": "Services", "Attribute": "Online Security", "Value": sec},
                    {"Category": "Services", "Attribute": "Streaming (TV / Movies)", "Value": f"{tv} / {movies}"}
                ]
                df_summary = pd.DataFrame(summary_records)
                st.dataframe(
                    df_summary,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Category": st.column_config.TextColumn("Category", width="small"),
                        "Attribute": st.column_config.TextColumn("Attribute", width="medium"),
                        "Value": st.column_config.TextColumn("Value", width="medium")
                    }
                )
                st.markdown("</div>", unsafe_allow_html=True)

            with prof_col2:
                st.markdown("""
                <div class="white-card">
                    <div style="font-size: 0.88rem; font-weight: 700; color: #0B1F3A; margin-bottom: 4px;">
                        Top Churn Drivers vs At-Risk Benchmark
                    </div>
                    <div style="font-size: 0.82rem; color: #64748B; margin-bottom: 14px;">
                        Comparing this customer with the typical profile of historical churners.
                    </div>
                """, unsafe_allow_html=True)

                # Top factors comparison metrics
                # Benchmark stats from Telco churners:
                # - Tenure: Avg 18.0 mos (Short tenure = high risk)
                # - Monthly Charges: Avg $74.44
                # - Contract: 88.6% Month-to-month
                # - Tech Support: 77.4% have No Tech Support
                # - Payment Method: 57.3% Electronic Check
                factors = [
                    "Payment Method",
                    "Tech Support",
                    "Monthly Charges",
                    "Contract Type",
                    "Tenure Risk"
                ]

                # Normalized Risk Alignment Score (0 - 100%)
                # Higher score = stronger alignment with churn driver
                tenure_risk_score = round(max(0, min(100, (1.0 - tenure / 72.0) * 100)))
                monthly_risk_score = round(min(100, (monthly / 125.0) * 100))
                contract_risk_score = 89 if contract == "Month-to-month" else (20 if contract == "One year" else 5)
                tech_risk_score = 77 if tech == "No" else (17 if tech == "Yes" else 10)
                payment_risk_score = 57 if payment == "Electronic check" else (20 if payment == "Mailed check" else 13)

                customer_scores = [payment_risk_score, tech_risk_score, monthly_risk_score, contract_risk_score, tenure_risk_score]
                customer_labels = [
                    payment,
                    f"Tech Support: {tech}",
                    f"${monthly:.2f}/mo",
                    contract,
                    f"{tenure} mos"
                ]

                benchmark_scores = [57, 77, 60, 89, 75]
                benchmark_labels = [
                    "Electronic check (57%)",
                    "No Tech Support (77%)",
                    "$74.44/mo avg",
                    "Month-to-month (89%)",
                    "18.0 mos avg"
                ]

                fig_factors = go.Figure()
                fig_factors.add_trace(go.Bar(
                    y=factors,
                    x=customer_scores,
                    orientation='h',
                    name='Current Customer',
                    text=customer_labels,
                    textposition='auto',
                    marker=dict(color='#0B1F3A')
                ))
                fig_factors.add_trace(go.Bar(
                    y=factors,
                    x=benchmark_scores,
                    orientation='h',
                    name='Avg At-Risk Customer',
                    text=benchmark_labels,
                    textposition='auto',
                    marker=dict(color='#94A3B8')
                ))

                fig_factors.update_layout(
                    barmode='group',
                    height=340,
                    margin=dict(l=20, r=20, t=10, b=25),
                    xaxis=dict(
                        title=dict(text='Risk Alignment Index (%)', font=dict(size=11, color='#64748B')),
                        range=[0, 115],
                        gridcolor='#E2E8F0',
                        tickfont=dict(size=10, color='#64748B')
                    ),
                    yaxis=dict(
                        gridcolor='#E2E8F0',
                        tickfont=dict(size=11, color='#0B1F3A', family='Inter, sans-serif')
                    ),
                    legend=dict(
                        orientation='h',
                        yanchor='bottom',
                        y=1.02,
                        xanchor='right',
                        x=1,
                        font=dict(size=11, color='#475569')
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Inter, sans-serif')
                )

                st.plotly_chart(fig_factors, use_container_width=True, config={'displayModeBar': False})
                st.markdown("</div>", unsafe_allow_html=True)

        # --- TAB 3: RETENTION PLAYBOOK ---
        with tab_playbook:
            annual_rev_at_risk = monthly * 12.0

            # Revenue at risk banner
            st.markdown(f"""
            <div class="revenue-banner">
                <div class="revenue-label">Estimated Annual Revenue at Risk</div>
                <div class="revenue-val">${annual_rev_at_risk:,.2f}</div>
                <div class="revenue-sub">Based on current monthly billing of ${monthly:.2f}/mo &times; 12 months for this account</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="font-size: 0.95rem; font-weight: 700; color: #0B1F3A; margin-bottom: 12px;">
                Prescribed Next-Best Actions: {assigned_persona}
            </div>
            """, unsafe_allow_html=True)

            # Persona-specific action playbook
            if assigned_persona == "Premium Long-Term Streamers":
                actions = [
                    (
                        "Proactive Senior Account Manager VIP Outreach",
                        "High lifetime billing and extended tenure (~40 mos) represent high-equity value. Dedicated outreach within 24 hours preempts competitor poaching."
                    ),
                    (
                        "Complimentary Next-Gen Wi-Fi 6 Router Hardware Upgrade",
                        "Heavy 4K streaming subscribers on Fiber optic require uninterrupted bandwidth. Hardware refreshes eliminate silent streaming frustration."
                    ),
                    (
                        "Loyalty Anniversary Credit ($30) & 15% Annual Lock-in",
                        "Flexible month-to-month contracts introduce defection risk. Combining a loyalty anniversary credit with an annual discount secures ARR continuity."
                    )
                ]
            elif assigned_persona == "Budget-Conscious Newcomers":
                actions = [
                    (
                        "Dedicated Care Onboarding Call within 48 Hours",
                        "Early subscriber tenure (~2.7 mos) experiences the steepest churn. Personalized onboarding resolves initial configuration and billing confusion."
                    ),
                    (
                        "Enrolment Incentive for Paperless Digital Autopay ($5/mo Discount)",
                        "Mailed checks and paper billing induce friction and payment dropouts. Autopay (ACH/Credit Card) stabilizes payment continuity."
                    ),
                    (
                        "1-Year Price-Lock Protection Guarantee",
                        "Extreme price sensitivity on entry-tier plans makes guaranteed rate stability the most compelling defense against budget competitors."
                    )
                ]
            elif assigned_persona == "High-Spend Fiber Newcomers":
                actions = [
                    (
                        "15% Discount on 1-Year Contract Extension",
                        "First-year fiber subscribers (~11 mos) face promo expiration ('bill shock') on premium tiers. Contract extensions protect recurring revenue."
                    ),
                    (
                        "Complimentary 60 Days Premium Tech Support",
                        "Fiber subscribers report 77% higher churn velocity when lacking tech assistance. Fast-tracked technical support resolves early friction."
                    ),
                    (
                        "Proactive Gigabit Line Speed & Latency Audit",
                        "Automated line diagnostics confirm delivered speed matches advertised bandwidth, preventing defection caused by perceived throughput issues."
                    )
                ]
            else:
                # Safe retention or general low-risk cohort
                actions = [
                    (
                        "Quarterly Relationship Check-in",
                        "Account is currently within the safe retention threshold. Regular touchpoints sustain satisfaction and detect early emerging concerns."
                    ),
                    (
                        "Targeted Bundle & Security Cross-Sell",
                        "Satisfied, established customers have higher propensity for high-margin service additions like Device Protection or Online Backup."
                    ),
                    (
                        "Customer Advocacy & Referral Program Enrollment",
                        "Low churn risk accounts are prime candidates for referral reward programs with minimum acquisition overhead."
                    )
                ]

            for i, (act_title, act_reason) in enumerate(actions, 1):
                st.markdown(f"""
                <div class="action-card">
                    <div class="action-num">{i}</div>
                    <div class="action-body">
                        <div class="action-title">{act_title}</div>
                        <div class="action-reason">{act_reason}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # 5. Footer
    st.markdown("""
    <div class="footer-line">
        Practical 10 - ML Lab - TY Computer Engineering | Model: Gradient Boosting + SMOTE | Dataset: Telco Customer Churn (IBM)
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
