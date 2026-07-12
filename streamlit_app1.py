import json
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mlflow
import pandas as pd
import plotly.express as px
import streamlit as st

# --- Ensure these imports match your actual local file structure ---
from dashboard.config import (
    API_URL, EDA_DIR, EXTERNAL_API_URL, EXTERNAL_MLFLOW_URL,
    EXTERNAL_PREFECT_URL, LOG_FILES, MLFLOW_URL, MODEL_PATH,
    PREDICTION_DB, PREPROCESS_DIR, PIPELINE_STATUS_PATH, RAW_DATA_PATH,
)
from dashboard.services import (
    get_app_health, get_app_metadata, get_data_preview,
    get_eda_images, get_mlflow_client, get_mlflow_experiments,
    get_mlflow_run_metrics, get_mlflow_run_params, get_mlflow_runs,
    get_model_info, get_monitoring_stats, get_pipeline_status,
    get_prefect_status, get_preprocessing_report, get_eda_report,
    get_image, get_prediction_history, init_prediction_db,
    make_prediction, read_log, save_prediction,
)

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & THEME
# ---------------------------------------------------------
st.set_page_config(
    page_title="MLOps DataOps Dashboard",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Injecting Custom CSS inspired by the reference image
# Injecting Custom CSS for "Glassmorphism" Design
CUSTOM_CSS = """
<style>
    /* Global App Background - Deep Slate/Blue Mesh */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    
    /* Sidebar styling - Frosted Glass */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.4) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }

    /* Glassmorphism "Hero" Card */
    .hero-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 2.5rem;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
    }
    .hero-card h1 { 
        background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 0; 
        padding-top: 0;
        font-weight: 800;
    }
    .hero-card p { font-size: 1.1rem; color: #cbd5e1; margin-bottom: 0;}

    /* Metric Cards - Frosted Glass */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        background: rgba(255, 255, 255, 0.04);
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetricValue"] {
        color: #38bdf8; /* Neon Cyan Accent */
        font-size: 2rem;
        font-weight: 700;
    }

    /* Modern Styled Buttons - Ghost/Glass effect */
    .stButton > button {
        background: rgba(56, 189, 248, 0.1);
        backdrop-filter: blur(4px);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 12px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background: rgba(56, 189, 248, 0.25);
        border-color: rgba(56, 189, 248, 0.6);
        transform: translateY(-2px);
        color: #ffffff;
        box-shadow: 0 8px 20px rgba(56, 189, 248, 0.15);
    }

    /* Sub-headers */
    h2, h3 { color: #e2e8f0; font-weight: 600; }
    
    /* Dataframe backgrounds */
    [data-testid="stDataFrame"] { 
        background-color: rgba(0, 0, 0, 0.2); 
        border-radius: 12px; 
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
</style>
"""
# CUSTOM_CSS = """
# <style>
#     /* Global App Background - Deep Dark */
#     .stApp {
#         background-color: #121212; 
#         color: #ffffff;
#     }
    
#     /* Sidebar styling */
#     [data-testid="stSidebar"] {
#         background-color: #1e1e1e;
#         border-right: 1px solid #333;
#     }

#     /* Beautiful Gradient "Hero" Card mimicking the UI */
#     .hero-card {
#         background: linear-gradient(135deg, #514A9D, #24C6DC);
#         border-radius: 20px;
#         padding: 2rem;
#         color: white;
#         margin-bottom: 2rem;
#         box-shadow: 0 8px 16px rgba(0,0,0,0.3);
#     }
#     .hero-card h1 { color: white !important; margin-top: 0; padding-top: 0;}
#     .hero-card p { font-size: 1.1rem; opacity: 0.9; margin-bottom: 0;}

#     /* Metric Cards - Rounded with subtle borders */
#     [data-testid="stMetric"] {
#         background-color: #1e1e1e;
#         border: 1px solid #333;
#         border-radius: 16px;
#         padding: 20px;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#     }
#     [data-testid="stMetricLabel"] {
#         color: #a0a0a0;
#         font-weight: 600;
#     }
#     [data-testid="stMetricValue"] {
#         color: #6C63FF; /* Vibrant Purple Accent */
#         font-size: 1.8rem;
#         font-weight: 700;
#     }

#     /* Modern Styled Buttons */
#     .stButton > button {
#         background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
#         color: white;
#         border-radius: 12px;
#         border: none;
#         padding: 0.5rem 1rem;
#         font-weight: 600;
#         transition: all 0.3s ease;
#         width: 100%;
#         box-shadow: 0 4px 6px rgba(124, 58, 237, 0.2);
#     }
#     .stButton > button:hover {
#         transform: translateY(-2px);
#         box-shadow: 0 6px 12px rgba(124, 58, 237, 0.4);
#         color: white;
#         border: none;
#     }

#     /* Sub-headers */
#     h2, h3 { color: #f0f6fc; }
    
#     /* Dataframe backgrounds */
#     [data-testid="stDataFrame"] { background-color: #1e1e1e; border-radius: 12px;}
# </style>
# """
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. APP STATE & NAVIGATION
# ---------------------------------------------------------
SERVICE_URLS = {
    "API Docs": f"{EXTERNAL_API_URL}/docs",
    "Prefect UI": EXTERNAL_PREFECT_URL,
    "MLflow UI": EXTERNAL_MLFLOW_URL,
}

PAGES = [
    "Home", "Data Pipeline", "Exploratory Data Analysis", "ML Pipeline",
    "MLflow Experiments", "Prefect Workflows", "Prediction",
    "Monitoring", "Logs", "Application Details",
]

if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Home"

sidebar = st.sidebar
sidebar.image("https://cdn-icons-png.flaticon.com/512/8637/8637099.png", width=60) # Placeholder logo
sidebar.title("MLOps Platform")
selected_radio = sidebar.radio("Navigation", PAGES, index=PAGES.index(st.session_state.selected_page))
if selected_radio != st.session_state.selected_page:
    st.session_state.selected_page = selected_radio
page = st.session_state.selected_page

sidebar.markdown("---")
sidebar.caption("⚡ Built with Streamlit, FastAPI, MLflow, Prefect")
sidebar.caption("🔄 Auto-refreshes every 5 seconds")

def link_button(label: str, url: str) -> None:
    # A custom external link button that matches the internal glass button style
    html = f"""
    <a href="{url}" target="_blank" style="text-decoration: none;">
        <div style="background: rgba(56, 189, 248, 0.1); backdrop-filter: blur(4px);
                    color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3);
                    border-radius: 12px; padding: 10px 18px; 
                    text-align: center; font-weight: 600; transition: all 0.3s ease;
                    margin-bottom: 10px;">
            {label} ↗
        </div>
    </a>
    """
    st.markdown(html, unsafe_allow_html=True)
# def link_button(label: str, url: str) -> None:
#     # A custom external link button that matches the internal button style
#     html = f"""
#     <a href="{url}" target="_blank" style="text-decoration: none;">
#         <div style="background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%); 
#                     color: white; border-radius: 12px; padding: 10px 18px; 
#                     text-align: center; font-weight: 600; box-shadow: 0 4px 6px rgba(124, 58, 237, 0.2);
#                     margin-bottom: 10px;">
#             {label} ↗
#         </div>
#     </a>
#     """
    st.markdown(html, unsafe_allow_html=True)


def maybe_rerun() -> None:
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


if not PREDICTION_DB.exists():
    init_prediction_db()

# ---------------------------------------------------------
# 3. PAGE ROUTING & UI LAYOUTS
# ---------------------------------------------------------
if page == "Home":
    st.markdown("""
        <div class="hero-card">
            <h1>🏠 MLOps Dashboard</h1>
            <p>Enterprise-grade model operations platform. Monitor your pipelines, track experiments, and serve predictions seamlessly.</p>
        </div>
    """, unsafe_allow_html=True)

    metadata = get_app_metadata()
    health = get_app_health()
    pipeline_status = get_pipeline_status()

    st.subheader("System Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Project", "Heart Disease UCI")
    c2.metric("Dataset", "Cleveland Heart Disease")
    c3.metric("Model Ver", metadata.get("model_version", "unknown"))
    c4.metric("Total Preds", len(get_prediction_history()))
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("API Status", health.get("status", "unknown"))
    c6.metric("Pipeline Status", pipeline_status.get("pipeline_status", "unknown"))
    c7.metric("Last Run", pipeline_status.get("last_pipeline_run", "unknown"))
    c8.metric("Active Model", Path(MODEL_PATH).name if MODEL_PATH else "None")

    st.markdown("---")
    
    col_nav, col_ext = st.columns([2, 1], gap="large")
    
    with col_nav:
        st.write("### 🚀 Quick Actions")
        t1, t2, t3 = st.columns(3)
        if t1.button("📥 Data Pipeline"):
            st.session_state.selected_page = "Data Pipeline"
        if t2.button("📊 Explore EDA"):
            st.session_state.selected_page = "Exploratory Data Analysis"
        if t3.button("🤖 ML Pipeline"):
            st.session_state.selected_page = "ML Pipeline"
        
        st.markdown("<br>", unsafe_allow_html=True)
        t4, t5, t6 = st.columns(3)
        if t4.button("🎯 Prediction"):
            st.session_state.selected_page = "Prediction"
        if t5.button("📈 Monitoring"):
            st.session_state.selected_page = "Monitoring"
        if t6.button("📜 View Logs"):
            st.session_state.selected_page = "Logs"
        
        page = st.session_state.selected_page

    with col_ext:
        st.write("### 🔗 External Tools")
        link_button("API Docs (Swagger)", SERVICE_URLS["API Docs"])
        link_button("Prefect Dashboard", SERVICE_URLS["Prefect UI"])
        link_button("MLflow Tracking", SERVICE_URLS["MLflow UI"])

elif page == "Data Pipeline":
    st.title("📥 Data Pipeline")
    
    c1, c2, c3 = st.columns(3)
    with c1: link_button("Open Prefect UI", SERVICE_URLS["Prefect UI"])
    with c2: link_button("Open API Docs", SERVICE_URLS["API Docs"])

    st.markdown("---")
    df = get_data_preview()
    if not df.empty:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"**Shape:** {df.shape[0]} Rows x {df.shape[1]} Columns")
    else:
        st.info("Raw data not available.")

    col1, col2 = st.columns(2)
    with col1:
        st.write("### Summary Statistics")
        st.dataframe(get_preprocessing_report("summary_statistics.csv"), use_container_width=True)
    with col2:
        st.write("### Missing Values")
        st.dataframe(get_preprocessing_report("missing_values.csv"), use_container_width=True)
        
    st.write("### Data Types")
    st.dataframe(get_preprocessing_report("data_types.csv"))
    st.write("### Processing Status")
    status = get_pipeline_status()
    st.json(status)

elif page == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis")
    raw = get_data_preview()
    
    if raw.empty:
        st.warning("EDA data not available.")
    else:
        st.subheader("Histogram")
        fig = px.histogram(raw, x=raw.columns[0], title="Sample Histogram")
        # Update plotly layout to match dark theme
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Correlation Matrix")
        st.dataframe(get_eda_report("correlation_matrix.csv"), use_container_width=True)
        
        available = get_eda_images()
        for i in range(0, len(available), 2):
            cols = st.columns(2)
            for col_idx, (title, img_path) in enumerate(available[i:i + 2]):
                with cols[col_idx]:
                    st.markdown(
                        f"<div style='padding: 0.65rem 0.8rem; margin-bottom: 0.4rem; "
                        "background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); "
                        "border-radius: 10px;'><strong>{title}</strong></div>",
                        unsafe_allow_html=True,
                    )
                    st.image(img_path, use_container_width=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                
        binned = get_eda_report("binned_features.csv")
        if not binned.empty:
            st.subheader("Binned Features")
            st.dataframe(binned, use_container_width=True)
            
        categorical_corr = get_eda_report("categorical_correlation.csv")
        if not categorical_corr.empty:
            st.subheader("Categorical Correlation")
            st.dataframe(categorical_corr, use_container_width=True)

elif page == "ML Pipeline":
    st.title("🤖 ML Pipeline")
    
    c1, c2, c3 = st.columns(3)
    with c1: link_button("Open MLflow UI", SERVICE_URLS["MLflow UI"])
    with c2: link_button("Open API Docs", SERVICE_URLS["API Docs"])

    st.markdown("---")
    st.subheader("Model and Training Overview")
    st.markdown(f"**Best Model:** {Path(MODEL_PATH).name}")
    st.markdown(f"**Model Path:** {MODEL_PATH}")
    
    processed = get_data_preview()
    if not processed.empty:
        st.markdown(f"**Dataset Rows:** {processed.shape[0]}")
        
    st.write("### Evaluation Metrics")
    st.info("Use the MLflow Experiments page for full metrics visualization.")

elif page == "MLflow Experiments":
    st.title("📜 MLflow Experiments")
    st.write("Connects to MLflow tracking server directly.")
    
    c1, c2, c3 = st.columns(3)
    with c1: link_button("Open MLflow UI", SERVICE_URLS["MLflow UI"])
    with c2: link_button("Open API Docs", SERVICE_URLS["API Docs"])
    
    st.markdown("---")
    client = get_mlflow_client()
    experiments = get_mlflow_experiments()
    exp_names = [exp.get("name") for exp in experiments if isinstance(exp, dict)]
    
    experiment_name = st.selectbox("Select Experiment", exp_names)
    selected = next((exp for exp in experiments if exp.get("name") == experiment_name), None)
    
    if selected:
        st.write(selected)
        runs = get_mlflow_runs(selected.get("experiment_id"))
        st.dataframe(pd.DataFrame(runs), use_container_width=True)
        if runs:
            selected_run = runs[0]
            run_id = selected_run.get("info", {}).get("run_id")
            if run_id:
                st.write("### Run Metrics")
                st.json(get_mlflow_run_metrics(run_id))
                st.write("### Run Parameters")
                st.json(get_mlflow_run_params(run_id))

elif page == "Prefect Workflows":
    st.title("🔄 Prefect Workflows")
    
    c1, c2, c3 = st.columns(3)
    with c1: link_button("Open Prefect UI", SERVICE_URLS["Prefect UI"])
    
    st.markdown("---")
    st.write("### Prefect Status")
    status = get_prefect_status()
    st.json(status)

elif page == "Prediction":
    st.title("🎯 Live Prediction")
    st.markdown("Generate real-time Heart Disease risk predictions using the UCI feature schema.")
    st.info("This dashboard uses the Cleveland Heart Disease dataset for prediction and model training.")

    with st.form("prediction_form", clear_on_submit=False):
        st.write("### Prediction Input")
        cols = st.columns(3)
        age = cols[0].number_input("Age", min_value=0, max_value=120, value=53)
        sex = cols[1].selectbox("Sex", ["Female", "Male"])
        cp = cols[2].selectbox("Chest pain type", ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])

        trestbps = cols[0].number_input("Resting blood pressure", min_value=0.0, value=140.0)
        chol = cols[1].number_input("Cholesterol", min_value=0.0, value=233.0)
        fbs = cols[2].selectbox("Fasting blood sugar > 120 mg/dl", ["No", "Yes"])

        restecg = cols[0].selectbox("Resting ECG", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
        thalach = cols[1].number_input("Max heart rate", min_value=0.0, value=150.0)
        exang = cols[2].selectbox("Exercise induced angina", ["No", "Yes"])

        oldpeak = cols[0].number_input("Oldpeak", min_value=0.0, value=2.3)
        slope = cols[1].selectbox("Slope", ["Upsloping", "Flat", "Downsloping"])
        ca = cols[2].number_input("Major vessels colored", min_value=0, max_value=4, value=0)

        thal = cols[0].selectbox("Thalassemia", ["Normal", "Fixed defect", "Reversible defect"])

        st.markdown("<br>", unsafe_allow_html=True)
        submit_col, _ = st.columns([1, 3])
        with submit_col:
            submitted = st.form_submit_button("Generate Prediction", use_container_width=True)

    if submitted:
        payload = {
            "age": int(age),
            "sex": 1 if sex == "Male" else 0,
            "cp": ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"].index(cp),
            "trestbps": float(trestbps),
            "chol": float(chol),
            "fbs": 1 if fbs == "Yes" else 0,
            "restecg": ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"].index(restecg),
            "thalach": float(thalach),
            "exang": 1 if exang == "Yes" else 0,
            "oldpeak": float(oldpeak),
            "slope": ["Upsloping", "Flat", "Downsloping"].index(slope),
            "ca": int(ca),
            "thal": ["Normal", "Fixed defect", "Reversible defect"].index(thal),
        }
        with st.spinner("Running heart disease prediction..."):
            try:
                result = make_prediction(payload)
                risk = "High" if result.get("probability", 0) > 0.5 else "Low"
                record = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "age": age,
                    "sex": 1 if sex == "Male" else 0,
                    "cp": ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"].index(cp),
                    "trestbps": float(trestbps),
                    "chol": float(chol),
                    "fbs": 1 if fbs == "Yes" else 0,
                    "restecg": ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"].index(restecg),
                    "thalach": float(thalach),
                    "exang": 1 if exang == "Yes" else 0,
                    "oldpeak": float(oldpeak),
                    "slope": ["Upsloping", "Flat", "Downsloping"].index(slope),
                    "ca": int(ca),
                    "thal": ["Normal", "Fixed defect", "Reversible defect"].index(thal),
                    "prediction": result.get("churn_prediction", result.get("prediction", 0)),
                    "probability": result.get("probability"),
                    "risk_level": risk,
                }
                save_prediction(record)

                if risk == "High":
                    st.error(f"⚠️ **High Risk** | Probability: {record['probability']:.2f}")
                else:
                    st.success(f"✅ **Low Risk** | Probability: {record['probability']:.2f}")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    history = get_prediction_history()
    if not history.empty:
        st.markdown("---")
        st.subheader("Recent Predictions")
        st.dataframe(history.tail(5), use_container_width=True)

elif page == "Monitoring":
    st.title("📈 System Monitoring")
    stats = get_monitoring_stats()
    
    st.write("### Resource Utilization")
    col1, col2, col3 = st.columns(3)
    col1.metric("CPU Usage", f"{stats.get('cpu_percent', 0)}%")
    col2.metric("Memory Usage", f"{stats.get('memory_percent', 0)}%")
    col3.metric("Disk Usage", f"{stats.get('disk_percent', 0)}%")
    
    st.markdown("---")
    st.write("### Service Health")
    st.json({
        "API Status": stats.get('api_status', 'Unknown'),
        "MLflow Service": stats.get('mlflow_status', 'Unknown'),
        "Prefect Service": stats.get('prefect_status', 'Unknown'),
    })

    st.markdown("---")
    st.write("### Model Info")
    model_info = get_model_info()
    st.json(model_info)

    experiments = get_mlflow_experiments()
    if experiments:
        st.markdown("---")
        st.write("### Latest MLflow Run Metrics")
        latest_experiment_id = experiments[0].get("experiment_id") or experiments[0].get("experiment_id")
        if latest_experiment_id:
            runs = get_mlflow_runs(latest_experiment_id)
            if runs:
                latest_run = runs[0]
                metrics = latest_run.get("data", {}).get("metrics") if isinstance(latest_run, dict) else None
                if not metrics:
                    metrics = latest_run.get("metrics") if isinstance(latest_run, dict) else {}
                st.json(metrics or {"message": "No metrics found for latest run."})
            else:
                st.info("No MLflow runs available for the selected experiment.")
        else:
            st.info("No MLflow experiment id found.")
    else:
        st.info("No MLflow experiments detected yet.")

elif page == "Logs":
    st.title("📜 Logs")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        log_type = st.selectbox("Select log stream", list(LOG_FILES.keys()))
    with c2:
        level = st.selectbox("Level filter", ["", "INFO", "WARNING", "ERROR"])
    with c3:
        search = st.text_input("Search")
        
    lines = read_log(LOG_FILES[log_type], level_filter=level if level else None, search_text=search or None)
    
    # Styled text area using Streamlit's container
    st.text_area(f"{log_type}", "\n".join(lines[-100:]), height=400)

elif page == "Application Details":
    st.title("⚙ Application Details")
    metadata = get_app_metadata()
    st.json(metadata)