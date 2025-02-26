import streamlit as st
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import pytz
from datetime import datetime
import matplotlib.pyplot as plt
import dagshub
import os

# Hiển thị các mô hình đã log
def list_logged_models(id):
    client = mlflow.MlflowClient()
    runs = client.search_runs(experiment_ids=[id])
    gmt7 = pytz.timezone("Asia/Bangkok")
    df = pd.DataFrame([{ 
        "Run ID": r.info.run_id, 
        "Run Name": r.data.tags.get("mlflow.runName", "N/A"), 
        "Start Time": pd.to_datetime(r.info.start_time, unit='ms').tz_localize('UTC').tz_convert(gmt7).strftime('%Y-%m-%d %H:%M:%S'), 
        "Status": "✅ Hoàn thành" if r.info.status == "FINISHED" else "❌ Lỗi"
    } for r in runs])
    return df

def display():
    # Khởi tạo kết nối với DagsHub
    try:
        dagshub.init(repo_owner='NewbieHocIT', repo_name='MocMayvsPython', mlflow=True)
        os.environ["MLFLOW_TRACKING_URI"] = st.secrets["dagshub"]["MLFLOW_TRACKING_URI"]
        os.environ["MLFLOW_TRACKING_USERNAME"] = st.secrets["dagshub"]["MLFLOW_TRACKING_USERNAME"]
        os.environ["MLFLOW_TRACKING_PASSWORD"] = st.secrets["dagshub"]["MLFLOW_TRACKING_PASSWORD"]  
        mlflow.set_tracking_uri("https://dagshub.com/NewbieHocIT/MocMayvsPython.mlflow")
        client = MlflowClient()
        experiments = client.search_experiments()
        print('ok')
    except Exception as e:
        st.warning("Không thể kết nối với MLflow hoặc DagsHub. Vui lòng kiểm tra cài đặt.")
        experiments = []

    st.title("🚀 MLflow Model Logging & Registry")

    # Chọn thí nghiệm
    experiments = client.search_experiments()
    experiment_names = [exp.name for exp in experiments]
    selected_experiment = st.selectbox("📊 Chọn thí nghiệm", experiment_names)
    experiment_id = next(exp.experiment_id for exp in experiments if exp.name == selected_experiment)
    st.subheader("📌 Các mô hình đã log")
    models_df = list_logged_models(id=experiment_id)
    st.dataframe(models_df.style.set_properties(**{"background-color": "#f0f2f6", "color": "black"}), use_container_width=True)

    # Chọn các Run Name để so sánh
    st.subheader("📈 So sánh các mô hình")
    available_run_names = models_df["Run Name"].tolist()
    selected_run_names = st.multiselect("🔍 Chọn Run Name để so sánh", available_run_names)

    if selected_run_names:
        comparison_data = []
        for run_name in selected_run_names:
            run_info = models_df[models_df["Run Name"] == run_name].iloc[0]
            run_id = run_info["Run ID"]
            run = client.get_run(run_id)
            comparison_data.append({
                "Run ID": run_id,
                "Run Name": run.data.tags.get("mlflow.runName", "N/A"),
                "Validation Accuracy": run.data.metrics.get("Validation Accuracy", None),
                "Validation Precision": run.data.metrics.get("Validation Precision", None),
                "Validation Recall": run.data.metrics.get("Validation Recall", None),
                "Validation F1-Score": run.data.metrics.get("Validation F1-Score", None),
                "Test Accuracy": run.data.metrics.get("Test Accuracy", None),
                "Test Precision": run.data.metrics.get("Test Precision", None),
                "Test Recall": run.data.metrics.get("Test Recall", None),
                "Test F1-Score": run.data.metrics.get("Test F1-Score", None),
                "Silhouette Score": run.data.metrics.get("Silhouette Score", None),
            })
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df.style.set_properties(**{"background-color": "#e8f5e9", "color": "black"}), use_container_width=True)
        
        # Vẽ biểu đồ so sánh các metric
        st.subheader("📊 Biểu đồ so sánh các metric")
        selected_metric = st.selectbox("📌 Chọn metric để vẽ biểu đồ", ["Validation Accuracy", "Validation Precision", "Validation Recall", "Validation F1-Score", "Test Accuracy", "Test Precision", "Test Recall", "Test F1-Score", "Silhouette Score"])
        
        if selected_metric:
            fig, ax = plt.subplots()
            ax.bar(comparison_df["Run Name"], comparison_df[selected_metric], color='skyblue')
            ax.set_xlabel("Run Name", fontsize=8)
            ax.set_ylabel(selected_metric, fontsize=8)
            ax.set_title(f"So sánh {selected_metric}", fontsize=8)
            ax.tick_params(axis='x', rotation=0, labelsize=5)
            ax.tick_params(axis='y', labelsize=8)
            st.pyplot(fig)
    
display()
