import streamlit as st
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import pytz
import os
import dagshub

@st.cache_resource
def initialize_dagshub():
    """
    Khởi tạo kết nối với DagsHub và MLflow.
    """
    dagshub.init(repo_owner='NewbieHocIT', repo_name='MocMayvsPython', mlflow=True)
    os.environ['MLFLOW_TRACKING_USERNAME'] = st.secrets["MLFLOW_TRACKING_USERNAME"]
    os.environ['MLFLOW_TRACKING_PASSWORD'] = st.secrets["MLFLOW_TRACKING_PASSWORD"]
    mlflow.set_tracking_uri(st.secrets["MLFLOW_TRACKING_URI"])

def list_logged_models(experiment_id):
    """
    Lấy danh sách các mô hình đã log trong một thí nghiệm.
    """
    client = MlflowClient()
    runs = client.search_runs(experiment_ids=[experiment_id])
    gmt7 = pytz.timezone("Asia/Bangkok")
    df = pd.DataFrame([{ 
        "Run ID": r.info.run_id, 
        "Run Name": r.data.tags.get("mlflow.runName", "N/A"), 
        "Model Type": r.data.tags.get("model_type", "N/A"),  
        "Start Time": pd.to_datetime(r.info.start_time, unit='ms')
                        .tz_localize('UTC')
                        .tz_convert(gmt7)
                        .strftime('%Y-%m-%d %H:%M:%S'), 
        "Status": "✅ Hoàn thành" if r.info.status == "FINISHED" else "❌ Lỗi",
        "Parameters": r.data.params,
        "Metrics": r.data.metrics
    } for r in runs])
    return df

def display():
    st.title("🚀 MLflow Model Logging & Registry")

    try:
        # Khởi tạo kết nối với DagsHub và MLflow
        initialize_dagshub()
        st.success("✅ Kết nối MLflow thành công!")
    except Exception as e:
        st.error(f"❌ Lỗi khi kết nối MLflow hoặc DagsHub: {e}")
        st.warning("Vui lòng kiểm tra lại cài đặt hoặc thông tin xác thực.")
        return

    try:
        # Lấy danh sách thí nghiệm
        client = MlflowClient()
        experiments = client.search_experiments()
        if not experiments:
            st.warning("Không tìm thấy thí nghiệm nào.")
            return

        # Hiển thị danh sách thí nghiệm
        experiment_names = [exp.name for exp in experiments]
        selected_experiment = st.selectbox("📊 Chọn thí nghiệm", experiment_names)
        experiment_id = next(exp.experiment_id for exp in experiments if exp.name == selected_experiment)

        # Hiển thị danh sách các mô hình đã log
        st.subheader("📌 Các mô hình đã log")
        models_df = list_logged_models(experiment_id)
        st.dataframe(models_df, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Đã xảy ra lỗi: {e}")
        st.warning("Vui lòng kiểm tra lại dữ liệu hoặc liên hệ quản trị viên.")

if __name__ == "__main__":
    display()
