import streamlit as st
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import pytz
import os
import matplotlib.pyplot as plt
import dagshub
import requests

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
        "Status": "✅ Hoàn thành" if r.info.status == "FINISHED" else "❌ Lỗi"
    } for r in runs])
    return df

def check_connection():
    """
    Kiểm tra kết nối đến DagsHub.
    """
    try:
        response = requests.get("https://dagshub.com/NewbieHocIT/MocMayvsPython.mlflow")
        if response.status_code == 200:
            st.success("Kết nối đến DagsHub thành công!")
        else:
            st.warning(f"Không thể kết nối đến DagsHub. Mã lỗi: {response.status_code}")
    except Exception as e:
        st.error(f"Lỗi khi kiểm tra kết nối: {e}")

def display():
    st.title("🚀 MLflow Model Logging & Registry")

    try:
        # Kiểm tra xem các khóa có tồn tại trong secrets không
        if "MLFLOW_TRACKING_URI" not in st.secrets or "MLFLOW_TRACKING_USERNAME" not in st.secrets or "MLFLOW_TRACKING_PASSWORD" not in st.secrets:
            st.error("❌ Thiếu thông tin xác thực trong secrets.toml. Vui lòng kiểm tra lại.")
            return

        # Lấy thông tin từ secrets.toml
        MLFLOW_TRACKING_URI = st.secrets["MLFLOW_TRACKING_URI"]
        MLFLOW_TRACKING_USERNAME = st.secrets["MLFLOW_TRACKING_USERNAME"]
        MLFLOW_TRACKING_PASSWORD = st.secrets["MLFLOW_TRACKING_PASSWORD"]

        # Khởi tạo kết nối với MLflow và DagsHub
        dagshub.init(repo_owner='NewbieHocIT', repo_name='MocMayvsPython', mlflow=True)
        os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
        os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

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

        # So sánh các mô hình
        st.subheader("📈 So sánh các mô hình")
        available_run_names = models_df["Run Name"].tolist()
        selected_run_names = st.multiselect("🔍 Chọn Run Name để so sánh", available_run_names)

        if selected_run_names:
            comparison_data = []
            for run_name in selected_run_names:
                run_info = models_df[models_df["Run Name"] == run_name].iloc[0]
                run_id = run_info["Run ID"]
                run = client.get_run(run_id)
                model_type = run.data.tags.get("model_type", "N/A")
                metrics = {"Run ID": run_id, "Run Name": run_name, "Model Type": model_type}
                metrics.update(run.data.metrics)
                comparison_data.append(metrics)

            comparison_df = pd.DataFrame(comparison_data)
            st.write("Dữ liệu so sánh:")
            st.dataframe(comparison_df, use_container_width=True)

            available_metrics = [col for col in comparison_df.columns if col not in ["Run ID", "Run Name", "Model Type"]]
            st.write("Các metric hợp lệ:", available_metrics)

            if available_metrics:
                selected_metric = st.selectbox("📌 Chọn metric để vẽ biểu đồ", available_metrics)

                if selected_metric:
                    comparison_df[selected_metric] = pd.to_numeric(comparison_df[selected_metric], errors='coerce')
                    valid_runs = comparison_df.dropna(subset=[selected_metric])
                    if not valid_runs.empty:
                        fig, ax = plt.subplots()
                        ax.bar(valid_runs["Run Name"], valid_runs[selected_metric], color='skyblue')
                        ax.set_xlabel("Run Name")
                        ax.set_ylabel(selected_metric)
                        ax.set_title(f"So sánh {selected_metric}")
                        ax.tick_params(axis='x', rotation=45)
                        st.pyplot(fig)
                        plt.close(fig)  # Đóng figure sau khi hiển thị
                    else:
                        st.warning(f"Không có dữ liệu {selected_metric} hợp lệ để vẽ biểu đồ.")
            else:
                st.warning("Không có metric nào để so sánh.")
    except Exception as e:
        st.error(f"❌ Đã xảy ra lỗi: {e}")
        st.warning("Vui lòng kiểm tra lại dữ liệu hoặc liên hệ quản trị viên.")

if __name__ == "__main__":
    display()
