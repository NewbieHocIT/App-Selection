import streamlit as st
from src import (
    linear_regression,
    processing,
    svm_mnist,
    decision_tree_mnist,
    clustering,
    mlflow_web
)

# Tiêu đề ứng dụng
st.sidebar.title("App Selection")

# Danh sách các tùy chọn
options = {
    "Pre Processing": processing.display,
    "Linear Regression": linear_regression.display,
    "SVM Mnist": svm_mnist.display,
    "Decision Tree Mnist": decision_tree_mnist.display,
    "Clustering": clustering.display,
    "ML-Flow": "mlflow"  # Đánh dấu tùy chọn ML-Flow
}

# Hiển thị dropdown để chọn tùy chọn
selected_option = st.sidebar.selectbox("Chọn lựa chọn phù hợp:", list(options.keys()))

# Xử lý lựa chọn
try:
    if selected_option in options:
        # Nếu tùy chọn là ML-Flow, khởi tạo MLflow trước khi gọi hàm
        if selected_option == "ML-Flow":
            import os
            import mlflow
            import dagshub

            def init_mlflow():
                try:
                    # Lấy thông tin từ secrets.toml
                    MLFLOW_TRACKING_URI = st.secrets["MLFLOW_TRACKING_URI"]
                    MLFLOW_TRACKING_USERNAME = st.secrets["MLFLOW_TRACKING_USERNAME"]
                    MLFLOW_TRACKING_PASSWORD = st.secrets["MLFLOW_TRACKING_PASSWORD"]

                    if MLFLOW_TRACKING_URI and MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD:
                        dagshub.init(repo_owner='NewbieHocIT', repo_name='MocMayvsPython', mlflow=True)
                        os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
                        os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
                        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
                        st.success("Kết nối MLflow thành công!")
                    else:
                        st.warning("Không thể kết nối với MLflow hoặc DagsHub. Vui lòng kiểm tra cài đặt.")
                except Exception as e:
                    st.error(f"Lỗi khi kết nối MLflow: {e}")

            # Khởi tạo MLflow
            init_mlflow()

            # Gọi hàm hiển thị của ML-Flow
            mlflow_web.display()
        else:
            # Gọi hàm tương ứng với tùy chọn được chọn
            options[selected_option]()
    else:
        st.warning("Lựa chọn không hợp lệ. Vui lòng chọn lại.")
except Exception as e:
    st.error(f"Đã xảy ra lỗi: {e}")
    st.warning("Vui lòng kiểm tra lại code hoặc liên hệ quản trị viên.")
