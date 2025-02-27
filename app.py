import streamlit as st
import os
import pandas as pd
import shutil
from src import linear_regression 
from src import processing
from src import svm_mnist
from src import decision_tree_mnist
from src import clustering
from src import mlflow_web
import mlflow
# Sidebar navigation
st.sidebar.title("App Selection")
option = st.sidebar.selectbox("Chọn lựa chọn phù hợp:", ["Pre Processing", "Linear Regression", "SVM Mnist", "Decision Tree Mnist",  "Clustering", "ML-Flow"])

if(option == 'Pre Processing'):
    processing.display()
elif(option == 'Linear Regression'):
    linear_regression.display()
elif(option == 'SVM Mnist'):
    svm_mnist.display()
elif(option == 'Decision Tree Mnist'):
    decision_tree_mnist.display()
elif(option == 'Clustering'):
    clustering.display()
elif(option == 'ML-Flow'):
    DAGSHUB_USERNAME = "NewbieHocIT"  # Thay bằng username của bạn
    DAGSHUB_REPO_NAME = "MocMayvsPython"
    DAGSHUB_TOKEN = "681dda9a41f9271a144aa94fa8624153a3c95696"  # Thay bằng Access Token của bạn

    # Đặt URI MLflow để trỏ đến DagsHub
    mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow")

    # Thiết lập authentication bằng Access Token
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
    try:
        # Lấy danh sách các thí nghiệm từ MLflow
        experiments = mlflow.search_experiments()

        if experiments:
            st.write("#### Danh sách thí nghiệm")
            experiment_data = []
            for exp in experiments:
                experiment_data.append({
                    "Experiment ID": exp.experiment_id,
                    "Experiment Name": exp.name,
                    "Artifact Location": exp.artifact_location
                })
            st.dataframe(pd.DataFrame(experiment_data))

            # Chọn thí nghiệm để xem chi tiết
            selected_exp_id = st.selectbox(
                "🔍 Chọn thí nghiệm để xem chi tiết",
                options=[exp.experiment_id for exp in experiments]
            )

            # Lấy danh sách runs trong thí nghiệm đã chọn
            runs = mlflow.search_runs(selected_exp_id)
            if not runs.empty:
                st.write("#### Danh sách runs")
                st.dataframe(runs)

                # Chọn run để xem chi tiết
                selected_run_id = st.selectbox(
                    "🔍 Chọn run để xem chi tiết",
                    options=runs["run_id"]
                )

                # Hiển thị chi tiết run
                run = mlflow.get_run(selected_run_id)
                st.write("##### Thông tin run")
                st.write(f"*Run ID:* {run.info.run_id}")
                st.write(f"*Experiment ID:* {run.info.experiment_id}")
                st.write(f"*Start Time:* {run.info.start_time}")

                # Hiển thị metrics
                st.write("##### Metrics")
                st.json(run.data.metrics)

                # Hiển thị params
                st.write("##### Params")
                st.json(run.data.params)

                # Hiển thị artifacts
                artifacts = mlflow.artifacts.list_artifacts(run.info.run_id)
                if artifacts:
                    st.write("##### Artifacts")
                    artifact_paths = [artifact.path for artifact in artifacts]
                    st.write(artifact_paths)

                    # Cho phép người dùng đổi tên file
                    st.write("#### Đổi tên artifacts")
                    for artifact in artifacts:
                        # Tạo một trường nhập tên mới cho từng file
                        new_name = st.text_input(f"Đổi tên cho file: {artifact.path}", artifact.path)
                        if new_name != artifact.path:
                            # Khi người dùng nhập tên mới, tải lại file từ MLflow và đổi tên
                            try:
                                # Tải file từ MLflow về hệ thống cục bộ
                                local_path = mlflow.artifacts.download_artifacts(run.info.run_id, artifact.path)
                                
                                # Tạo đường dẫn mới với tên mới
                                new_local_path = os.path.join(os.path.dirname(local_path), new_name)

                                # Đổi tên file cục bộ
                                shutil.move(local_path, new_local_path)

                                # Tải lại file với tên mới lên MLflow
                                mlflow.log_artifact(new_local_path)

                                # Xóa file cũ nếu cần thiết (file cũ sẽ không được tái sử dụng)
                                os.remove(local_path)

                                st.success(f"✅ Đã đổi tên thành công: {artifact.path} → {new_name}")
                            except Exception as e:
                                st.error(f"❌ Đã xảy ra lỗi khi đổi tên: {e}")
                else:
                    st.write("Không có artifacts nào.")
            else:
                st.warning("Không có runs nào trong thí nghiệm này.")
        else:
            st.warning("Không có thí nghiệm nào được tìm thấy.")
    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi lấy danh sách thí nghiệm: {e}")
