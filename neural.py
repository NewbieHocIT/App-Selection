import tensorflow as tf
import tensorflow as tf
Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Adam = tf.keras.optimizers.Adam


import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
import pandas as pd
import os
import mlflow
from datetime import datetime
from sklearn.model_selection import cross_val_score
from streamlit_drawable_canvas import st_canvas


def load_mnist():
    X = np.load("data/mnist/X.npy")
    y = np.load("data/mnist/y.npy")
    return X, y

def data():
    st.header("MNIST Dataset")
    st.write("""
      **MNIST** là một trong những bộ dữ liệu nổi tiếng và phổ biến nhất trong cộng đồng học máy, 
      đặc biệt là trong các nghiên cứu về nhận diện mẫu và phân loại hình ảnh.
  
      - Bộ dữ liệu bao gồm tổng cộng **70.000 ảnh chữ số viết tay** từ **0** đến **9**, 
        mỗi ảnh có kích thước **28 x 28 pixel**.
      - Chia thành:
        - **Training set**: 60.000 ảnh để huấn luyện.
        - **Test set**: 10.000 ảnh để kiểm tra.
      - Mỗi hình ảnh là một chữ số viết tay, được chuẩn hóa và chuyển thành dạng grayscale (đen trắng).
  
      Dữ liệu này được sử dụng rộng rãi để xây dựng các mô hình nhận diện chữ số.
      """)

    st.subheader("Một số hình ảnh từ MNIST Dataset")
    st.image("mnit.png", caption="Một số hình ảnh từ MNIST Dataset", use_container_width=True)

    st.subheader("Ứng dụng thực tế của MNIST")
    st.write("""
      Bộ dữ liệu MNIST đã được sử dụng trong nhiều ứng dụng nhận dạng chữ số viết tay, chẳng hạn như:
      - Nhận diện số trên các hoá đơn thanh toán, biên lai cửa hàng.
      - Xử lý chữ số trên các bưu kiện gửi qua bưu điện.
      - Ứng dụng trong các hệ thống nhận diện tài liệu tự động.
    """)

    st.subheader("Ví dụ về các mô hình học máy với MNIST")
    st.write("""
      Các mô hình học máy phổ biến đã được huấn luyện với bộ dữ liệu MNIST bao gồm:
      - **Logistic Regression**
      - **Decision Trees**
      - **K-Nearest Neighbors (KNN)**
      - **Support Vector Machines (SVM)**
      - **Convolutional Neural Networks (CNNs)**
    """)

def split_data():
    st.title("📌 Chia dữ liệu Train/Test")

    # Đọc dữ liệu
    X, y = load_mnist() 
    total_samples = X.shape[0]

    # Nếu chưa có cờ "data_split_done", đặt mặc định là False
    if "data_split_done" not in st.session_state:
        st.session_state.data_split_done = False  

    # Thanh kéo chọn số lượng ảnh để train
    num_samples = st.number_input("📌 Nhập số lượng ảnh để train:", min_value=1000, max_value=70000, value=20000, step=1000)
    
    # Thanh kéo chọn tỷ lệ Train/Test
    test_size = st.slider("📌 Chọn % dữ liệu Test", 10, 50, 20)
    remaining_size = 100 - test_size
    val_size = st.slider("📌 Chọn % dữ liệu Validation (trong phần Train)", 0, 50, 15)
    st.write(f"📌 **Tỷ lệ phân chia:** Test={test_size}%, Validation={val_size}%, Train={remaining_size - val_size}%")

    # Placeholder để hiển thị bảng
    table_placeholder = st.empty()

    # Nút xác nhận và lưu dữ liệu lần đầu
    if st.button("✅ Xác nhận & Lưu", key="luu"):
        st.session_state.data_split_done = True  # Đánh dấu đã chia dữ liệu
        
        if num_samples == total_samples:
            X_selected, y_selected = X, y
        else:
            X_selected, _, y_selected, _ = train_test_split(
                X, y, train_size=num_samples, stratify=y, random_state=42
            )

        # Chia train/test
        stratify_option = y_selected if len(np.unique(y_selected)) > 1 else None
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_selected, y_selected, test_size=test_size/100, stratify=stratify_option, random_state=42
        )

        # Chia train/val
        if val_size > 0:
            stratify_option = y_train_full if len(np.unique(y_train_full)) > 1 else None
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full, test_size=val_size / (100 - test_size),
                stratify=stratify_option, random_state=42
            )
        else:
            X_train, y_train = X_train_full, y_train_full
            X_val, y_val = np.array([]), np.array([])  # Validation rỗng nếu val_size = 0

        # Lưu dữ liệu vào session_state
        st.session_state.total_samples = num_samples
        st.session_state["neural_X_train"] = X_train
        st.session_state["neural_X_val"] = X_val
        st.session_state["neural_X_test"] = X_test
        st.session_state["neural_y_train"] = y_train
        st.session_state["neural_y_val"] = y_val
        st.session_state["neural_y_test"] = y_test
        st.session_state.test_size = X_test.shape[0]
        st.session_state.val_size = X_val.shape[0]
        st.session_state.train_size = X_train.shape[0]

        # Lưu bảng kết quả vào session_state và hiển thị trong placeholder
        st.session_state.summary_df = pd.DataFrame({
            "Tập dữ liệu": ["Train", "Validation", "Test"],
            "Số lượng mẫu": [X_train.shape[0], X_val.shape[0], X_test.shape[0]]
        })
        st.success("✅ Dữ liệu đã được chia thành công!")
        table_placeholder.table(st.session_state.summary_df)

    # Nếu dữ liệu đã được chia trước đó
    if st.session_state.data_split_done:
        # Hiển thị bảng cũ trong placeholder nếu chưa nhấn "Chia lại dữ liệu"
        if "summary_df" in st.session_state:
            table_placeholder.table(st.session_state.summary_df)
        
        st.info("✅ Dữ liệu đã được chia. Nhấn nút dưới đây để chia lại dữ liệu nếu muốn thay đổi.")
        
        # Nút chia lại dữ liệu
        if st.button("🔄 Chia lại dữ liệu", key="chia_lai"):
            # Xóa nội dung placeholder trước khi chia lại
            table_placeholder.empty()
            
            if num_samples == total_samples:
                X_selected, y_selected = X, y
            else:
                X_selected, _, y_selected, _ = train_test_split(
                    X, y, train_size=num_samples, stratify=y, random_state=42
                )

            # Chia train/test
            stratify_option = y_selected if len(np.unique(y_selected)) > 1 else None
            X_train_full, X_test, y_train_full, y_test = train_test_split(
                X_selected, y_selected, test_size=test_size/100, stratify=stratify_option, random_state=42
            )

            # Chia train/val
            if val_size > 0:
                stratify_option = y_train_full if len(np.unique(y_train_full)) > 1 else None
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_full, y_train_full, test_size=val_size / (100 - test_size),
                    stratify=stratify_option, random_state=42
                )
            else:
                X_train, y_train = X_train_full, y_train_full
                X_val, y_val = np.array([]), np.array([])  # Validation rỗng nếu val_size = 0

            # Lưu dữ liệu vào session_state
            st.session_state.total_samples = num_samples
            st.session_state["neural_X_train"] = X_train
            st.session_state["neural_X_val"] = X_val
            st.session_state["neural_X_test"] = X_test
            st.session_state["neural_y_train"] = y_train
            st.session_state["neural_y_val"] = y_val
            st.session_state["neural_y_test"] = y_test
            st.session_state.test_size = X_test.shape[0]
            st.session_state.val_size = X_val.shape[0]
            st.session_state.train_size = X_train.shape[0]

            # Lưu và hiển thị bảng kết quả mới trong placeholder
            st.session_state.summary_df = pd.DataFrame({
                "Tập dữ liệu": ["Train", "Validation", "Test"],
                "Số lượng mẫu": [X_train.shape[0], X_val.shape[0], X_test.shape[0]]
            })
            st.success("✅ Dữ liệu đã được chia lại thành công!")
            table_placeholder.table(st.session_state.summary_df)

def mlflow_input():
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/NewbieHocIT/MocMayvsPython.mlflow"
    st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)

    os.environ["MLFLOW_TRACKING_USERNAME"] = "NewbieHocIT"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "681dda9a41f9271a144aa94fa8624153a3c95696"

    mlflow.set_experiment("neural")
from tensorflow.python.keras.engine import data_adapter

def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)

data_adapter._is_distributed_dataset = _is_distributed_dataset

from tensorflow.python.keras.callbacks import Callback

# Callback tùy chỉnh để cập nhật thanh tiến trình cho huấn luyện
class ProgressBarCallback(Callback):
    def __init__(self, total_epochs, progress_bar, status_text, max_train_progress=80):
        super(ProgressBarCallback, self).__init__()
        self.total_epochs = total_epochs
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.max_train_progress = max_train_progress  # Giới hạn tiến trình huấn luyện (80%)

    def on_epoch_begin(self, epoch, logs=None):
        progress = (epoch + 1) / self.total_epochs * self.max_train_progress
        self.progress_bar.progress(min(int(progress), self.max_train_progress))
        self.status_text.text(f"🛠️ Đang huấn luyện mô hình... Epoch {epoch + 1}/{self.total_epochs}")

    def on_train_end(self, logs=None):
        self.progress_bar.progress(self.max_train_progress)
        self.status_text.text("✅ Huấn luyện mô hình hoàn tất, đang chuẩn bị logging...")



# Hàm thực hiện Pseudo Labelling
# Hàm thực hiện Pseudo Labelling
# Hàm thực hiện Pseudo Labelling
def pseudo_labeling():
    st.header("⚙️ Pseudo Labelling với Neural Network")

    # Kiểm tra dữ liệu đã được chia chưa
    if "neural_X_train" not in st.session_state or "neural_X_test" not in st.session_state:
        st.error("⚠️ Chưa có dữ liệu! Hãy chia dữ liệu trước.")
        return

    # Lấy dữ liệu từ session_state
    X_train_full = st.session_state["neural_X_train"]
    y_train_full = st.session_state["neural_y_train"]
    X_test = st.session_state["neural_X_test"]
    y_test = st.session_state["neural_y_test"]

    # Chuẩn hóa dữ liệu
    X_train_full = X_train_full.reshape(-1, 28 * 28) / 255.0
    X_test = X_test.reshape(-1, 28 * 28) / 255.0

    # (1) Lấy 1% dữ liệu ban đầu cho mỗi class
    st.subheader("Bước 1: Lấy 1% dữ liệu có nhãn ban đầu")
    X_initial = []
    y_initial = []
    for digit in range(10):
        indices = np.where(y_train_full == digit)[0]
        num_samples_per_class = max(1, int(0.01 * len(indices)))
        selected_indices = np.random.choice(indices, size=num_samples_per_class, replace=False)
        X_initial.append(X_train_full[selected_indices])
        y_initial.append(y_train_full[selected_indices])

    X_initial = np.concatenate(X_initial, axis=0)
    y_initial = np.concatenate(y_initial, axis=0)

    # Hiển thị thông tin
    st.write(f"✅ Số lượng mẫu ban đầu: {X_initial.shape[0]} (khoảng 1% của {y_train_full.shape[0]} mẫu)")
    mask = np.ones(len(X_train_full), dtype=bool)
    mask[np.concatenate([np.where(y_train_full == digit)[0][:max(1, int(0.01 * len(np.where(y_train_full == digit)[0])))] for digit in range(10)])] = False
    X_unlabeled = X_train_full[mask]
    st.write(f"✅ Số lượng mẫu chưa gán nhãn: {X_unlabeled.shape[0]}")

    # Tham số người dùng chọn
    num_layers = st.slider("Số lớp ẩn", 1, 5, 2, key="pseudo_num_layers")
    num_nodes = st.slider("Số node mỗi lớp", 32, 256, 128, key="pseudo_num_nodes")
    activation = st.selectbox("Hàm kích hoạt", ["relu", "sigmoid", "tanh"], key="pseudo_activation")
    epochs = st.slider("Số epoch mỗi vòng", 1, 50, 10, key="pseudo_epochs")
    threshold = st.slider("Ngưỡng gán nhãn (threshold)", 0.5, 1.0, 0.95, step=0.01, key="pseudo_threshold")
    learn_rate = st.number_input(
        "Tốc độ học (learning rate)", 
        min_value=0.0001, 
        max_value=0.1, 
        value=0.001, 
        step=0.0001, 
        format="%.4f",
        key="pseudo_learn_rate"
    )
    
    # Tùy chọn chế độ lặp
    iteration_mode = st.selectbox("Chọn chế độ lặp:", ["Số vòng lặp cố định", "Gán hết toàn bộ tập train"], key="pseudo_iteration_mode")
    if iteration_mode == "Số vòng lặp cố định":
        max_iterations = st.slider("Số vòng lặp tối đa", 1, 10, 5, key="pseudo_max_iter")
    else:
        max_iterations = None
        st.warning("⚠️ Thời gian sẽ lâu do có thể lặp nhiều khi chọn 'Gán hết toàn bộ tập train'!")

    # Nhập tên run cho MLflow
    run_name = st.text_input("🔹 Nhập tên Run:", "Pseudo_Default_Run", key="pseudo_run_name_input")

    # Khởi tạo mô hình với learn_rate
    def build_model():
        model = Sequential()
        model.add(Dense(num_nodes, input_shape=(28 * 28,), activation=activation))
        for _ in range(num_layers - 1):
            model.add(Dense(num_nodes, activation=activation))
            model.add(Dropout(0.2))
        model.add(Dense(10, activation='softmax'))
        model.compile(optimizer=Adam(learning_rate=learn_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    # Bắt đầu quá trình Pseudo Labelling
    if st.button("Bắt đầu Pseudo Labelling", key="pseudo_start"):
        mlflow_input()  # Cài đặt MLflow
        with mlflow.start_run(run_name=f"Pseudo_{run_name}"):
            X_labeled = X_initial.copy()
            y_labeled = y_initial.copy()
            X_unlabeled_remaining = X_unlabeled.copy()

            # Thanh tiến trình tổng quát
            progress_bar = st.progress(0)
            status_text = st.empty()
            total_samples = X_train_full.shape[0]
            initial_labeled_samples = X_initial.shape[0]

            iteration = 0
            while True:
                iteration += 1
                if max_iterations is not None and iteration > max_iterations:
                    st.write(f"Đã đạt số vòng lặp tối đa: {max_iterations}")
                    break

                st.write(f"### Vòng lặp {iteration}")

                # (2) Huấn luyện mô hình
                model = build_model()
                progress_callback = ProgressBarCallback(epochs, progress_bar, status_text, max_train_progress=50)
                model.fit(X_labeled, y_labeled, epochs=epochs, callbacks=[progress_callback], verbose=0)

                # Đánh giá trên tập test
                status_text.text("📊 Đang đánh giá mô hình trên test set...")
                progress_bar.progress(60)
                test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
                st.write(f"Độ chính xác trên tập test: {test_acc:.4f}")
                mlflow.log_metric(f"test_accuracy_iter_{iteration}", test_acc)

                # (3) Dự đoán nhãn
                if X_unlabeled_remaining.shape[0] == 0:
                    st.write("✅ Đã gán nhãn hết dữ liệu!")
                    break

                status_text.text("🔍 Đang dự đoán nhãn cho dữ liệu chưa gán...")
                progress_bar.progress(70)
                probabilities = model.predict(X_unlabeled_remaining)
                max_probs = np.max(probabilities, axis=1)
                predicted_labels = np.argmax(probabilities, axis=1)

                # (4) Gán nhãn giả
                confident_mask = max_probs >= threshold
                X_confident = X_unlabeled_remaining[confident_mask]
                y_confident = predicted_labels[confident_mask]

                st.write(f"Số mẫu được gán nhãn giả: {X_confident.shape[0]} (ngưỡng: {threshold})")
                st.write(f"Số mẫu chưa gán nhãn còn lại: {X_unlabeled_remaining.shape[0] - X_confident.shape[0]}")

                # (5) Thêm dữ liệu mới gán nhãn
                X_labeled = np.concatenate([X_labeled, X_confident])
                y_labeled = np.concatenate([y_labeled, y_confident])
                X_unlabeled_remaining = X_unlabeled_remaining[~confident_mask]

                # Cập nhật tiến trình
                labeled_fraction = X_labeled.shape[0] / total_samples
                progress_bar.progress(min(int(70 + 25 * labeled_fraction), 95))
                status_text.text(f"📈 Đã gán nhãn: {X_labeled.shape[0]}/{total_samples} mẫu ({labeled_fraction:.2%})")

                # Logging
                mlflow.log_metric(f"labeled_samples_iter_{iteration}", X_labeled.shape[0])
                mlflow.log_metric(f"unlabeled_samples_iter_{iteration}", X_unlabeled_remaining.shape[0])

                if X_unlabeled_remaining.shape[0] == 0:
                    st.write("✅ Hoàn tất: Đã gán nhãn toàn bộ tập train!")
                    break
                if X_confident.shape[0] == 0:
                    st.write("✅ Hoàn tất: Không còn mẫu nào vượt ngưỡng!")
                    break

            # Lưu mô hình cuối cùng
            status_text.text("💾 Đang lưu mô hình và logging...")
            progress_bar.progress(95)
            model_path = f"pseudo_model_final.h5"
            model.save(model_path)
            mlflow.log_artifact(model_path)

            # Lưu mô hình vào session_state
            if "neural_models" not in st.session_state:
                st.session_state["neural_models"] = []
            model_name = f"pseudo_{num_layers}layers_{num_nodes}nodes_{activation}"
            st.session_state["neural_models"].append({"name": model_name, "model": model})
            st.success(f"✅ Mô hình Pseudo Labelling đã được lưu với tên: {model_name}")

            # Hoàn tất tiến trình
            progress_bar.progress(100)
            status_text.text("✅ Quá trình Pseudo Labelling hoàn tất!")

# Hàm train() (đặt sau pseudo_labeling() nếu cần)
def train():
    mlflow_input()

    if (
        "neural_X_train" not in st.session_state
        or "neural_X_val" not in st.session_state
        or "neural_X_test" not in st.session_state
    ):
        st.error("⚠️ Chưa có dữ liệu! Hãy chia dữ liệu trước.")
        return

    st.header("⚙️ Chọn mô hình & Huấn luyện")
    training_mode = st.selectbox("Chọn chế độ huấn luyện:", ["Regular Neural Network", "Pseudo Labelling"], key="train_mode")

    if training_mode == "Regular Neural Network":
        X_train = st.session_state["neural_X_train"]
        X_val = st.session_state["neural_X_val"]
        X_test = st.session_state["neural_X_test"]
        y_train = st.session_state["neural_y_train"]
        y_val = st.session_state["neural_y_val"]
        y_test = st.session_state["neural_y_test"]

        X_train = X_train.reshape(-1, 28 * 28) / 255.0
        X_test = X_test.reshape(-1, 28 * 28) / 255.0
        if X_val.size > 0:
            X_val = X_val.reshape(-1, 28 * 28) / 255.0

        num_layers = st.slider("Số lớp ẩn", 1, 5, 2, key="neural_num_layers_slider")
        num_nodes = st.slider("Số node mỗi lớp", 32, 256, 128, key="neural_num_nodes_slider")
        activation = st.selectbox("Hàm kích hoạt", ["relu", "sigmoid", "tanh"], key="neural_activation_selectbox")
        epochs = st.slider("Số epoch", 1, 50, 10, key="neural_epochs_slider")
        learn_rate = st.number_input(
        "Tốc độ học (learning rate)", 
        min_value=0.0001, 
        max_value=0.1, 
        value=0.001, 
        step=0.0001, 
        format="%.4f",  # Hiển thị 4 chữ số thập phân
        key="NN_learn_rate"
    )

        model = Sequential()
        model.add(Dense(num_nodes, input_shape=(28 * 28,), activation=activation))
        for _ in range(num_layers - 1):
            model.add(Dense(num_nodes, activation=activation))
            model.add(Dropout(0.1))
        model.add(Dense(10, activation='softmax'))
        model.compile(optimizer=Adam(learning_rate=learn_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        run_name = st.text_input("🔹 Nhập tên Run:", "Default_Run", key="neural_run_name_input")
        if st.button("Huấn luyện mô hình", key="neural_train_button"):
            with mlflow.start_run(run_name=f"Train_{run_name}"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                progress_callback = ProgressBarCallback(epochs, progress_bar, status_text)
                if X_val.size > 0:
                    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), callbacks=[progress_callback], verbose=0)
                else:
                    history = model.fit(X_train, y_train, epochs=epochs, callbacks=[progress_callback], verbose=0)

                test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
                st.success(f"✅ **Độ chính xác trên test set**: {test_acc:.4f}")
                mlflow.log_metric("test_accuracy", test_acc)

                model_path = f"model_{run_name}.h5"
                model.save(model_path)
                mlflow.log_artifact(model_path)

                model_name = f"neural_{num_layers}layers_{num_nodes}nodes_{activation}"
                if "neural_models" not in st.session_state:
                    st.session_state["neural_models"] = []
                st.session_state["neural_models"].append({"name": model_name, "model": model})
                st.success(f"✅ Mô hình đã được lưu với tên: {model_name}")

    elif training_mode == "Pseudo Labelling":
        pseudo_labeling()
            
def preprocess_image(image):
    """Xử lý ảnh đầu vào: Chuyển về grayscale, resize, chuẩn hóa"""
    image = image.convert("L")
    image = image.resize((28, 28))
    img_array = np.array(image) / 255.0
    # Đảo ngược màu nếu cần (nếu dữ liệu huấn luyện có nền đen)
    img_array = 1 - img_array  # Thử thêm dòng này
    st.image(img_array.reshape(28, 28), caption="Ảnh sau khi xử lý", clamp=True)
    return img_array.reshape(1, -1)

def du_doan():
    st.title("🔢 Dự đoán chữ số viết tay")

    # Kiểm tra xem đã có mô hình chưa
    if "neural_models" not in st.session_state or not st.session_state["neural_models"]:
        st.error("⚠️ Chưa có mô hình nào được huấn luyện. Hãy huấn luyện mô hình trước.")
        return

    # Chọn mô hình
    model_names = [model["name"] for model in st.session_state["neural_models"]]
    selected_model_name = st.selectbox("🔍 Chọn mô hình đã huấn luyện:", model_names)
    selected_model = next(model["model"] for model in st.session_state["neural_models"] if model["name"] == selected_model_name)

    # Chọn cách nhập ảnh: Tải lên hoặc Vẽ
    option = st.radio("📌 Chọn cách nhập ảnh:", ["🖼️ Tải ảnh lên", "✍️ Vẽ số"], key="input_option_radio")

    img_array = None  # Khởi tạo ảnh đầu vào

    # 1️⃣ 🖼️ Nếu tải ảnh lên
    if option == "🖼️ Tải ảnh lên":
        uploaded_file = st.file_uploader("📤 Tải ảnh chữ số viết tay (28x28 pixel)", type=["png", "jpg", "jpeg"], key="upfile")
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Ảnh đã tải lên", use_container_width=True)
            img_array = preprocess_image(image)  # Xử lý ảnh

    # 2️⃣ ✍️ Nếu vẽ số
    else:
        st.write("🎨 Vẽ số trong khung dưới đây:")
        
        # Canvas để vẽ
        canvas_result = st_canvas(
            fill_color="black",  # Màu nền
            stroke_width=10,
            stroke_color="black",
            background_color="white",
            height=250,
            width=250,
            drawing_mode="freedraw",
            key="canvas_draw"
        )

        # Khi người dùng bấm "Dự đoán"
        if st.button("Dự đoán số", key="dudoan"):
            if canvas_result.image_data is not None:
                # Chuyển đổi ảnh từ canvas thành định dạng PIL
                image = Image.fromarray((canvas_result.image_data[:, :, :3]).astype(np.uint8))
                img_array = preprocess_image(image)  # Xử lý ảnh
            else:
                st.error("⚠️ Hãy vẽ một số trước khi dự đoán!")

    # 🔍 Dự đoán nếu có ảnh đầu vào hợp lệ
    if img_array is not None:
        prediction = np.argmax(selected_model.predict(img_array), axis=1)[0]
        probabilities = selected_model.predict(img_array)[0]  # Lấy toàn bộ xác suất của các lớp

        # 🏆 Hiển thị kết quả dự đoán
        st.success(f"🔢 Dự đoán: **{prediction}**")

        # 📊 Hiển thị toàn bộ độ tin cậy theo từng lớp
        st.write("### 🔢 Độ tin cậy :")

        # 📊 Vẽ biểu đồ độ tin cậy
        st.bar_chart(probabilities)

def show_experiment_selector():
    st.title("📊 MLflow Experiments")

    experiment_name = "neural"
    
    # Lấy danh sách experiment
    experiments = mlflow.search_experiments()
    selected_experiment = next((exp for exp in experiments if exp.name == experiment_name), None)

    if not selected_experiment:
        st.error(f"❌ Experiment '{experiment_name}' không tồn tại!")
        return

    st.subheader(f"📌 Experiment: {experiment_name}")
    st.write(f"**Experiment ID:** {selected_experiment.experiment_id}")
    st.write(f"**Trạng thái:** {'Active' if selected_experiment.lifecycle_stage == 'active' else 'Deleted'}")
    st.write(f"**Vị trí lưu trữ:** {selected_experiment.artifact_location}")

    # Lấy danh sách runs trong experiment
    runs = mlflow.search_runs(experiment_ids=[selected_experiment.experiment_id])

    if runs.empty:
        st.warning("⚠ Không có runs nào trong experiment này.")
        return

    st.write("### 🏃‍♂️ Các Runs gần đây:")

    # Tạo danh sách run name và map với run_id
    run_dict = {}
    for _, run in runs.iterrows():
        run_name = run.get("tags.mlflow.runName", f"Run {run['run_id'][:8]}")
        run_dict[run_name] = run["run_id"]  # Map run_name -> run_id

    # Chọn run theo tên
    selected_run_name = st.selectbox("🔍 Chọn một run:", list(run_dict.keys()),key="runname")
    selected_run_id = run_dict[selected_run_name]

    # Lấy thông tin của run đã chọn
    selected_run = mlflow.get_run(selected_run_id)

    if selected_run:
        st.subheader(f"📌 Thông tin Run: {selected_run_name}")
        st.write(f"**Run ID:** {selected_run_id}")
        st.write(f"**Trạng thái:** {selected_run.info.status}")
        
        start_time_ms = selected_run.info.start_time
        start_time = datetime.fromtimestamp(start_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S") if start_time_ms else "Không có thông tin"

        st.write(f"**Thời gian chạy:** {start_time}")

        params = selected_run.data.params
        metrics = selected_run.data.metrics

        if params:
            st.write("### ⚙️ Parameters:")
            st.json(params)

        if metrics:
            st.write("### 📊 Metrics:")
            st.json(metrics)

    else:
        st.warning("⚠ Không tìm thấy thông tin cho run này.")
        
# Cập nhật hàm Neural() để tích hợp
def Neural():
    st.title("🖊️ MNIST Neural Network App")
    tab1, tab2, tab3, tab4 = st.tabs(["📘 Data", "⚙️ Huấn luyện", "🔢 Dự đoán", "🔥 Mlflow"])

    with tab1:
        data()
    with tab2:
        split_data()
        train()
    with tab3:
        du_doan()
    with tab4:
        show_experiment_selector()

if __name__=="__main__":
    Neural()
