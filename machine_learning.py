import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from evalml.automl import AutoMLSearch

def render_ml_page():
    st.title("Machine Learning AutoML")

    # Đọc dữ liệu đã tiền xử lý
    if not (os.path.exists("X_train.csv") and os.path.exists("y_train.csv") and os.path.exists("X_test.csv") and os.path.exists("y_test.csv")):
        st.warning("Vui lòng thực hiện tiền xử lý dữ liệu trước ở tab 'Profiling'.")
        return

    X_train = pd.read_csv("X_train.csv")
    y_train = pd.read_csv("y_train.csv").squeeze()
    X_test = pd.read_csv("X_test.csv")
    y_test = pd.read_csv("y_test.csv").squeeze()

    # Chọn loại bài toán và chế độ
    problem_type = st.selectbox("Chọn loại bài toán", ["binary", "multiclass", "regression"])
    mode = st.radio("Chọn chế độ cấu hình", ["Auto", "Manual"])

    # Cấu hình theo chế độ
    if mode == "Manual":
        if problem_type == "binary":
            objective = st.selectbox("Chọn objective chính", ["Log Loss Binary", "F1", "AUC"])
            objectives = st.multiselect("Chọn các objective phụ", ["Log Loss Binary", "F1", "AUC"], default=[objective])
        elif problem_type == "multiclass":
            objective = st.selectbox("Chọn objective chính", ["Log Loss Multiclass", "F1 Micro", "Accuracy"])
            objectives = st.multiselect("Chọn các objective phụ", ["Log Loss Multiclass", "F1 Micro", "Accuracy"], default=[objective])
        else:  # regression
            objective = st.selectbox("Chọn objective chính", ["R2", "Mean Absolute Error", "Root Mean Squared Error"])
            objectives = st.multiselect("Chọn các objective phụ", ["R2", "Mean Absolute Error", "Root Mean Squared Error"], default=[objective])

        model_families = st.multiselect("Chọn các họ mô hình", ["random_forest", "linear_model", "xgboost", "lightgbm"], default=["random_forest", "xgboost"])
        max_iterations = st.slider("Số vòng lặp tối đa", min_value=1, max_value=50, value=1)
        optimize_thresholds = st.checkbox("Tối ưu ngưỡng (chỉ dùng cho bài toán phân loại)", value=True)
    else:
        # Auto mode
        if problem_type == "binary":
            objective = "Log Loss Binary"
            objectives = ["Log Loss Binary", "F1", "AUC"]
            model_families = ["random_forest", "xgboost", "lightgbm"]
            max_iterations = 10
            optimize_thresholds = True
        elif problem_type == "multiclass":
            objective = "Log Loss Multiclass"
            objectives = ["Log Loss Multiclass", "F1 Micro", "Accuracy"]
            model_families = ["random_forest", "xgboost", "lightgbm"]
            max_iterations = 10
            optimize_thresholds = False
        else:
            objective = "R2"
            objectives = ["R2", "Mean Absolute Error", "Root Mean Squared Error"]
            model_families = ["random_forest", "linear_model", "xgboost"]
            max_iterations = 10
            optimize_thresholds = False

    optimizer_display = st.selectbox("Chiến lược tìm kiếm", [
        "Default",
        "Iterative",
        "Random"
    ])
    optimizer = optimizer_display.lower()

    tuner = st.selectbox("Chiến lược tuner", ["default", "skopt", "random"])

    tuner_class = None
    if tuner == "skopt":
        from evalml.tuners import SKOptTuner
        tuner_class = SKOptTuner
    elif tuner == "random":
        from evalml.tuners import RandomSearchTuner
        tuner_class = RandomSearchTuner

    # Chạy AutoML
    if st.button("Chạy AutoML"):
        automl = AutoMLSearch(
            X_train=X_train,
            y_train=y_train,
            problem_type=problem_type,
            objective=objective,
            additional_objectives=objectives,
            max_iterations=max_iterations,
            optimize_thresholds=optimize_thresholds,
            allowed_model_families=model_families,
            random_seed=42,
            verbose=True,
            automl_algorithm=optimizer,
            tuner_class=tuner_class,
        )

        with st.spinner("Đang chạy AutoML..."):
            automl.search()

        # Hiển thị pipeline tốt nhất
        st.subheader("Pipeline tốt nhất")
        st.write(automl.best_pipeline)

        st.markdown("### Giải thích các chỉ số")
        st.markdown("""
        **score**: điểm số chính dùng để tối ưu (ví dụ: F1, R2, Log Loss...).

        **ranking_score**: giống `score`, dùng để xếp hạng pipeline.

        **validation_score**: điểm số đo trên tập validation trong quá trình cross-validation.

        **Các chỉ số phụ**: là những tiêu chí bổ sung do bạn chọn khi cấu hình AutoML. Các chỉ số này giúp bạn đánh giá mô hình theo nhiều khía cạnh khác nhau. Ví dụ: `accuracy`, `auc`, `log_loss`, `mean_absolute_error`, v.v.

        **Baseline model**: Pipeline có tên *Baseline* là mô hình cực kỳ đơn giản (dự đoán theo số đông hoặc trung bình) được dùng làm mốc để so sánh.
        """)
        # Hiển thị bảng chỉ chứa các chỉ số liên quan
        st.subheader("Chỉ số của các pipeline (bao gồm chỉ số chính và phụ)")

        # Lấy tất cả các chỉ số số học để hiển thị và vẽ, loại bỏ các cột "id" và "search_order"
        numeric_metrics = [col for col in automl.rankings.select_dtypes(include="number").columns if col not in ["id", "search_order"]]
        metric_df = automl.rankings[["pipeline_name"] + numeric_metrics].copy()
        # st.write("Các chỉ số phụ được theo dõi:", objectives)
        st.dataframe(metric_df)

        # Biểu đồ cột so sánh chỉ số
        st.subheader("So sánh chỉ số giữa các mô hình")

        plot_metrics = numeric_metrics
        plot_df = metric_df.copy()

        plot_df_clean = plot_df.dropna(subset=plot_metrics)
        plot_df_clean.set_index("pipeline_name", inplace=True)

        plot_df_numeric = plot_df_clean[plot_metrics].apply(pd.to_numeric, errors="coerce")

        if plot_df_numeric.shape[1] == 0 or plot_df_numeric.dropna(how="all").empty:
            st.warning("Không có metric dạng số để vẽ biểu đồ.")
        else:
            for metric in plot_metrics:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.barh(plot_df_numeric.index, plot_df_numeric[metric])
                ax.set_title(f"Chỉ số: {metric}")
                ax.set_xlabel("Giá trị")
                ax.set_ylabel("Pipeline")
                st.pyplot(fig)

        # Đánh giá mô hình tốt nhất trên tập test
        st.subheader("Đánh giá pipeline tốt nhất trên tập test")

        st.write("Pipeline được chọn:", automl.rankings.iloc[0]["pipeline_name"])
        best_pipeline = automl.best_pipeline
        best_pipeline.fit(X_train, y_train)
        X_test = X_test.astype(X_train.dtypes.to_dict())
        test_score = best_pipeline.score(X_test, y_test, objectives=[objective] + objectives)

        for obj, score in test_score.items():
            st.write(f"{obj}: {score:.4f}")