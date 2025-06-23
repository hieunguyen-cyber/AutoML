import pandas as pd
import os
import sweetviz as sv
import streamlit as st
import webbrowser

from sklearn.ensemble import RandomForestRegressor
from imblearn.over_sampling import ADASYN

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
from sklearn.decomposition import PCA

def perform_preprocessing(df, target, apply_pca=False, n_components=None):
    st.info(f"Tiền xử lý dữ liệu với cột mục tiêu: {target}")

    X = df.drop(columns=[target])
    y = df[target]

    # Gọi pipeline xử lý cho toàn bộ dữ liệu trước khi chia
    X_processed, y_processed, pca_model, feature_columns = preprocess_pipeline(X, y, apply_pca=apply_pca, n_components=n_components)

    # Chia tập train-test sau khi tiền xử lý
    X_train_processed, X_test, y_train_processed, y_test = train_test_split(X_processed, y_processed, test_size=0.2, random_state=42)

    # Lưu cả 4 file
    X_train_processed.to_csv("X_train.csv", index=False)
    y_train_processed.to_csv("y_train.csv", index=False)
    X_test.to_csv("X_test.csv", index=False)
    y_test.to_csv("y_test.csv", index=False)

    st.success("✅ Tiền xử lý và chia tập hoàn tất. Đã lưu ra 4 file CSV.")
    st.write("X_train:")
    st.dataframe(X_train_processed.head())

def preprocess_pipeline(X, y, apply_pca=False, n_components=None):
    # --- Handle Missing Values ---
    for col in X.columns:
        missing_ratio = X[col].isna().mean()

        if missing_ratio == 0:
            continue
        elif missing_ratio < 0.01:
            valid_idx = ~X[col].isna()
            X = X.loc[valid_idx]
            y = y.loc[valid_idx]
        elif missing_ratio < 0.05:
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna("Unknown")
        elif missing_ratio < 0.5:
            if pd.api.types.is_numeric_dtype(X[col]):
                not_null = X[X[col].notnull()]
                null = X[X[col].isnull()]

                if not_null.shape[0] > 0:
                    model = RandomForestRegressor()
                    temp_X = pd.get_dummies(not_null.drop(columns=[col]), drop_first=True)
                    model.fit(temp_X, not_null[col])

                    null_X = pd.get_dummies(null.drop(columns=[col]), drop_first=True)
                    null_X = null_X.reindex(columns=temp_X.columns, fill_value=0)

                    X.loc[X[col].isnull(), col] = model.predict(null_X)
                else:
                    X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna("Unknown")
        else:
            X = X.drop(columns=[col])

    y = y.loc[X.index]

    # --- Handle Outliers ---
    for col in X.select_dtypes(include=["int64", "float64"]).columns:
        if X[col].nunique() > 10:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            X[col] = X[col].clip(lower, upper)
        else:
            freqs = X[col].value_counts(normalize=True)
            rare_vals = freqs[freqs < 0.03].index
            max_val = X[col].max()
            X[col] = X[col].apply(lambda x: max_val + 1 if x in rare_vals else x)

    # --- Handle Categorical ---
    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    cols_to_drop = []
    for col in cat_cols:
        freqs = X[col].value_counts(normalize=True)
        rare_vals = freqs[freqs < 0.05].index
        X[col] = X[col].apply(lambda x: "Unknown" if x in rare_vals else x)

        if X[col].nunique() > 100:
            cols_to_drop.append(col)

    X.drop(columns=cols_to_drop, inplace=True)
    X = pd.get_dummies(X, drop_first=True)

    # --- Balance Data (ADASYN) ---
    pos_ratio = y.value_counts(normalize=True).min()
    if pos_ratio < 0.3:
        ada = ADASYN(random_state=42)
        X, y = ada.fit_resample(X, y)

    pca_model = None
    if apply_pca and n_components is not None:
        pca_model = PCA(n_components=n_components)
        X = pca_model.fit_transform(X)
        X = pd.DataFrame(X, columns=[f"PCA_{i+1}" for i in range(n_components)])

    return X, y, pca_model, X.columns

def render_profiling_page():
    st.title("Exploratory Data Analysis (EDA)")

    data_path = "source_data.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        st.subheader("Chọn cột mục tiêu (target)")
        target_column = st.selectbox("Target column", df.columns)

        apply_pca = st.checkbox("Áp dụng PCA sau tiền xử lý")
        n_components = None
        if apply_pca:
            max_components = min(len(df.columns) - 1, 50)
            n_components = st.slider("Số thành phần PCA", min_value=1, max_value=max_components, value=min(10, max_components))

        if st.button("Thực hiện tiền xử lý"):
            perform_preprocessing(df, target_column, apply_pca=apply_pca, n_components=n_components)

        if st.button("Mở báo cáo Sweetviz"):
            report = sv.analyze(df)
            report.show_html("sweetviz_report.html")
            webbrowser.open("sweetviz_report.html")
    else:
        st.warning("Không tìm thấy dữ liệu! Vui lòng upload dữ liệu trước ở tab 'Upload'.")
