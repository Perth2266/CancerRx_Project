import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from category_encoders import BinaryEncoder

st.set_page_config(page_title="CancerRx Predictor", layout="wide")

# FILE PATHS
DATA_PATH = "Data_CancerRx_Predictor_CLEAN_DATA"
CLS_PATH  = "Cancer_Classification_Model"
REG_PATH  = "Cancer_Regression_Model"

# LOAD DATA
@st.cache_resource
def load_data():
    cls_df = pd.read_csv(os.path.join(DATA_PATH, "gdsc1_classification.csv"))
    reg_df = pd.read_csv(os.path.join(DATA_PATH, "gdsc1_regression.csv"))
    return cls_df, reg_df

# LAZY LOAD MODELS — only loaded when actually needed
@st.cache_resource
def load_cls_model(name):
    paths = {
        "Random Forest Classification": os.path.join(CLS_PATH, "cancer_RFC_classification_model.pkl"),
        "KNN":                          os.path.join(CLS_PATH, "cls_knn.pkl"),
        "Logistic Regression":          os.path.join(CLS_PATH, "cls_logistic.pkl"),
        "SVM":                          os.path.join(CLS_PATH, "cls_svm.pkl"),
    }
    return joblib.load(paths[name])

@st.cache_resource
def load_reg_model(name):
    paths = {
        "Random Forest Regression": os.path.join(REG_PATH, "cancer_RFC_regression_model.pkl"),
        "KNN":                      os.path.join(REG_PATH, "reg_knn.pkl"),
        "Linear Regression":        os.path.join(REG_PATH, "reg_linear.pkl"),
    }
    return joblib.load(paths[name])

CLS_MODEL_NAMES = ["Random Forest Classification", "KNN", "Logistic Regression", "SVM"]
REG_MODEL_NAMES = ["Random Forest Regression", "KNN", "Linear Regression"]

cls_df, reg_df = load_data()

# NOMINAL COLUMNS
nominal_columns = [
    "TCGA_DESC",
    "GDSC Tissue descriptor 1",
    "GDSC Tissue descriptor 2",
    "Cancer Type (matching TCGA label)",
    "TARGET",
    "TARGET_PATHWAY",
    "Screen Medium",
    "Growth Properties",
    "Microsatellite instability Status (MSI)"
]

# CACHED SPLITS
@st.cache_resource
def get_cls_splits():
    from sklearn.model_selection import train_test_split
    be = BinaryEncoder(cols=nominal_columns)
    X  = be.fit_transform(cls_df[nominal_columns]).to_numpy()
    y  = cls_df["SENSITIVE"].to_numpy()
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, y_test

@st.cache_resource
def get_reg_splits():
    from sklearn.model_selection import train_test_split
    be = BinaryEncoder(cols=nominal_columns)
    X  = be.fit_transform(reg_df[nominal_columns]).to_numpy()
    y  = reg_df["LN_IC50"].to_numpy()
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_test, y_test

# SIDEBAR
st.sidebar.title("CancerRx Predictor")
page = st.sidebar.radio("Navigate", [
    "CancerRx",
    "Data Explorer",
    "Model Comparison",
    "Prediction"
])

# ─────────────────────────────────────────────
# PAGE 1: HOME
# ─────────────────────────────────────────────
if page == "CancerRx":
    st.title("CancerRx Predictor")
    st.write("This app predicts cancer drug sensitivity using machine learning.")

    col_1, col_2 = st.columns(2)

    col_1.subheader("What is IC50?")
    col_1.write("""
        IC50 is the concentration of the drug needed to kill 50% of cancer cells during treatment.

        In simple terms:

        - **More sensitive cells reach IC50 faster (with less drug)**
        - **Resistant cells need more drug to reach the same effect**

        We use **LN_IC50** (a log-transformed version) to make the values more stable and easier for machine learning models to understand.
    """)

    col_2.subheader("Regression vs Classification")
    col_2.write("""
        **Regression** predicts the exact LN_IC50 value

            Lower Predicted LN_IC50  📉 → drug works better
            Higher Predicted LN_IC50 📈 → drug works worse

        **Classification** predicts whether a cancer cell line is:

            Sensitive (1) — drug will likely work
            Resistant (0) — drug will likely not work
    """)

    st.subheader("How Drug Dose Affects Cancer Cells")

    dose      = np.linspace(0, 10, 100)
    sensitive = 1 / (1 + np.exp(-(dose - 3)))
    resistant = 1 / (1 + np.exp(-(dose - 7)))

    fig, ax = plt.subplots()
    ax.plot(dose, sensitive, label="Sensitive Cell (Low IC50)",  color="steelblue")
    ax.plot(dose, resistant, label="Resistant Cell (High IC50)", color="coral")
    ax.set_xlabel("Drug Dose")
    ax.set_ylabel("Cell Death (%)")
    ax.set_title("How Drug Dose Affects Cancer Cells")
    ax.legend()
    st.pyplot(fig)

# ─────────────────────────────────────────────
# PAGE 2: DATA EXPLORER
# ─────────────────────────────────────────────
elif page == "Data Explorer":
    st.title("Data Explorer")

    task = st.radio("Choose Dataset", ["Classification", "Regression"], horizontal=True)
    df   = cls_df if task == "Classification" else reg_df

    st.subheader("Preview")
    st.dataframe(df.head(5))

    col_1, col_2, col_3 = st.columns(3)
    col_1.metric("Rows",                f"{df.shape[0]:,}")
    col_2.metric("Missing Values",      int(df.isnull().sum().sum()))
    col_3.metric("Unique Cancer Types", df["Cancer Type (matching TCGA label)"].nunique())

    st.subheader("Summary Statistics")
    st.write(df.describe())

    st.subheader("Filtering Data By Categorical Column")
    filter_col  = st.selectbox("Filter by column:", nominal_columns)
    unique_vals = df[filter_col].unique()
    chosen_val  = st.selectbox("Choose value from the column:", unique_vals)
    df_filtered = df[df[filter_col] == chosen_val]
    st.write(f"Filtered rows: {len(df_filtered):,}")
    st.dataframe(df_filtered.head(10))

    st.subheader("Data Visualization")
    target_col = "SENSITIVE" if task == "Classification" else "LN_IC50"

    # 1. Target Distribution
    st.markdown("**1. Target Distribution**")
    fig, ax = plt.subplots()
    if task == "Classification":
        df[target_col].value_counts().plot.bar(ax=ax, color=["steelblue", "coral"])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Resistant (0)", "Sensitive (1)"], rotation=0)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_title("Sensitive vs Resistant Cell Lines")
    else:
        sns.histplot(df[target_col], bins=50, ax=ax, color="steelblue")
        ax.set_xlabel("LN_IC50")
        ax.set_ylabel("")
        ax.set_title("Distribution of LN_IC50 Values")
    st.pyplot(fig)

    # 2. Box Plot
    st.markdown("**2. Box Plot**")
    if task == "Classification":
        fig, ax = plt.subplots()
        temp_df = cls_df.copy()
        temp_df["LN_IC50"]   = reg_df["LN_IC50"]
        temp_df["Sensitive"] = temp_df["SENSITIVE"].map({0: "Resistant (0)", 1: "Sensitive (1)"})
        sns.boxplot(
            x="Sensitive", y="LN_IC50", hue="Sensitive",
            data=temp_df, ax=ax,
            palette={"Resistant (0)": "coral", "Sensitive (1)": "steelblue"},
            legend=False
        )
        ax.set_title("LN_IC50 Distribution: Sensitive vs Resistant")
        ax.set_xlabel("")
        ax.set_ylabel("LN_IC50")
    else:
        fig, ax = plt.subplots()
        sns.boxplot(x=df[target_col], ax=ax, color="steelblue")
        ax.set_title("Box Plot: LN_IC50")
        ax.set_xlabel("LN_IC50")
    st.pyplot(fig)

    # 3. Top 10 Cancer Types
    st.markdown("**3. Top 10 Cancer Types**")
    top_cancers = df["Cancer Type (matching TCGA label)"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=top_cancers.index, y=top_cancers.values, ax=ax, color="steelblue")
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_title("Top 10 Cancer Types in Dataset")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

    # 4. Drug Target Pathway Distribution
    st.markdown("**4. Drug Target Pathway Distribution**")
    top_pathways = df["TARGET_PATHWAY"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=top_pathways.index, y=top_pathways.values, ax=ax, color="coral")
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_title("Top 10 Drug Target Pathways")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

    # 5. LN_IC50 by Growth Properties
    st.markdown("**5. LN_IC50 by Growth Properties**")
    if task == "Classification":
        fig, ax = plt.subplots(figsize=(8, 4))
        growth_counts = df.groupby(["Growth Properties", "SENSITIVE"]).size().unstack()
        growth_counts.plot.bar(ax=ax, color=["steelblue", "coral"])
        ax.set_title("Sensitive vs Resistant by Growth Properties")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.legend(["Resistant", "Sensitive"])
        plt.xticks(rotation=0)
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(
            x="Growth Properties", y="LN_IC50",
            hue="Growth Properties", data=df, ax=ax,
            palette="Set2", legend=False
        )
        ax.set_title("LN_IC50 Distribution by Growth Properties")
        ax.set_xlabel("")
        ax.set_ylabel("")
    st.pyplot(fig)

    # 6. Custom exploration
    st.markdown("**6. Explore Any Categorical Column vs Target**")
    chosen_cat = st.selectbox("Pick a column to explore", nominal_columns, key="viz7")
    top_vals   = df[chosen_cat].value_counts().head(8).index
    df_top     = df[df[chosen_cat].isin(top_vals)]

    if task == "Classification":
        fig, ax = plt.subplots(figsize=(10, 4))
        counts = df_top.groupby([chosen_cat, "SENSITIVE"]).size().unstack()
        counts.plot.bar(ax=ax, color=["steelblue", "coral"])
        ax.set_title(f"Sensitive vs Resistant by {chosen_cat}")
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        ax.legend(["Resistant", "Sensitive"])
        plt.xticks(rotation=45, ha="right")
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.boxplot(
            x=chosen_cat, y="LN_IC50",
            hue=chosen_cat, data=df_top, ax=ax,
            palette="Set2", legend=False
        )
        ax.set_title(f"LN_IC50 by {chosen_cat}")
        ax.set_xlabel("")
        plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

# ─────────────────────────────────────────────
# PAGE 3: MODEL COMPARISON
# ─────────────────────────────────────────────
elif page == "Model Comparison":
    st.title("Model Comparison")

    task = st.radio("Task", ["Classification", "Regression"], horizontal=True)

    if task == "Classification":
        from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

        X_test, y_test = get_cls_splits()
        results = []

        for name in CLS_MODEL_NAMES:
            with st.spinner(f"Evaluating {name}..."):
                model  = load_cls_model(name)
                y_pred = model.predict(X_test)
                acc    = round(accuracy_score(y_test, y_pred), 4)
                recall = round(recall_score(y_test, y_pred), 4)
                try:
                    auc = round(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]), 4)
                except:
                    auc = "N/A"
                results.append({"Model": name, "Accuracy": acc, "Recall": recall, "ROC AUC": auc})

        results_df = pd.DataFrame(results).sort_values("Recall", ascending=False)
        st.dataframe(results_df)
        best = results_df.iloc[0]["Model"]
        st.success(f"Best Model: {best} (highest Recall)")

        fig, ax = plt.subplots()
        ax.bar(results_df["Model"], results_df["Recall"], color="steelblue")
        ax.set_ylabel("Recall")
        ax.set_title("Model Comparison: Recall")
        plt.xticks(rotation=15)
        st.pyplot(fig)

    else:
        from sklearn.metrics import mean_squared_error, r2_score

        X_test, y_test = get_reg_splits()
        results = []

        for name in REG_MODEL_NAMES:
            with st.spinner(f"Evaluating {name}..."):
                model  = load_reg_model(name)
                y_pred = model.predict(X_test)
                mse    = round(mean_squared_error(y_test, y_pred), 4)
                r2     = round(r2_score(y_test, y_pred), 4)
                results.append({"Model": name, "MSE": mse, "R²": r2})

        results_df = pd.DataFrame(results).sort_values("R²", ascending=False)
        st.dataframe(results_df)
        best = results_df.iloc[0]["Model"]
        st.success(f"Best Model: {best} (highest R²)")

        fig, ax = plt.subplots()
        ax.bar(results_df["Model"], results_df["R²"], color="steelblue")
        ax.set_ylabel("R²")
        ax.set_title("Model Comparison: R²")
        plt.xticks(rotation=15)
        st.pyplot(fig)

# ─────────────────────────────────────────────
# PAGE 4: PREDICTION
# ─────────────────────────────────────────────
elif page == "Prediction":
    st.title("Prediction")

    task       = st.radio("Task", ["Classification", "Regression"], horizontal=True)
    model_names = CLS_MODEL_NAMES if task == "Classification" else REG_MODEL_NAMES
    model_name  = st.selectbox("Choose Model", model_names)

    st.subheader("Enter Input Features")

    col_1, col_2, col_3 = st.columns(3)
    df = cls_df if task == "Classification" else reg_df

    tcga_desc   = col_1.selectbox("TCGA_DESC",                               df["TCGA_DESC"].unique())
    tissue_1    = col_2.selectbox("GDSC Tissue descriptor 1",                df["GDSC Tissue descriptor 1"].unique())
    tissue_2    = col_3.selectbox("GDSC Tissue descriptor 2",                df["GDSC Tissue descriptor 2"].unique())
    cancer_type = col_1.selectbox("Cancer Type (matching TCGA label)",       df["Cancer Type (matching TCGA label)"].unique())
    target      = col_2.selectbox("TARGET",                                  df["TARGET"].unique())
    pathway     = col_3.selectbox("TARGET_PATHWAY",                          df["TARGET_PATHWAY"].unique())
    screen_med  = col_1.selectbox("Screen Medium",                           df["Screen Medium"].unique())
    growth      = col_2.selectbox("Growth Properties",                       df["Growth Properties"].unique())
    msi         = col_3.selectbox("Microsatellite instability Status (MSI)", df["Microsatellite instability Status (MSI)"].unique())

    with st.expander("See your inputs"):
        st.write({
            "TCGA_DESC":                               tcga_desc,
            "GDSC Tissue descriptor 1":                tissue_1,
            "GDSC Tissue descriptor 2":                tissue_2,
            "Cancer Type (matching TCGA label)":       cancer_type,
            "TARGET":                                  target,
            "TARGET_PATHWAY":                          pathway,
            "Screen Medium":                           screen_med,
            "Growth Properties":                       growth,
            "Microsatellite instability Status (MSI)": msi
        })

    clicked = st.button("Predict")

    if clicked:
        with st.spinner(f"Loading {model_name} and predicting..."):
            if task == "Classification":
                model = load_cls_model(model_name)
            else:
                model = load_reg_model(model_name)

            user_df = pd.DataFrame([{
                "TCGA_DESC":                                tcga_desc,
                "GDSC Tissue descriptor 1":                 tissue_1,
                "GDSC Tissue descriptor 2":                 tissue_2,
                "Cancer Type (matching TCGA label)":        cancer_type,
                "TARGET":                                   target,
                "TARGET_PATHWAY":                           pathway,
                "Screen Medium":                            screen_med,
                "Growth Properties":                        growth,
                "Microsatellite instability Status (MSI)":  msi
            }])

            be = BinaryEncoder(cols=nominal_columns)
            be.fit(df[nominal_columns])
            user_encoded = be.transform(user_df).to_numpy()
            prediction   = model.predict(user_encoded)

        st.divider()

        if task == "Classification":
            if prediction[0] == 1:
                st.success("Prediction: ✅ SENSITIVE — This cell line is likely to respond to the drug.")
            else:
                st.error("Prediction: ❌ RESISTANT — This cell line is unlikely to respond to the drug.")

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(user_encoded)[0]
                col_a, col_b = st.columns(2)
                col_a.metric("Sensitive Probability", f"{round(proba[1] * 100, 1)}%")
                col_b.metric("Resistant Probability", f"{round(proba[0] * 100, 1)}%")

        else:
            predicted_value = round(float(prediction[0]), 4)
            if predicted_value < 1.50:
                st.success(f"Predicted LN_IC50: {predicted_value} — Low value, cell is likely SENSITIVE to the drug.")
            elif predicted_value < 4.70:
                st.warning(f"Predicted LN_IC50: {predicted_value} — Moderate value, response is uncertain.")
            else:
                st.error(f"Predicted LN_IC50: {predicted_value} — High value, cell is likely RESISTANT to the drug.")