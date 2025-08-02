import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import os

# ----------------------------
# 📦 Load required components
# ----------------------------
nltk.download("punkt")
nltk.download("wordnet")

df_base = pd.read_csv("data/processed/task_management_nlp_preprocessed.csv")
priority_model = joblib.load("results/models/priority_predictor_xgb_fusion.joblib")
priority_encoder = joblib.load("results/models/priority_label_encoder.joblib")
classifier_model = joblib.load("results/models/task_classifier_xgb_fusion.joblib")
vectorizer = joblib.load("results/models/tfidf_vectorizer.joblib")

# ----------------------------
# 🧹 Text cleaning function
# ----------------------------
lemmatizer = WordNetLemmatizer()

def clean(text):
    tokens = word_tokenize(str(text).lower())
    tokens = [t for t in tokens if t.isalpha()]
    return " ".join(lemmatizer.lemmatize(t) for t in tokens)

# ----------------------------
# 🚀 Streamlit UI starts here
# ----------------------------
st.sidebar.title("📋 Menu")
section = st.sidebar.radio("Go to", ["🔍 Task Predictor", "🧠 Task Classifier", "📦 Workload Balancer"])

# ----------------------------
# 🔍 TASK PREDICTOR TAB
# ----------------------------
if section == "🔍 Task Predictor":
    st.title("🔍 Task Priority Predictor")

    with st.form("predict_form"):
        title = st.text_input("Task Title")
        description = st.text_area("Task Description")
        complexity = st.slider("Complexity Score", 1.0, 10.0, 5.0)
        estimated_hours = st.slider("Estimated Hours", 1.0, 15.0, 5.0)
        experience = st.slider("User Experience Level", 1.0, 10.0, 5.0)
        workload = st.slider("User Current Workload", 1, 12, 5)
        department = st.selectbox("Department", df_base["department"].unique())
        submitted = st.form_submit_button("Predict")

    if submitted:
        desc_clean = clean(description)
        X_text = vectorizer.transform([desc_clean]).toarray()

        structured = np.array([[complexity, estimated_hours, experience, workload]])
        scaler = StandardScaler()
        scaler.fit(df_base[["complexity_score", "estimated_hours", "user_experience_level", "user_current_workload"]])
        X_struct = scaler.transform(structured)

        dept_ohe = pd.get_dummies(df_base["department"])
        dept_vec = np.zeros((1, dept_ohe.shape[1]))
        if department in dept_ohe.columns:
            dept_idx = dept_ohe.columns.get_loc(department)
            dept_vec[0, dept_idx] = 1

        X_final = np.hstack([X_text, X_struct, dept_vec])
        pred = priority_model.predict(X_final)
        pred_label = priority_encoder.inverse_transform(pred)[0]

        st.success(f"📌 Predicted Priority: **{pred_label}**")

# ----------------------------
# 🧠 TASK CLASSIFIER TAB
# ----------------------------
elif section == "🧠 Task Classifier":
    st.title("🧠 Task Type Classifier")

    with st.form("classify_form"):
        desc = st.text_area("Enter task description:")
        submitted = st.form_submit_button("Classify Task")

    if submitted:
        desc_clean = clean(desc)
        X_text = vectorizer.transform([desc_clean]).toarray()

        # Use default structured values
        structured = np.array([[5.0, 5.0, 5.0, 5.0]])
        scaler = StandardScaler()
        scaler.fit(df_base[["complexity_score", "estimated_hours", "user_experience_level", "user_current_workload"]])
        X_struct = scaler.transform(structured)

        dept_ohe = pd.get_dummies(df_base["department"])
        department = df_base["department"].unique()[0]  # default
        dept_vec = np.zeros((1, dept_ohe.shape[1]))
        if department in dept_ohe.columns:
            dept_idx = dept_ohe.columns.get_loc(department)
            dept_vec[0, dept_idx] = 1

        X_final = np.hstack([X_text, X_struct, dept_vec])
        pred_class = classifier_model.predict(X_final)[0]

        st.success(f"🧠 Predicted Task Category: **{pred_class}**")

# ----------------------------
# 📦 WORKLOAD BALANCER TAB
# ----------------------------
elif section == "📦 Workload Balancer":
    st.title("📦 Workload Rebalancer")
    uploaded_file = st.file_uploader("📁 Upload task CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        users = df[["assigned_to", "user_current_workload", "user_experience_level"]].drop_duplicates()
        users.set_index("assigned_to", inplace=True)

        users["norm_workload"] = (users["user_current_workload"] - users["user_current_workload"].min()) / \
                                 (users["user_current_workload"].max() - users["user_current_workload"].min())
        users["norm_experience"] = (users["user_experience_level"] - users["user_experience_level"].min()) / \
                                   (users["user_experience_level"].max() - users["user_experience_level"].min())

        weight_w = 0.6
        weight_e = 0.4
        df["new_assigned_to"] = ""

        for idx, row in df.iterrows():
            scores = users["norm_workload"] * weight_w - users["norm_experience"] * weight_e
            best_user = scores.idxmin()
            df.at[idx, "new_assigned_to"] = best_user
            users.at[best_user, "user_current_workload"] += row["estimated_hours"]
            w = users["user_current_workload"]
            users["norm_workload"] = (w - w.min()) / (w.max() - w.min())

        st.success("✅ Rebalanced task assignments.")
        st.dataframe(df[["task_id", "title", "priority", "new_assigned_to"]])
        st.download_button("⬇️ Download CSV", df.to_csv(index=False), file_name="balanced_tasks.csv")