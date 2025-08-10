
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Aplikasi Data Mining", layout="wide")
st.title("📊 Aplikasi Data Mining Berbasis Web - Streamlit")

st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Pilih file CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Dataset")
    st.dataframe(df)

    target_column = st.sidebar.selectbox("Pilih kolom target", df.columns)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.sidebar.header("Pilih Algoritma")
    algoritma = st.sidebar.selectbox("Metode", ["Decision Tree", "Random Forest", "KNN", "Logistic Regression", "Linear Regression", "Naive Bayes"])

    if st.sidebar.button("🔍 Jalankan Model"):
        model = None
        if algoritma == "Decision Tree":
            model = DecisionTreeClassifier()
        elif algoritma == "Random Forest":
            model = RandomForestClassifier()
        elif algoritma == "KNN":
            model = KNeighborsClassifier()
        elif algoritma == "Logistic Regression":
            model = LogisticRegression(max_iter=200)
        elif algoritma == "Linear Regression":
            model = LinearRegression()
        elif algoritma == "Naive Bayes":
            model = GaussianNB()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader("📊 Hasil Evaluasi")
        if algoritma == "Linear Regression":
            mse = mean_squared_error(y_test, y_pred)
            st.write(f"Mean Squared Error: {mse}")
        else:
            acc = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy: {acc}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))
            st.write("Confusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

else:
    st.info("Silakan upload file CSV untuk memulai.")
