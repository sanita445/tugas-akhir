
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

st.title("Aplikasi Data Mining Berbasis Web")
st.write("Upload dataset CSV, pilih fitur & target, lalu latih model Decision Tree.")

# Upload file CSV
uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview Dataset")
    st.dataframe(df.head())

    # Pilih fitur dan target
    columns = df.columns.tolist()
    fitur = st.multiselect("Pilih fitur (X)", columns)
    target = st.selectbox("Pilih target (y)", columns)

    if st.button("Latih Model"):
        if fitur and target:
            X = df[fitur]
            y = df[target]

            # Label encoding untuk data kategorikal
            for col in X.select_dtypes(include=['object']).columns:
                X[col] = LabelEncoder().fit_transform(X[col])
            if y.dtype == 'object':
                y = LabelEncoder().fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = DecisionTreeClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.subheader("Hasil Evaluasi")
            st.write("Akurasi:", accuracy_score(y_test, y_pred))
            st.text("Laporan Klasifikasi:\n" + classification_report(y_test, y_pred))
        else:
            st.warning("Pilih minimal satu fitur dan target.")
else:
    st.info("Silakan unggah file CSV terlebih dahulu.")
