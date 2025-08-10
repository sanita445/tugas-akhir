
\"\"\"
Aplikasi Data Mining berbasis Web menggunakan Streamlit
File: streamlit_data_mining_app.py
Instruksi singkat:
1. Buat virtual environment (opsional): python -m venv venv
2. Aktifkan venv: 
   - Windows: venv\\Scripts\\activate
   - macOS/Linux: source venv/bin/activate
3. Install requirement: pip install -r requirements.txt
4. Jalankan: streamlit run streamlit_data_mining_app.py

Fitur utama:
- Upload CSV
- Pratinjau data dan statistik
- Preprocessing dasar: dropna, imputasi, encoding, scaling
- Pilih target, tentukan masalah (classification/regression)
- Model: Decision Tree, Random Forest, KNN, LogisticRegression (classification), LinearRegression (regression), Naive Bayes
- Evaluasi: confusion matrix, classification report, R2, MAE, MSE
- Download prediksi dan model (.pkl)

Catatan: ini adalah template edukasi. Silakan minta fitur tambahan (cross-val, hyperparam tuning, visual interaktif, dsb.).
\"\"\"

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# models
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB

# metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title=\"Aplikasi Data Mining (Streamlit)\", layout='wide')

# --- Helper functions ---

def download_bytes(obj, filename):
    bio = BytesIO()
    joblib.dump(obj, bio)
    bio.seek(0)
    return bio

def to_csv_bytes(df: pd.DataFrame):
    return df.to_csv(index=False).encode('utf-8')

def detect_problem_type(y: pd.Series):
    # heuristik sederhana: jika y numeric dengan banyak unique values -> regression
    if pd.api.types.is_numeric_dtype(y):
        if y.nunique() <= 20:
            return 'classification'
        else:
            return 'regression'
    else:
        return 'classification'


# --- UI ---
st.title('📊 Aplikasi Data Mining (Streamlit)')
st.write('Template sederhana untuk eksplorasi data, preprocessing, training model, dan evaluasi.')

with st.sidebar:
    st.header('1. Upload & Pengaturan')
    uploaded_file = st.file_uploader('Unggah file CSV', type=['csv'])
    sample_button = st.button('Gunakan dataset contoh (Iris)')

    st.markdown('---')
    st.header('2. Pilihan Model & Hyperparameter')
    model_type = st.selectbox('Pilih model', ['Decision Tree', 'Random Forest', 'K-Nearest Neighbors', 'Logistic Regression / Linear Regression', 'Naive Bayes (classification only)'])

    # hyperparameters sederhana
    if model_type == 'Decision Tree':
        max_depth = st.number_input('max_depth (None=0)', min_value=0, step=1, value=0)
    elif model_type == 'Random Forest':
        n_estimators = st.number_input('n_estimators', min_value=1, step=1, value=100)
        rf_max_depth = st.number_input('max_depth (None=0)', min_value=0, step=1, value=0)
    elif model_type == 'K-Nearest Neighbors':
        n_neighbors = st.number_input('n_neighbors', min_value=1, step=1, value=5)
    elif model_type == 'Logistic Regression / Linear Regression':
        lr_C = st.number_input('C (regularization inverse, for Logistic)', min_value=0.0001, value=1.0, format=\"%.4f\")
    elif model_type == 'Naive Bayes (classification only)':
        pass

    st.markdown('---')
    st.header('3. Preprocessing')
    drop_na = st.checkbox('Drop rows with NA', value=False)
    impute_strategy = st.selectbox('Impute missing (if not drop)', ['mean', 'median', 'most_frequent', 'constant'])
    encode_categorical = st.checkbox('One-Hot Encode categorical features', value=True)
    scale_features = st.selectbox('Scale numeric features', ['None', 'StandardScaler', 'MinMaxScaler'])

    st.markdown('---')
    st.header('4. Train/Test')
    test_size = st.slider('Test set proportion', min_value=0.05, max_value=0.5, value=0.2, step=0.05)
    random_state = st.number_input('Random state', value=42)

    st.markdown('---')
    st.header('5. Actions')
    train_button = st.button('Train Model')

# Load example dataset if requested
if sample_button:
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    df = iris.frame
    df['target'] = df['target']
else:
    df = None

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f'Gagal membaca CSV: {e}')

if df is None:
    st.info('Unggah file CSV di sidebar atau klik \"Gunakan dataset contoh\" untuk mencoba.')
    st.stop()

st.subheader('Preview Data')
st.dataframe(df.head(100))

st.subheader('Ringkasan Statistik')
st.write(df.describe(include='all'))

# Select target
all_columns = df.columns.tolist()

with st.expander('Pilih target dan fitur'):
    target_col = st.selectbox('Pilih kolom target (label)', options=all_columns)
    feature_cols = st.multiselect('Pilih fitur (kosong = gunakan semua kecuali target)', options=[c for c in all_columns if c != target_col])

if not feature_cols:
    feature_cols = [c for c in all_columns if c != target_col]

X = df[feature_cols]
y = df[target_col]

problem_type = detect_problem_type(y)
st.info(f'Dideteksi tipe masalah: **{problem_type}** (heuristik). Anda bisa memilih model yang sesuai di sidebar.')

# Preprocessing
X_proc = X.copy()
y_proc = y.copy()

if drop_na:
    X_proc = X_proc.dropna()
    y_proc = y_proc.loc[X_proc.index]
else:
    # simple imputation
    num_cols = X_proc.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_proc.select_dtypes(exclude=[np.number]).columns.tolist()

    if len(num_cols) > 0:
        imp_num = SimpleImputer(strategy=impute_strategy if impute_strategy in ['mean', 'median'] else 'mean')
        X_proc[num_cols] = imp_num.fit_transform(X_proc[num_cols])

    if len(cat_cols) > 0:
        if impute_strategy == 'most_frequent':
            imp_cat = SimpleImputer(strategy='most_frequent')
        else:
            imp_cat = SimpleImputer(strategy='constant', fill_value='missing')
        X_proc[cat_cols] = imp_cat.fit_transform(X_proc[cat_cols])

# Encoding
numeric_features = X_proc.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X_proc.select_dtypes(exclude=[np.number]).columns.tolist()

transformers = []
if len(numeric_features) > 0 and scale_features != 'None':
    if scale_features == 'StandardScaler':
        transformers.append(('num', StandardScaler(), numeric_features))
    else:
        transformers.append(('num', MinMaxScaler(), numeric_features))

if encode_categorical and len(cat_features) > 0:
    transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_features))

if transformers:
    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
    X_ready = pd.DataFrame(preprocessor.fit_transform(X_proc))
    # try to recover column names if OneHot used
    try:
        out_cols = []
        for name, trans, cols in preprocessor.transformers_:
            if name == 'cat':
                # OneHotEncoder
                ohe = trans
                categories = ohe.categories_
                for ci, c in enumerate(cols):
                    cats = categories[ci]
                    out_cols.extend([f\"{c}__{str(cat)}\" for cat in cats])
            elif name == 'num':
                out_cols.extend(cols)
        if preprocessor.remainder == 'passthrough':
            passthrough = [c for c in X_proc.columns if c not in numeric_features + cat_features]
            out_cols.extend(passthrough)
        if len(out_cols) == X_ready.shape[1]:
            X_ready.columns = out_cols
    except Exception:
        pass
else:
    X_ready = X_proc.copy()

st.subheader('Data setelah preprocessing (preview)')
st.dataframe(X_ready.head(100))

# Train/test split
try:
    X_train, X_test, y_train, y_test = train_test_split(X_ready, y_proc, test_size=test_size, random_state=int(random_state))
except Exception as e:
    st.error(f'Error saat melakukan train_test_split: {e}')
    st.stop()

st.write(f'Train shape: {X_train.shape}, Test shape: {X_test.shape}')

# Model selection
model = None
if model_type == 'Decision Tree':
    md = None if max_depth == 0 else int(max_depth)
    if problem_type == 'classification':
        model = DecisionTreeClassifier(max_depth=md, random_state=int(random_state))
    else:
        model = DecisionTreeRegressor(max_depth=md, random_state=int(random_state))

elif model_type == 'Random Forest':
    md = None if rf_max_depth == 0 else int(rf_max_depth)
    if problem_type == 'classification':
        model = RandomForestClassifier(n_estimators=int(n_estimators), max_depth=md, random_state=int(random_state))
    else:
        model = RandomForestRegressor(n_estimators=int(n_estimators), max_depth=md, random_state=int(random_state))

elif model_type == 'K-Nearest Neighbors':
    if problem_type == 'classification':
        model = KNeighborsClassifier(n_neighbors=int(n_neighbors))
    else:
        model = KNeighborsRegressor(n_neighbors=int(n_neighbors))

elif model_type == 'Logistic Regression / Linear Regression':
    if problem_type == 'classification':
        model = LogisticRegression(C=float(lr_C), max_iter=1000)
    else:
        model = LinearRegression()

elif model_type == 'Naive Bayes (classification only)':
    if problem_type == 'classification':
        model = GaussianNB()
    else:
        st.warning('Naive Bayes hanya tersedia untuk classification.')
        model = None

if model is None:
    st.warning('Pilih model yang sesuai.')

# Train
if train_button and model is not None:
    with st.spinner('Training model...'):
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            st.error(f'Gagal saat training: {e}')
            st.stop()

    st.success('Training selesai ✅')

    # Predict
    y_pred = model.predict(X_test)

    if problem_type == 'classification':
        acc = accuracy_score(y_test, y_pred)
        st.metric('Accuracy', f'{acc:.4f}')
        st.write('Classification Report:')
        st.text(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        st.write('Confusion Matrix:')
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

        # feature importance if available
        if hasattr(model, 'feature_importances_'):
            fi = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(30)
            st.write('Feature Importances (top 30):')
            st.bar_chart(fi)

    else:
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        st.metric('R2', f'{r2:.4f}')
        st.metric('MAE', f'{mae:.4f}')
        st.metric('MSE', f'{mse:.4f}')

        # scatter actual vs pred
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
        st.pyplot(fig)

    # save model btn
    model_bytes = download_bytes(model, 'model.pkl')
    st.download_button('Download model (.pkl)', data=model_bytes, file_name='model.pkl', mime='application/octet-stream')

    # Save predictions
    out_df = X_test.copy()
    out_df['actual'] = y_test.values
    out_df['predicted'] = y_pred
    csv_bytes = to_csv_bytes(out_df)
    st.download_button('Download predictions (CSV)', data=csv_bytes, file_name='predictions.csv', mime='text/csv')

    st.info('Selesai. Anda bisa menyesuaikan preprocessing / model dan latih lagi.')

else:
    st.info('Tekan tombol \"Train Model\" di sidebar untuk mulai melatih model dengan pengaturan saat ini.')

# Footer
st.markdown('---')
st.write('Butuh fitur tambahan? Contoh: cross-validation, hyperparameter tuning, model comparison, pipeline saving, visualisasi interaktif. Saya bantu kembangkan.')
