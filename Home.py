import streamlit as st
import pandas as pd
import joblib
import time
import shap
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
from style import load_css, add_footer  

st.set_page_config(page_title="Students Performance Predictor System",
                   page_icon="🎓",
                   layout="wide",
                   initial_sidebar_state="auto")

def to_dense(x):
    try:
        return x.toarray()
    except Exception:
        return np.asarray(x)

def df_from_transformed(arr, feature_names):
    dense = to_dense(arr)
    return pd.DataFrame(dense, columns=feature_names)

@st.cache_data(ttl=3600)
def load_data(path="study_performance.csv"):
    df = pd.read_csv(path)
    return df

@st.cache_resource
def load_pipeline(path="student_performance_pipeline.joblib", data_path="study_performance.csv"):
    res = {
        'pipe': None,
        'explainer': None,
        'feature_names': [],
        'classes': None,
        'meta': {}
    }
    try:
        pipe = joblib.load(path)
    except FileNotFoundError:
        st.error("Model file not found: 'student_performance_pipeline.joblib'. Make sure it's in the app folder.")
        return res
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return res

    try:
        df = load_data(data_path)
    except FileNotFoundError:
        df = None
    
    try:
        preprocessor = pipe.named_steps.get('preprocessor')
        clf = pipe.named_steps.get('clf') or pipe.named_steps.get('classifier')
    except Exception:
        preprocessor, clf = None, None

    feature_names = []
    if preprocessor is not None and df is not None:
        try:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c not in ['avg_score', 'performance_level']] 
            categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
            
            ohe = preprocessor.named_transformers_['cat']
            
            try:
                ohe_feature_names = ohe.get_feature_names_out(categorical_cols).tolist()
            except AttributeError:
                ohe_feature_names = []
                for i, col in enumerate(categorical_cols):
                    cats = ohe.categories_[i]
                    ohe_feature_names.extend([f"{col}_{c}" for c in cats])
            
            feature_names = numeric_cols + ohe_feature_names
        except Exception as e:
            feature_names = df.columns.tolist()
            
    elif df is not None:
        feature_names = df.columns.tolist()
        
    explainer = None
    if clf is not None and df is not None and preprocessor is not None:
        try:
            X = df.drop(columns=['avg_score', 'performance_level'], errors='ignore')
            X_trans = preprocessor.transform(X)
            X_bg = shap.sample(X_trans, 100)
            X_bg_dense = to_dense(X_bg)

            if len(feature_names) != X_bg_dense.shape[1]:
                feature_names = [f"f{i}" for i in range(X_bg_dense.shape[1])]
            
            explainer = shap.TreeExplainer(clf, X_bg_dense)
            explainer.data = pd.DataFrame(X_bg_dense, columns=feature_names)
        except Exception:
            try:
                X = df.drop(columns=['avg_score', 'performance_level'], errors='ignore')
                X_trans = preprocessor.transform(X)
                explainer = shap.KernelExplainer(clf.predict_proba, to_dense(shap.sample(X_trans, 50)))
            except Exception:
                explainer = None

    res['pipe'] = pipe
    res['explainer'] = explainer
    res['feature_names'] = feature_names
    try:
        res['classes'] = pipe.classes_
    except Exception:
        res['classes'] = None
    res['meta'] = {
        'model_type': type(clf).__name__ if clf is not None else 'Unknown',
        'dataset_rows': len(df) if df is not None else 0,
        'feature_count': len(feature_names) if feature_names else 0
    }
    return res

st.sidebar.markdown("## 🎨 Display Settings")
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True
st.sidebar.toggle("Enable Dark Mode", key='dark_mode')

load_css(st.session_state.dark_mode)

with st.spinner("Loading model and data..."):
    model_bundle = load_pipeline()
pipe = model_bundle['pipe']
explainer = model_bundle['explainer']
feature_names = model_bundle['feature_names']
classes = model_bundle['classes']

st.title("Students Performance Predictor System")
st.markdown("<p style='text-align: center; color: var(--muted);'>\"Unlocking Potential, One Prediction at a Time\"</p>", unsafe_allow_html=True)

st.markdown("---")
left, right = st.columns([1, 2])
with left:
    st.markdown("### 👤 Student Profile")
    gender = st.selectbox("Gender", ['female', 'male'], index=0, help='Student gender')
    race = st.selectbox("Race/Ethnicity", ['group A', 'group B', 'group C', 'group D', 'group E'], index=2)
    parent_edu = st.selectbox("Parental Education", ["associate's degree", "bachelor's degree", 'high school', "master's degree", 'some college', 'some high school'], index=1)
    lunch = st.selectbox("Lunch", ['free/reduced', 'standard'], index=1)
    test_prep = st.selectbox("Test Preparation", ['completed', 'none'], index=1)
    st.markdown("---")
    st.markdown("### Academic Scores")
    math_score = st.slider("Math Score", 0, 100, 50)
    reading_score = st.slider("Reading Score", 0, 100, 50)
    writing_score = st.slider("Writing Score", 0, 100, 50)
    st.markdown("---")
    col1_btn, col2_btn = st.columns(2)
    with col1_btn:
        if st.button("Get Prediction", type="primary", use_container_width=True):
            st.session_state.input_data = {
                'gender': gender, 'race_ethnicity': race, 'parental_level_of_education': parent_edu,
                'lunch': lunch, 'test_preparation_course': test_prep, 'math_score': math_score,
                'reading_score': reading_score, 'writing_score': writing_score
            }
            st.session_state.prediction_made = True
    with col2_btn:
        if st.button("Reset Inputs", use_container_width=True):
            if 'prediction_made' in st.session_state: del st.session_state.prediction_made
            if 'input_data' in st.session_state: del st.session_state.input_data
            st.experimental_rerun()

with right:
    st.markdown("### ✨ Prediction & Explanation")
    if st.session_state.get('prediction_made', False):
        input_data = st.session_state.input_data
        input_df = pd.DataFrame([input_data])
        if pipe:
            try:
                preproc = pipe.named_steps.get('preprocessor')
                model = pipe.named_steps.get('clf') or pipe.named_steps.get('classifier') or pipe
                proc = preproc.transform(input_df) if preproc is not None else input_df
                probs = pipe.predict_proba(input_df)[0]
                classes = pipe.classes_
                pred = pipe.predict(input_df)[0]
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                probs, classes, pred = None, None, None
            if probs is not None:
                prob_df = pd.DataFrame({'Performance Level': classes, 'Probability': probs}).sort_values('Probability', ascending=False)
                st.subheader("Prediction")
                st.markdown(f"<div class='prediction-card'><h3>Predicted: <strong>{pred}</strong></h3><p class='small-muted'>Confidence: {probs.max():.2f}</p></div>", unsafe_allow_html=True)
                st.subheader("Probabilities")
                st.bar_chart(prob_df.set_index('Performance Level'))
                with st.expander("SHAP Explanation for this prediction"):
                    pass
                csv = input_df.copy()
                csv['predicted'] = pred
                for i, cls in enumerate(classes): csv[f'prob_{cls}'] = probs[i]
                buf = csv.to_csv(index=False).encode('utf-8')
                st.download_button("Download prediction (CSV)", data=buf, file_name='prediction.csv', mime='text/csv')
            else:
                st.info("Model could not compute probabilities.")
        else:
            st.info("Model is not available.")
    else:
        st.info("Please fill out the student profile on the left and click 'Get Prediction'.")

# FOOTER 
st.markdown("---")
add_footer("Jawad Larik")