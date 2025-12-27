# app.py -- SupplyChainAI Streamlit single-file app
# Requirements: streamlit, pandas, numpy, joblib, matplotlib, seaborn, scikit-learn, werkzeug
# Optional: shap (wrapped in try/except)
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
import traceback
import sqlite3
import altair as alt
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

# Visuals
import matplotlib.pyplot as plt
import seaborn as sns
import difflib

# Sklearn helpers
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# --- constants ---
DB_FILE = "users.db"
MODEL_FILE_DEFAULT = "rf_model.pkl"

# --- page config ---
st.set_page_config(page_title="SupplyChain AI", layout="wide", initial_sidebar_state="expanded")
sns.set_style("darkgrid")

# ---------- SQLite helpers ----------
def get_db_conn():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            full_name TEXT,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()

def create_user_db(email, password, full_name=""):
    email = email.strip().lower()
    conn = get_db_conn()
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO users (email, full_name, password_hash, created_at) VALUES (?, ?, ?, ?)",
                    (email, full_name, generate_password_hash(password), datetime.utcnow().isoformat()))
        conn.commit()
        return True, None
    except sqlite3.IntegrityError:
        return False, "Email already registered"
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()

def get_user_by_email(email):
    email = email.strip().lower()
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, email, full_name, password_hash, created_at FROM users WHERE email = ?", (email,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None

def authenticate_user_db(email, password):
    user = get_user_by_email(email)
    if not user:
        return False, "No such user"
    try:
        ok = check_password_hash(user["password_hash"], password)
        if ok:
            user.pop("password_hash", None)
            return True, user
        else:
            return False, "Incorrect password"
    except Exception as e:
        return False, str(e)

# Initialize DB on run
init_db()

# ---------- helpers to patch ColumnTransformer (compat across sklearn versions) ----------
def _patch_column_transformer(obj):
    """
    Recursive patch to fix sklearn version mismatch issues in loaded pipelines,
    specifically related to ColumnTransformer internals.
    """
    try:
        from sklearn.compose import ColumnTransformer as SCT
    except Exception:
        SCT = ColumnTransformer  

    if isinstance(obj, SCT):
        if not hasattr(obj, "_name_to_fitted_passthrough"):
            try:
                setattr(obj, "_name_to_fitted_passthrough", {})
            except Exception:
                pass
        try:
            for name, trans, cols in getattr(obj, "transformers_", []):
                _patch_column_transformer(trans)
        except Exception:
            pass
    elif isinstance(obj, Pipeline):
        for step in (obj.named_steps or {}).values():
            _patch_column_transformer(step)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _patch_column_transformer(item)
    elif isinstance(obj, dict):
        for item in obj.values():
            _patch_column_transformer(item)

def safe_load_model(bytes_io):
    """
    Load a model and patch ColumnTransformer internals to be tolerant across sklearn versions.
    Accepts bytes, file-like, or path.
    """
    model = None
    try:
        if isinstance(bytes_io, (bytes, bytearray)):
            model = joblib.load(io.BytesIO(bytes_io))
        elif hasattr(bytes_io, "read"):
            model = joblib.load(bytes_io)
        elif isinstance(bytes_io, str) and os.path.exists(bytes_io):
            model = joblib.load(bytes_io)
        else:
            model = joblib.load(bytes_io)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    try:
        _patch_column_transformer(model)
    except Exception:
        pass
    return model

# ---------- Model & data helpers ----------
@st.cache_resource
def load_model_from_path(path=MODEL_FILE_DEFAULT):
    if not os.path.exists(path):
        return None, f"Model file not found at {path}"
    try:
        model = safe_load_model(path)
        return model, None
    except Exception as e:
        return None, str(e)

@st.cache_data
def read_csv_bytes(fileobj):
    fileobj.seek(0)
    try:
        return pd.read_csv(fileobj)
    except Exception:
        try:
            fileobj.seek(0)
            return pd.read_csv(fileobj, encoding="latin1")
        except Exception:
            fileobj.seek(0)
            raw = fileobj.read()
            if isinstance(raw, bytes):
                text = raw.decode("utf8", errors="ignore")
                return pd.read_csv(io.StringIO(text))
            else:
                return pd.read_csv(io.StringIO(raw))

def infer_model_columns(model):
    """
    Return (expected_columns_list or None, numeric_cols_list, categorical_cols_list)
    """
    try:
        if hasattr(model, "input_columns") and getattr(model, "input_columns"):
            return list(model.input_columns), getattr(model, "numeric_cols", []), getattr(model, "categorical_cols", [])
    except Exception:
        pass

    numeric_cols, cat_cols = [], []
    try:
        if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
            pre = model.named_steps["preprocessor"]
            if hasattr(pre, "transformers_"):
                for name, trans, cols in pre.transformers_:
                    try:
                        if name == "num":
                            numeric_cols.extend(list(cols))
                        elif name == "cat":
                            cat_cols.extend(list(cols))
                    except Exception:
                        pass
            expected = list(cat_cols) + list(numeric_cols)
            if expected:
                return expected, numeric_cols, cat_cols
    except Exception:
        pass

    try:
        final = model
        if hasattr(model, "named_steps") and "classifier" in model.named_steps:
            final = model.named_steps["classifier"]
        if hasattr(final, "feature_names_in_"):
            cols = list(final.feature_names_in_)
            return cols, [], []
    except Exception:
        pass

    return None, numeric_cols, cat_cols

def prepare_df_for_model(df, model, numeric_default=0, cat_default="UNKNOWN"):
    if df is None:
        return None, []
    df = df.copy()
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    expected, num_cols, cat_cols = infer_model_columns(model)
    if expected is None:
        return df, []
    
    lowercase_map = {c.lower(): c for c in df.columns if isinstance(c, str)}
    rename_map = {}
    for exp in expected:
        if not isinstance(exp, str):
            continue
        if exp in df.columns:
            continue
        exp_l = exp.lower()
        if exp_l in lowercase_map:
            rename_map[lowercase_map[exp_l]] = exp
        else:
            matches = difflib.get_close_matches(exp_l, list(lowercase_map.keys()), n=1, cutoff=0.70)
            if matches:
                rename_map[lowercase_map[matches[0]]] = exp
    if rename_map:
        df = df.rename(columns=rename_map)
    
    added = []
    for col in expected:
        if col not in df.columns:
            if col in num_cols:
                df[col] = numeric_default
            else:
                df[col] = cat_default
            added.append(col)
    
    df_aligned = df.reindex(columns=expected)
    for c in (num_cols or []):
        if c in df_aligned.columns:
            df_aligned[c] = pd.to_numeric(df_aligned[c], errors="coerce")
    for c in df_aligned.columns:
        if df_aligned[c].dtype == object:
            df_aligned[c] = df_aligned[c].fillna("").astype(str)
    return df_aligned, added

def safe_predict(model, df):
    try:
        probs = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df)[:, 1].tolist()
        preds = model.predict(df).tolist()
        return preds, probs, None
    except Exception as e:
        return None, None, str(e)

def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.patch.set_facecolor("#071226"); ax.set_facecolor("#071226")
    plt.tight_layout()
    return fig

def persist_model_bytes(model):
    bio = io.BytesIO()
    joblib.dump(model, bio)
    bio.seek(0)
    return bio.read()

# ---------- Session defaults ----------
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

if "last_model" not in st.session_state:
    loaded_model, load_err = load_model_from_path(MODEL_FILE_DEFAULT)
    st.session_state["last_model"] = loaded_model
    st.session_state["last_model_err"] = load_err
    st.session_state["model_loaded"] = loaded_model is not None

if "current_user" not in st.session_state:
    st.session_state["current_user"] = None

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg,#071226 0%, #041426 100%); color: #e6eef8; }
    [data-testid="stSidebar"] { background: #041426; color: #cfe6f8; }
    .brand { font-weight:800; font-size:18px; color:#fff }
    .hero-title { font-size:34px; font-weight:800; color:#fff; margin-bottom:6px; }
    .hero-sub { color:#9fb3d2; margin-bottom:14px; }
    .kpi { background: rgba(255,255,255,0.03); padding:14px; border-radius:8px; text-align:center }
    .card { background: rgba(255,255,255,0.02); padding:14px; border-radius:8px }
    .small { color:#9fb3d2; font-size:13px }
    .topbtn { background:transparent; border:1px solid rgba(255,255,255,0.08); padding:6px 10px; border-radius:8px; color:#dbe9f6 }
    .cta { background: linear-gradient(90deg,#18c0d0,#6b62dd); color:white; padding:10px 18px; border-radius:8px; font-weight:700; border:none; }
    </style>
    """,
    unsafe_allow_html=True,
)

def kpi_card(title, value, delta, positive=True):
    color = "#16c784" if positive else "#ff4d4f"
    arrow = "▲" if positive else "▼"
    return f"""
    <div style="background:rgba(255,255,255,0.04);padding:18px;border-radius:14px">
        <div style="font-size:13px;color:#9fb3d2">{title}</div>
        <div style="font-size:28px;font-weight:800">{value}</div>
        <div style="color:{color};font-size:13px">{arrow} {delta}</div>
    </div>
    """

# ---------- Sidebar (Refactored) ----------
with st.sidebar:

    # ----- Brand -----
    st.markdown(
        """
        <div style='padding:10px 8px;display:flex;align-items:center;gap:10px'>
          <div style='width:42px;height:42px;border-radius:8px;
               background:linear-gradient(90deg,#18c0d0,#6b62dd)'></div>
          <div>
            <strong style='color:#fff'>SupplyChain AI</strong>
            <div style='font-size:12px;color:#93a6b7'>Delivery Predictor</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ----- MAIN NAVIGATION -----
    main_pages = [
        "Home",
        "Dashboard",
        "Upload & EDA",
        "Train Model",
        "Model Insights",
        "Single Prediction",
        "Batch Prediction",
    ]

    selected_main = st.radio(
        "Navigation",
        main_pages,
        index=main_pages.index(st.session_state["page"])
        if st.session_state["page"] in main_pages else 0
    )

    # Update page
    st.session_state["page"] = selected_main

    # ----- DIVIDER -----
    st.markdown("---")

    # ----- AUTH SECTION -----
    if st.session_state.get("current_user"):
        user = st.session_state["current_user"]

        st.markdown("### 👤 Account")
        st.write(user.get("full_name") or user.get("email"))

        if st.button("🚪 Logout"):
            st.session_state["current_user"] = None
            st.session_state["page"] = "Home"
            st.rerun()

    else:
        auth_page = st.radio(
            "Account",
            ["Sign In", "Sign Up"]
        )
        st.session_state["page"] = auth_page

page = st.session_state["page"]

# ---------- Top nav (Minimal) ----------
# Since Sidebar handles Nav/Auth now, Top nav is just visual branding
st.markdown(
    """
    <div style='padding:15px 0px;display:flex;align-items:center;justify-content:space-between'>
        <div class='brand' style='font-size:24px'>SupplyChain AI</div>
        <div style='font-size:13px;color:#9fb3d2'>Enterprise ML Solution</div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------- HOME ----------
if page == "Home":
    left, right = st.columns([3,1])
    with left:
        st.markdown("<div class='hero-title'>AI-Powered Supply Chain</div>", unsafe_allow_html=True)
        st.markdown("<div class='small'>Predict Delivery Delays with Machine Learning</div>", unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom:12px'>A data-driven model achieving 98.12% accuracy in predicting supply chain delivery delays. Make informed decisions and optimize your logistics operations.</div>", unsafe_allow_html=True)
        c1, c2 = st.columns([1,1])
        with c1:
            if st.button("Get Started Free", key="home_cta_start"):
                st.session_state["page"] = "Sign Up"
        with c2:
            if st.button("View Demo", key="home_cta_demo"):
                st.session_state["page"] = "Single Prediction"
    with right:
        st.markdown("<div style='height:140px;border-radius:8px;background:linear-gradient(90deg,#18c0d0,#6b62dd)'></div>", unsafe_allow_html=True)

    st.markdown("---")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown("<div class='kpi'><div style='font-weight:800;font-size:20px'>98.12%</div><div class='small'>Prediction Accuracy</div></div>", unsafe_allow_html=True)
    k2.markdown("<div class='kpi'><div style='font-weight:800;font-size:20px'>50K+</div><div class='small'>Predictions Daily</div></div>", unsafe_allow_html=True)
    k3.markdown("<div class='kpi'><div style='font-weight:800;font-size:20px'>23</div><div class='small'>Data Points</div></div>", unsafe_allow_html=True)
    k4.markdown("<div class='kpi'><div style='font-weight:800;font-size:20px'>&lt;1s</div><div class='small'>Response Time</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## Supply Chain Analytics Dashboard")
    f1, f2, f3, f4 = st.columns(4)
    f1.markdown("<div class='card'><strong>AI-Powered Predictions</strong><div class='small'>Advanced Random Forest model with enterprise accuracy predicts delivery delays with confidence.</div></div>", unsafe_allow_html=True)
    f2.markdown("<div class='card'><strong>Interactive Dashboard</strong><div class='small'>Real-time analytics with visual insights into your supply chain performance.</div></div>", unsafe_allow_html=True)
    f3.markdown("<div class='card'><strong>Batch Processing</strong><div class='small'>Upload CSV files for bulk predictions and comprehensive analysis.</div></div>", unsafe_allow_html=True)
    f4.markdown("<div class='card'><strong>Model Explainability</strong><div class='small'>Understand which factors impact your deliveries (explains available where safe).</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## Why Choose SupplyChain AI")
    st.markdown("- 98.12% prediction accuracy with Random Forest ML\n- Real-time supply chain visibility\n- Comprehensive delay risk assessment\n- Actionable insights for optimization\n- Secure cloud-based infrastructure\n- Export and download prediction results")

    st.markdown("---")
    st.markdown("## Machine Learning Model")
    st.markdown("Random Forest classifier trained on thousands of supply chain data points, achieving industry-leading accuracy in delay prediction. Continuous improvement and visual analytics included.")

    st.markdown("---")
    st.markdown("<div style='text-align:left'><a class='cta' href='#'>Start Predicting Now</a></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

# ---------- SIGN IN ----------
if page == "Sign In":
    st.subheader("Welcome back — Sign in")
    st.markdown("Enter your work email and password.")
    cL, cR = st.columns([2,1])
    with cL:
        with st.form("signin_form"):
            si_email = st.text_input("Email")
            si_password = st.text_input("Password", type="password")
            remember = st.checkbox("Remember me (session only)", value=True)
            submitted = st.form_submit_button("Sign in")
        if submitted:
            if not si_email or not si_password:
                st.error("Enter both email and password.")
            else:
                ok, user_or_err = authenticate_user_db(si_email, si_password)
                if ok:
                    st.success("Signed in successfully.")
                    st.session_state["current_user"] = user_or_err
                    st.session_state["page"] = "Dashboard"
                    st.rerun()
                else:
                    st.error(f"Sign in failed: {user_or_err}")
    with cR:
        st.markdown("<div class='card'><strong>Need help?</strong><div class='small'>If you forgot your password, contact the admin (demo app).</div></div>", unsafe_allow_html=True)

# ---------- SIGN UP ----------
if page == "Sign Up":
    st.subheader("Create your account")
    st.markdown("Sign up to access SupplyChain AI.")

    cL, cR = st.columns([2,1])
    with cL:
        with st.form("signup_form"):
            su_name = st.text_input("Full Name")
            su_email = st.text_input("Email")
            su_password = st.text_input("Password", type="password")
            su_confirm = st.text_input("Confirm Password", type="password")

            submitted = st.form_submit_button("Create Account")

        if submitted:
            if not su_email or not su_password:
                st.error("Email and password are required.")
            elif su_password != su_confirm:
                st.error("Passwords do not match.")
            else:
                ok, err = create_user_db(
                    email=su_email,
                    password=su_password,
                    full_name=su_name
                )
                if ok:
                    st.success("Account created successfully. Please sign in.")
                    st.session_state["page"] = "Sign In"
                else:
                    st.error(err)

    with cR:
        st.markdown(
            "<div class='card'><strong>Already have an account?</strong>"
            "<div class='small'>Go to Sign In page.</div></div>",
            unsafe_allow_html=True
        )

# ---------- DASHBOARD ----------
if page == "Dashboard":
    if not st.session_state.get("current_user"):
        st.warning("Please sign in to access the Dashboard.")
    else:
        st.markdown("## 📊 Supply Chain Dashboard")
        st.markdown("<div class='small'>Real-time insights into delivery performance</div>", unsafe_allow_html=True)

        # KPI Cards
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(kpi_card("Total Orders", "1,247", "12.5%"), unsafe_allow_html=True)
        k2.markdown(kpi_card("Late Deliveries", "8.4%", "3.2%", positive=False), unsafe_allow_html=True)
        k3.markdown(kpi_card("Model Accuracy", "98.12%", "0.8%"), unsafe_allow_html=True)
        k4.markdown(kpi_card("High-Risk Markets", "23", "5.1%", positive=False), unsafe_allow_html=True)

        st.markdown("---")

        # BAR CHART
        st.markdown("### 📦 Delivery Status by Category")
        df_bar = pd.DataFrame({
            "Category": ["Electronics", "Fashion", "Home & Garden", "Sports"],
            "On Time": [250, 180, 210, 150],
            "Late": [30, 40, 20, 15]
        }).melt("Category", var_name="Status", value_name="Orders")

        bar_chart = alt.Chart(df_bar).mark_bar(
            cornerRadiusTopLeft=6,
            cornerRadiusTopRight=6
        ).encode(
            x=alt.X("Category:N", title=""),
            y=alt.Y("Orders:Q", title="Orders"),
            color=alt.Color(
                "Status:N",
                scale=alt.Scale(range=["#4dabf7", "#ff922b"])
            ),
            tooltip=["Category", "Status", "Orders"]
        ).properties(height=320)

        st.altair_chart(bar_chart, use_container_width=True)

        st.markdown("---")

        # DONUT CHART
        st.markdown("### 🌍 Late Deliveries by Region")
        df_pie = pd.DataFrame({
            "Region": ["North America", "South America", "Asia", "Europe"],
            "Percent": [28, 14, 36, 22]
        })

        donut_chart = alt.Chart(df_pie).mark_arc(innerRadius=70).encode(
            theta="Percent:Q",
            color=alt.Color(
                "Region:N",
                legend=alt.Legend(orient="bottom")
            ),
            tooltip=["Region", "Percent"]
        ).properties(height=350)

        st.altair_chart(donut_chart, use_container_width=True)

# ---------- Upload & EDA ----------
if page == "Upload & EDA":
    st.header("Upload dataset and explore")
    uploaded = st.file_uploader("Upload CSV for EDA", type=["csv"])
    if uploaded is not None:
        try:
            df = read_csv_bytes(uploaded)
            st.success("File loaded")
            with st.expander("Preview (first 20 rows)"):
                st.dataframe(df.head(20))
            st.write("Rows:", df.shape[0], "Columns:", df.shape[1])
            st.write("Missing values per column (top 20):")
            st.dataframe(df.isna().sum().sort_values(ascending=False).head(20))
            numeric = df.select_dtypes(include=[np.number])
            categorical = df.select_dtypes(include=["object", "category"])
            with st.expander("Numeric summary"):
                st.dataframe(numeric.describe().T)
            with st.expander("Categorical summary (top values)"):
                tops = {c: df[c].value_counts().head(5).to_dict() for c in categorical.columns[:10]}
                st.json(tops)
        except Exception as e:
            st.error("Failed to load CSV: " + str(e))
            st.exception(traceback.format_exc())


# ---------- Train Model ----------
if page == "Train Model":
    st.header("Train RandomForest model (quick)")
    uploaded = st.file_uploader("Upload CSV for training", type=["csv"], key="train_csv")
    if uploaded is not None:
        try:
            df = read_csv_bytes(uploaded)
            st.success("Training CSV loaded")
            st.dataframe(df.head(10))
        except Exception as e:
            st.error("Failed to read training CSV: " + str(e))
            df = None
        if df is not None:
            target = st.selectbox("Select target column (binary: 0/1 or label)", options=[None] + list(df.columns))
            if target:
                if st.button("Run quick train (50 trees)"):
                    try:
                        drop_cols = [c for c in df.columns if c.lower().endswith("id") or c.lower()=="id"]
                        X = df.drop(columns=[target] + drop_cols, errors="ignore")
                        y = df[target]
                        mask = ~y.isna()
                        X = X.loc[mask]
                        y = y.loc[mask]
                        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
                        cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
                        num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
                        cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))])
                        pre = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)], remainder="drop")
                        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                        model = Pipeline([("preprocessor", pre), ("classifier", rf)])
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=(y if y.nunique()<=10 else None))
                        model.fit(X_train, y_train)
                        try:
                            model.input_columns = list(X.columns)
                            model.numeric_cols = num_cols
                            model.categorical_cols = cat_cols
                        except Exception:
                            pass
                        try:
                            _patch_column_transformer(model)
                        except Exception:
                            pass
                        st.session_state["last_model"] = model
                        st.session_state["model_loaded"] = True
                        preds = model.predict(X_test)
                        acc = accuracy_score(y_test, preds)
                        st.success(f"Trained. Test accuracy: {acc:.3f}")
                        cm = confusion_matrix(y_test, preds)
                        st.pyplot(plot_confusion_matrix(cm))
                        b = persist_model_bytes(model)
                        st.download_button("Download trained model", data=b, file_name=f"rf_model_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pkl")
                    except Exception as e:
                        st.error("Training failed: " + str(e))
                        st.exception(traceback.format_exc())

# ---------- Single Prediction ----------
if page == "Single Prediction":
    st.header("Single Prediction")
    st.info("Create a single input via form, JSON paste, or one-row CSV.")
    model = st.session_state.get("last_model")
    if not model:
        st.warning("Model not loaded. Upload in Model Insights or place rf_model.pkl in app root.")
    mode = st.radio("Input mode", ["Manual form", "Paste JSON", "Upload CSV"], horizontal=True)
    input_df = None
    if mode == "Manual form":
        with st.form("single_form"):
            c1, c2 = st.columns(2)
            with c1:
                product_name = st.text_input("Product Name", "Phone X")
                product_cat = st.selectbox("Category", ["Electronics", "Fashion", "Home & Garden", "Sports"])
                market = st.text_input("Market", "Asia")
                order_qty = st.number_input("Order Item Quantity", min_value=1, value=1)
            with c2:
                shipping_mode = st.selectbox("Shipping Mode", ["Standard", "Express", "Same Day"])
                days_real = st.number_input("Days for shipping (real)", min_value=0, value=5)
                days_sched = st.number_input("Days for shipment (scheduled)", min_value=0, value=4)
                sales = st.number_input("Sales", value=150.0)
            submitted = st.form_submit_button("Create input")
        if submitted:
            input_df = pd.DataFrame([{
                "Product Name": product_name,
                "Category Name": product_cat,
                "Market": market,
                "Order Item Quantity": order_qty,
                "Shipping Mode": shipping_mode,
                "Days for shipping (real)": days_real,
                "Days for shipment (scheduled)": days_sched,
                "Sales": sales
            }])
            st.dataframe(input_df)
    elif mode == "Paste JSON":
        txt = st.text_area("Paste JSON object or list", height=160)
        if st.button("Load JSON"):
            try:
                txt_str = txt.strip()
                if not txt_str:
                    st.error("Empty input.")
                else:
                    if txt_str.startswith("["):
                        parsed = pd.read_json(io.StringIO(txt_str))
                    else:
                        parsed = pd.DataFrame([eval(txt_str)])
                    input_df = parsed
                    st.success("Loaded JSON")
                    st.dataframe(input_df)
            except Exception as e:
                st.error(f"Failed to parse JSON/object: {e}")
    else:
        uploaded = st.file_uploader("Upload a one-row CSV", type=["csv"])
        if uploaded is not None:
            try:
                input_df = read_csv_bytes(uploaded)
                st.dataframe(input_df)
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
    if input_df is not None and st.button("Predict this row"):
        if not model:
            st.error("Model not loaded.")
        else:
            try:
                df_prepared, added = prepare_df_for_model(input_df, model)
                if added:
                    st.warning(f"{len(added)} columns added to align with model (defaults).")
                preds, probs, perr = safe_predict(model, df_prepared)
                if perr:
                    st.error(f"Prediction error: {perr}")
                else:
                    st.success("Prediction completed")
                    st.write("Predicted class:", preds[0])
                    if probs:
                        st.write("Probability (late):", f"{probs[0]:.4f}")
                    out = input_df.copy()
                    out["predicted"] = preds
                    if probs:
                        out["probability"] = probs
                    st.download_button("Download result CSV", data=out.to_csv(index=False).encode("utf-8"), file_name="single_prediction.csv")
            except Exception as e:
                st.error("Prediction pipeline failed: " + str(e))
                st.exception(traceback.format_exc())

# ---------- Batch Prediction ----------
if page == "Batch Prediction":
    st.header("Batch Prediction")
    st.markdown("Upload a CSV and either use the loaded model (rf_model.pkl or session model) or upload a model file to run batch predictions.")
    uploaded = st.file_uploader("Upload CSV for batch prediction", type=["csv"], key="batch_csv")
    model_upload = st.file_uploader("Upload model (.pkl/.joblib) to use (optional)", type=["pkl","joblib"], key="batch_model")
    if model_upload is not None:
        try:
            mdl = safe_load_model(io.BytesIO(model_upload.read()))
            st.session_state["last_model"] = mdl
            st.session_state["model_loaded"] = True
            st.success("Uploaded model loaded into session.")
        except Exception as e:
            st.error("Failed to load uploaded model: " + str(e))
    model = st.session_state.get("last_model")
    if uploaded is not None:
        try:
            df = read_csv_bytes(uploaded)
            st.write("Rows:", len(df))
            st.dataframe(df.head(5))
        except Exception as e:
            st.error("Failed to read CSV: " + str(e))
            df = None
        if df is not None and st.button("Run batch prediction now"):
            if not model:
                st.error("No model available.")
            else:
                try:
                    df_prep, added = prepare_df_for_model(df, model)
                    if added:
                        st.warning(f"{len(added)} columns added to align with model (default filled).")
                    preds, probs, perr = safe_predict(model, df_prep)
                    if perr:
                        st.error("Prediction error: " + perr)
                    else:
                        df_out = df.copy()
                        df_out["predicted_delay"] = preds
                        if probs:
                            df_out["probability"] = probs
                        st.dataframe(df_out.head(50))
                        st.download_button("Download predictions CSV", data=df_out.to_csv(index=False).encode("utf-8"), file_name="batch_predictions.csv")
                except Exception as e:
                    st.error("Batch prediction failed: " + str(e))
                    st.exception(traceback.format_exc())

# ---------- Model Insights ----------
if page == "Model Insights":
    st.header("Model Insights & Upload")
    model = st.session_state.get("last_model")
    if model is None:
        st.info("No model in session. Upload or place rf_model.pkl in app root and press Reload.")
        if st.button("Reload model from disk"):
            loaded_model, load_err = load_model_from_path(MODEL_FILE_DEFAULT)
            st.session_state["last_model"] = loaded_model
            st.session_state["last_model_err"] = load_err
            st.session_state["model_loaded"] = loaded_model is not None
    else:
        st.success("Model in session")
        expected, num_cols, cat_cols = infer_model_columns(model)
        st.write("Model type:", type(model))
        st.write("Expected columns (first 30):", expected[:30] if expected else "Unknown")
        if num_cols:
            st.write("Numeric cols (sample):", num_cols[:10])
        if cat_cols:
            st.write("Categorical cols (sample):", cat_cols[:10])

        with st.expander("Model attributes & methods"):
            try:
                attrs = {k: getattr(model, k) for k in dir(model) if (not k.startswith("_")) and k.endswith("_")}
                simple = {k: str(v) for k,v in list(attrs.items())[:40]}
                st.json(simple)
            except Exception:
                st.write("Could not introspect fully.")

        fi = None
        try:
            clf = model.named_steps.get("classifier", None) if hasattr(model, "named_steps") else None
            if clf is not None and hasattr(clf, "feature_importances_"):
                importances = clf.feature_importances_
                feat_names = []
                pre = model.named_steps.get("preprocessor", None) if hasattr(model, "named_steps") else None
                if pre is not None and hasattr(pre, "transformers_"):
                    for name, trans, cols in pre.transformers_:
                        if isinstance(cols, (list, tuple, np.ndarray)):
                            feat_names.extend(cols)
                if len(importances) == len(feat_names):
                    fi = pd.Series(importances, index=feat_names).sort_values(ascending=False).head(20)
                else:
                    fi = pd.Series(importances).sort_values(ascending=False).head(20)
        except Exception:
            fi = None

        if fi is not None:
            st.subheader("Feature importances (top 20)")
            st.bar_chart(fi)
            with st.expander("Download feature importance CSV"):
                csv = fi.reset_index().rename(columns={"index":"feature", 0:"importance"}).to_csv(index=False)
                st.download_button("Download FI CSV", data=csv.encode("utf-8"), file_name="feature_importances.csv")
        else:
            st.info("Feature importances not available for this model.")

    st.markdown("---")
    st.markdown("Upload a model (pkl/joblib) to inspect in session")
    upload_model = st.file_uploader("Upload model file", type=["pkl", "joblib"], key="inspect_model")
    if upload_model is not None:
        try:
            content = upload_model.read()
            m = safe_load_model(io.BytesIO(content))
            st.session_state["last_model"] = m
            st.session_state["model_loaded"] = True
            st.success("Uploaded model loaded into session.")
        except Exception as e:
            st.error("Failed to load model: " + str(e))

# footer spacing
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)