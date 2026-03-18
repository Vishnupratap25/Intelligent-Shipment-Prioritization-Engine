import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import shap

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Intelligent Shipment Prioritization Engine",
    layout="wide"
)
# ==============================
# DARK MODE DASHBOARD STYLING (NEW ADDITION)
# ==============================

st.markdown("""
<style>

/* Keep Streamlit header visible */
header {visibility: visible;}
footer {visibility: hidden;}

/* Fix Streamlit top header color */
header[data-testid="stHeader"]{
    background-color:#0E1117 !important;
}

[data-testid="stToolbar"]{
    background-color:#0E1117 !important;
}

header{
    box-shadow:none !important;
}

/* Main App Background */
[data-testid="stAppViewContainer"]{
    background-color:#0E1117;
}

/* Sidebar */
[data-testid="stSidebar"]{
    background-color:#111827;
}

/* Headings */
h1, h2, h3, h4, h5, h6{
    color:#FFFFFF !important;
    font-weight:700;
}

/* Text */
p, span, div, label{
    color:#F9FAFB !important;
}

/* Metric Cards */
div[data-testid="metric-container"]{
    background-color:#1F2937;
    border-radius:10px;
    padding:15px;
    border:1px solid #374151;
}

/* Tabs */
.stTabs [data-baseweb="tab"]{
    font-size:16px;
    color:#E5E7EB;
}

.stTabs [aria-selected="true"]{
    color:#22C55E !important;
    font-weight:600;
}

/* File uploader container */
[data-testid="stFileUploader"]{
    background-color:#1F2937;
    border-radius:10px;
    border:1px solid #374151;
    padding:10px;
}

/* Fix Drag & Drop text visibility */
[data-testid="stFileUploaderDropzone"]{
    background-color:#1F2937 !important;
}

[data-testid="stFileUploaderDropzone"] *{
    color:#F9FAFB !important;
}

[data-testid="stFileUploaderDropzoneInstructions"]{
    color:#F9FAFB !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] span{
    color:#D1D5DB !important;
}

/* Input fields */
input{
    background-color:#1F2937 !important;
    color:white !important;
    border:1px solid #374151 !important;
}

/* Buttons */
button{
    background-color:#2563EB !important;
    color:white !important;
    border-radius:8px !important;
    border:none !important;
}

/* Secondary button */
button[kind="secondary"]{
    background-color:#22C55E !important;
}

/* Hover */
button:hover{
    background-color:#1D4ED8 !important;
}

/* =====================================================
   FIX DROPDOWN TEXT VISIBILITY (ADDED — NOTHING REMOVED)
   ===================================================== */

div[data-baseweb="popover"]{
    background-color:#1F2937 !important;
}

div[data-baseweb="menu"] div{
    color:white !important;
}

div[data-baseweb="tag"]{
    background-color:#2563EB !important;
    color:white !important;
}

div[data-baseweb="select"] span{
    color:white !important;
}

div[data-baseweb="menu"] div:hover{
    background-color:#374151 !important;
    color:white !important;
}

/* =====================================================
   ADVANCED DROPDOWN FIX
   ===================================================== */

[data-baseweb="popover"] [role="listbox"]{
    background-color:#1F2937 !important;
}

[data-baseweb="popover"] [role="option"]{
    background-color:#1F2937 !important;
    color:#FFFFFF !important;
    font-weight:500 !important;
}

[data-baseweb="popover"] [role="option"]:hover{
    background-color:#374151 !important;
}

[data-baseweb="popover"] [aria-selected="true"]{
    background-color:#2563EB !important;
}

/* =====================================================
   FINAL FIX FOR SELECT BOX VISIBILITY
   ===================================================== */

div[data-baseweb="select"] > div{
    background-color:#1F2937 !important;
    color:#FFFFFF !important;
    border:1px solid #374151 !important;
}

div[data-baseweb="select"] input{
    color:#FFFFFF !important;
}

/* =====================================================
   TOOLTIP VISIBILITY FIX
   ===================================================== */

[data-testid="stTooltipContent"]{
    background-color:#FFFFFF !important;
    color:#000000 !important;
    font-size:14px !important;
    border-radius:6px !important;
    padding:8px !important;
    max-width:260px !important;
}

[data-testid="stTooltipContent"] *{
    color:#000000 !important;
}

/* =====================================================
   DATAFRAME COLUMN MENU FIX
   ===================================================== */

[data-testid="stDataFrame"] [role="menu"]{
    background-color:#FFFFFF !important;
}

[data-testid="stDataFrame"] [role="menuitem"]{
    color:#000000 !important;
    font-weight:500 !important;
}

[data-testid="stDataFrame"] [role="menuitem"]:hover{
    background-color:#E5E7EB !important;
}

/* =====================================================
   AG-GRID MENU FIX
   ===================================================== */

.ag-menu{
    background-color:#FFFFFF !important;
}

.ag-menu-option{
    color:#000000 !important;
    font-weight:500 !important;
}

.ag-menu-option:hover{
    background-color:#E5E7EB !important;
}

.ag-menu-option-icon{
    color:#000000 !important;
}

.ag-menu-separator{
    border-top:1px solid #D1D5DB !important;
}

.ag-menu-option.ag-disabled{
    color:#6B7280 !important;
}

</style>
""", unsafe_allow_html=True)


st.title("🚀 Intelligent Shipment Prioritization Engine")
st.caption("AI-Powered SLA Breach Risk Dashboard")

# ==============================
# HUB CODE → CITY MAPPING
# ==============================

HUB_TO_CITY = {
    "BLRA": "BANGALORE",
    "NDCA": "DELHI",
    "HYDBG": "HYDERABAD",
    "MAATS": "CHENNAI"
}

# ==============================
# MAJOR INDIAN CITIES + CAPITALS
# ==============================

CITY_COORDS = {

# Capitals
"DELHI": (28.7041,77.1025),
"MUMBAI": (19.0760,72.8777),
"CHENNAI": (13.0827,80.2707),
"BANGALORE": (12.9716, 77.5946),
"HYDERABAD": (17.3850,78.4867),
"KOLKATA": (22.5726,88.3639),
"JAIPUR": (26.9124,75.7873),
"LUCKNOW": (26.8467,80.9462),
"PATNA": (25.5941,85.1376),
"BHOPAL": (23.2599,77.4126),
"RAIPUR": (21.2514,81.6296),
"RANCHI": (23.3441,85.3096),
"DEHRADUN": (30.3165,78.0322),
"SHIMLA": (31.1048,77.1734),
"GANDHINAGAR": (23.2156,72.6369),
"CHANDIGARH": (30.7333,76.7794),

# Major Cities
"PUNE": (18.5204,73.8567),
"AHMEDABAD": (23.0225,72.5714),
"SURAT": (21.1702,72.8311),
"NAGPUR": (21.1458,79.0882),
"INDORE": (22.7196,75.8577),
"VADODARA": (22.3072,73.1812),
"VISAKHAPATNAM": (17.6868,83.2185),
"COIMBATORE": (11.0168,76.9558),
"MADURAI": (9.9252,78.1198),
"TIRUCHIRAPPALLI": (10.7905,78.7047),
"KOCHI": (9.9312,76.2673),
"NOIDA": (28.5355,77.3910),
"GURGAON": (28.4595,77.0266),
"GHAZIABAD": (28.6692,77.4538),
"KANPUR": (26.4499,80.3319),
"VARANASI": (25.3176,82.9739),
"AGRA": (27.1767,78.0081),
"MEERUT": (28.9845,77.7064),
"MYSORE": (12.2958,76.6394),
"MANGALORE": (12.9141,74.8560),
"HUBLI": (15.3647,75.1240),
"VIJAYAWADA": (16.5062,80.6480),
"GUNTUR": (16.3067,80.4365),
"GUWAHATI": (26.1445,91.7362),
"SILIGURI": (26.7271,88.3953),
"THANE": (19.2183,72.9781),
"NASHIK": (19.9975,73.7898),
"AURANGABAD": (19.8762,75.3433),
"KOLHAPUR": (16.7050,74.2433)
}

# ==============================
# LOAD MODEL
# ==============================

@st.cache_resource
def load_artifacts():

    MODEL_PATH = "models/sla_binary_stable_model_20260304_1121.pkl"

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found: {MODEL_PATH}")
        st.stop()

    model = joblib.load(MODEL_PATH)
    label_encoders = joblib.load("models/label_encoders.pkl")
    threshold = joblib.load("models/failure_threshold.pkl")

    return model,label_encoders,threshold

# ==============================
# PREPROCESS
# ==============================

def preprocess(df,label_encoders,model):

    df=df.copy().fillna("Unknown")

    for col,le in label_encoders.items():
        if col in df.columns:
            df[col]=df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )

    drop_cols=[
        "commit_status","target","target_binary",
        "IST_svc_commit_tmstp","time_diff_hours","last_scan"
    ]

    X=df.drop(columns=drop_cols,errors="ignore")

    for col in X.columns:
        X[col]=pd.to_numeric(X[col],errors="coerce")

    X=X.fillna(0)

    if hasattr(model,"feature_names_in_"):

        expected_cols=list(model.feature_names_in_)

        for col in expected_cols:
            if col not in X.columns:
                X[col]=0

        X=X[expected_cols]

    return X.astype(float)

# ==============================
# POWER BI EXPORT CLEANER (NEW ADDITION)
# ==============================

def clean_powerbi_export(df):

    df = df.copy()

    # Remove empty rows
    df = df.dropna(how="all")

    # Reset index
    df = df.reset_index(drop=True)

    # Fix Power BI column format like Table[Column]
    df.columns = [str(col).split("[")[-1].replace("]", "") for col in df.columns]

    # Strip spaces
    df.columns = df.columns.str.strip()

    return df

# ==============================
# RISK CATEGORY
# ==============================

def categorize_risk(p):

    if p>=70:
        return "High Risk"
    elif p>=40:
        return "Medium Risk"
    else:
        return "Low Risk"

# ==============================
# CACHE PREDICTION
# ==============================

@st.cache_data
def run_model_prediction(df):

    X=preprocess(df,label_encoders,model)

    probs=model.predict_proba(X)[:,1]

    return probs
# ==============================
# FILE UPLOAD
# ==============================

uploaded_file = st.file_uploader(
    "Upload Shipment File (CSV / Excel)",
    type=["csv", "xlsx"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

    model, label_encoders, threshold = load_artifacts()

    st.sidebar.success("Model Loaded: SLA Binary Stable Model")

    if "Trk Nos" in df.columns:
        df["Trk Nos"] = df["Trk Nos"].astype(str)

    for col in df.columns:
        if "city" in col.lower() or "loc" in col.lower():
            df[col] = df[col].replace(HUB_TO_CITY)

    probs = run_model_prediction(df)

    df["Failure_Risk_%"] = (probs * 100).round(2)
    df["Risk_Category"] = df["Failure_Risk_%"].apply(categorize_risk)

    # # ==============================
    # # SIDEBAR FILTERS
    # # ==============================

    # st.sidebar.header("🔎 Filter Shipments")

    # risk_filter = st.sidebar.multiselect(
    #     "Risk Category",
    #     df["Risk_Category"].unique(),
    #     default=df["Risk_Category"].unique()
    # )

    # min_risk, max_risk = st.sidebar.slider("Failure Risk %", 0, 100, (0, 100))

    # filtered_df = df[
    #     (df["Risk_Category"].isin(risk_filter)) &
    #     (df["Failure_Risk_%"] >= min_risk) &
    #     (df["Failure_Risk_%"] <= max_risk)
    # ]

    # ==============================
    # CREATE TABS
    # ==============================

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Overview",
            "📈 Risk Analysis",
            "⏳ NCL Monitor",
            "🔮 Predict Single Shipment",
        ]
    )
# ==============================
# TAB 1
# ==============================

    with tab1:

        # Fix dropdown text visibility in dark mode
        st.markdown("""
        <style>
        div[data-baseweb="select"] > div {
            color: white !important;
        }
        div[data-baseweb="select"] input {
            color: white !important;
        }
        </style>
        """, unsafe_allow_html=True)

        total = len(filtered_df)
        high_count = (filtered_df["Risk_Category"] == "High Risk").sum()
        medium_count = (filtered_df["Risk_Category"] == "Medium Risk").sum()
        low_count = (filtered_df["Risk_Category"] == "Low Risk").sum()

        high_pct = round((high_count / total) * 100, 2)
        expected_breaches = (filtered_df["Failure_Risk_%"] >= threshold * 100).sum()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Shipments", total)
        col2.metric("High Risk %", f"{high_pct}%")
        col3.metric("Medium Risk", medium_count)
        col4.metric("Predicted Breaches", expected_breaches)

        st.subheader("📦 Risk Category Breakdown")

        c1, c2, c3 = st.columns(3)

        c1.metric("🔴 High Risk", high_count)
        c2.metric("🟠 Medium Risk", medium_count)
        c3.metric("🟢 Low Risk", low_count)

        st.divider()

        st.subheader("📍 Complete City-Level Risk Dashboard")

        city_cols = [c for c in filtered_df.columns if "city" in c.lower()]

        if city_cols:

            city_col = city_cols[0]

            city_summary = (
                filtered_df
                .groupby(city_col)
                .agg(
                    Total_Shipments=("Risk_Category", "count"),
                    High_Risk_Shipments=("Risk_Category", lambda x: (x == "High Risk").sum()),
                    Medium_Risk_Shipments=("Risk_Category", lambda x: (x == "Medium Risk").sum()),
                    Low_Risk_Shipments=("Risk_Category", lambda x: (x == "Low Risk").sum())
                )
                .reset_index()
            )

            # ==============================
            # SORTING OPTIONS
            # ==============================

            colA, colB = st.columns(2)

            with colA:
                sort_col = st.selectbox(
                    "Sort By",
                    [
                        city_col,
                        "Total_Shipments",
                        "High_Risk_Shipments",
                        "Medium_Risk_Shipments",
                        "Low_Risk_Shipments"
                    ]
                )

            with colB:
                sort_order = st.selectbox(
                    "Order",
                    ["Descending", "Ascending"]
                )

            ascending = True if sort_order == "Ascending" else False

            city_summary = city_summary.sort_values(
                by=sort_col,
                ascending=ascending
            )

            st.dataframe(city_summary, use_container_width=True)

        # ==============================
        # CITY SEARCH DROPDOWN
        # ==============================

        st.divider()
        st.subheader("🔎 Search City Shipment Risk")

        city_list = sorted(city_summary[city_col].dropna().unique())
        city_list = ["Select City"] + city_list

        selected_city = st.selectbox(
            "Select or Search City",
            city_list
        )

        if selected_city != "Select City":

            city_data = city_summary[city_summary[city_col] == selected_city]

            if not city_data.empty:

                shipments = int(city_data["Total_Shipments"].values[0])
                high_risk_city = int(city_data["High_Risk_Shipments"].values[0])
                medium_risk_city = int(city_data["Medium_Risk_Shipments"].values[0])
                low_risk_city = int(city_data["Low_Risk_Shipments"].values[0])

                colX, colY, colZ, colW = st.columns(4)

                colX.metric("📦 Total Shipments", shipments)
                colY.metric("🔴 High Risk Shipments", high_risk_city)
                colZ.metric("🟠 Medium Risk Shipments", medium_risk_city)
                colW.metric("🟢 Low Risk Shipments", low_risk_city)

                city_shipments = filtered_df[
                    filtered_df[city_col] == selected_city
                ]

                st.divider()
                st.subheader(f"📦 Shipment Details for {selected_city}")

                if not city_shipments.empty:

                    city_shipments = city_shipments.sort_values(
                        by="Failure_Risk_%",
                        ascending=False
                    )

                    st.dataframe(city_shipments, use_container_width=True)

                    st.download_button(
                        label="⬇ Download Shipment Data",
                        data=city_shipments.to_csv(index=False),
                        file_name=f"{selected_city}_shipments.csv",
                        mime="text/csv"
                    )

        # ==============================
        # TOP 10 HIGH RISK CITIES (RISK COMPOSITION)
        # ==============================

        st.divider()
        st.subheader("🔥 Top 10 Cities Risk Composition")

        # Filter cities with enough shipments
        city_summary_filtered = city_summary[
            city_summary["Total_Shipments"] >= 10
        ]

        # Select top 10 cities by high risk shipments
        top_risk_cities = city_summary_filtered.sort_values(
            "High_Risk_Shipments",
            ascending=False
        ).head(10)

        # Create stacked bar chart
        fig_top = px.bar(
            top_risk_cities,
            x=city_col,
            y=[
                "High_Risk_Shipments",
                "Medium_Risk_Shipments",
                "Low_Risk_Shipments"
            ],
            title="Top 10 Cities Shipment Risk Distribution",
            color_discrete_map={
                "High_Risk_Shipments": "#EF4444",   # red
                "Medium_Risk_Shipments": "#F59E0B", # orange
                "Low_Risk_Shipments": "#22C55E"     # green
            }
        )

        fig_top.update_layout(
            barmode="stack",
            xaxis_title="City",
            yaxis_title="Number of Shipments"
        )

        st.plotly_chart(fig_top, use_container_width=True)

        # ==============================
        # AI OPERATIONAL INSIGHTS
        # ==============================

        st.divider()
        st.subheader("🧠 AI Operational Insights")

        worst_city = city_summary.sort_values(
            "High_Risk_Shipments",
            ascending=False
        ).iloc[0]

        worst_city_name = worst_city[city_col]
        high_ship_count = worst_city["High_Risk_Shipments"]
        total_ship_count = worst_city["Total_Shipments"]

        colA, colB = st.columns(2)

        with colA:
            st.info(
    f"""
    🚨 **Highest Risk Region**

    **{worst_city_name}** has the highest number of high-risk shipments.

    Suggested action:
    • Monitor shipments closely  
    • Prioritize hub processing
    """
            )

        with colB:
            st.warning(
    f"""
    ⚠ **Operational Alert**

    **{worst_city_name}** currently has the highest number of high-risk shipments.

    📦 Total Shipments: **{int(total_ship_count)}**

    🔴 High Risk Shipments: **{int(high_ship_count)}**
    """
                    )
            
        # ==============================
        # INDIA SHIPMENT RISK HEATMAP
        # ==============================

        st.subheader("🗺 India Shipment Risk Heatmap")

        # ==============================
        # CITY ALIAS MAPPING
        # ==============================

        CITY_ALIAS = {
            "BENGALURU": "BANGALORE",
            "BANGALORE URBAN": "BANGALORE",
            "BANGALORE RURAL": "BANGALORE",
            "BLR": "BANGALORE",
            "DELHI NCR": "DELHI"
        }

        # ==============================
        # FILTER ONLY IMPORTANT CITIES
        # ==============================

        city_summary["city_upper"] = (
            city_summary[city_col]
            .astype(str)
            .str.upper()
            .str.strip()
            .replace(CITY_ALIAS)
        )

        city_summary = city_summary[
            city_summary["city_upper"].isin(CITY_COORDS.keys())
        ]

        # ==============================
        # MAP VIEW SELECTOR
        # ==============================

        map_type = st.radio(
            "Select Map View",
            ["City Risk Map", "Risk Hotspot Map"],
            horizontal=True
        )

        # ==============================
        # BUILD MAP DATA
        # ==============================

        map_data = []

        for _, row in city_summary.iterrows():

            city_upper = row["city_upper"]

            coords = CITY_COORDS.get(city_upper)

            if coords:

                lat, lon = coords

                total_ship = int(row["Total_Shipments"])
                high_ship = int(row["High_Risk_Shipments"])
                medium_ship = int(row["Medium_Risk_Shipments"])
                low_ship = int(row["Low_Risk_Shipments"])

                map_data.append({
                    "city": city_upper,
                    "lat": lat,
                    "lon": lon,

                    "risk": max(high_ship, 1),

                    "bubble_size": 18,

                    "Total_Shipments": total_ship,
                    "High_Risk_Shipments": high_ship,
                    "Medium_Risk_Shipments": medium_ship,
                    "Low_Risk_Shipments": low_ship
                })

        # ==============================
        # CREATE DATAFRAME
        # ==============================

        if map_data:

            map_df = pd.DataFrame(map_data)

            # LOG SCALE COLOR
            map_df["risk_log"] = np.log1p(map_df["risk"])

            # ==============================
            # CITY RISK BUBBLE MAP
            # ==============================

            fig_map = px.scatter_mapbox(
                map_df,
                lat="lat",
                lon="lon",
                size="bubble_size",
                size_max=18,
                color="risk_log",

                hover_name="city",

                # TOOLTIP CONTROL (ONLY SHOW THESE)
                hover_data={
                    "lat": False,
                    "lon": False,
                    "risk": False,
                    "risk_log": False,
                    "bubble_size": False,
                    "Total_Shipments": True,
                    "High_Risk_Shipments": True,
                    "Medium_Risk_Shipments": True,
                    "Low_Risk_Shipments": True
                },

                # Stronger color palette
                color_continuous_scale=[
                    "#FFE5E5",
                    "#FF9999",
                    "#FF4D4D",
                    "#E60000",
                    "#990000"
                ],

                zoom=4,
                height=700
            )

            fig_map.update_traces(
                marker=dict(opacity=0.95)
            )

            fig_map.update_layout(
                mapbox_style="carto-positron",
                mapbox=dict(
                    center=dict(lat=22.5, lon=78.9),
                    zoom=4
                ),
                margin={"r":0,"t":0,"l":0,"b":0}
            )

            # ==============================
            # RISK HOTSPOT HEATMAP
            # ==============================

            import plotly.graph_objects as go

            fig_hotspot = go.Figure(go.Densitymapbox(
                lat=map_df["lat"],
                lon=map_df["lon"],
                z=map_df["High_Risk_Shipments"],
                radius=40,
                colorscale="Reds",
                showscale=True
            ))

            fig_hotspot.update_layout(
                mapbox_style="carto-positron",
                mapbox_zoom=4,
                mapbox_center={"lat": 22.5, "lon": 78.9},
                height=700,
                margin={"r":0,"t":0,"l":0,"b":0}
            )

            # ==============================
            # MAP SWITCH
            # ==============================

            if map_type == "City Risk Map":

                st.plotly_chart(fig_map, use_container_width=True)

            else:

                st.plotly_chart(fig_hotspot, use_container_width=True)

        else:

            st.warning("No city coordinates matched. Check CITY_COORDS dictionary.")
# ==============================
# TAB 2
# ==============================

    with tab2:

        st.subheader("🔥 Top 10 High Risk Shipments")

        top10=filtered_df.sort_values("Failure_Risk_%",ascending=False).head(10)

        st.dataframe(top10,use_container_width=True)

        st.subheader("📋 Shipment Preview")

        st.dataframe(filtered_df.head(100),use_container_width=True)

        csv=filtered_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇ Download Filtered Data",
            data=csv,
            file_name=f"filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

        # ==============================
        # MODEL PERFORMANCE SECTION
        # ==============================

        st.divider()
        st.subheader("📊 Model Performance")

        try:

            # ==========================================
            # NEW PROFESSIONAL METHOD (LOAD TRAIN METRICS)
            # ==========================================

            metrics_path = "models/model_metrics.pkl"

            if os.path.exists(metrics_path):

                metrics = joblib.load(metrics_path)

                col1,col2,col3,col4,col5 = st.columns(5)

                col1.metric(
                    "Accuracy",
                    f"{metrics['accuracy']:.2f}",
                    help="Accuracy shows how often the model makes correct predictions overall."
                )

                col2.metric(
                    "Precision",
                    f"{metrics['precision']:.2f}",
                    help="Precision means: out of all shipments predicted as risky, how many were actually risky."
                )

                col3.metric(
                    "Recall",
                    f"{metrics['recall']:.2f}",
                    help="Recall means: out of all truly risky shipments, how many the model successfully detected."
                )

                col4.metric(
                    "F1 Score",
                    f"{metrics['f1']:.2f}",
                    help="F1 Score balances Precision and Recall. Higher value means the model performs well overall."
                )

                col5.metric(
                    "ROC AUC",
                    f"{metrics['roc_auc']:.2f}",
                    help="ROC AUC shows how well the model separates risky shipments from safe ones. Closer to 1 is better."
                )

                st.caption("Model performance based on training test dataset.")

            else:

                # ==========================================
                # FALLBACK METHOD (YOUR ORIGINAL CODE)
                # ==========================================

                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                from sklearn.metrics import confusion_matrix, roc_curve, auc

                if "target_binary" in filtered_df.columns:

                    # Prepare model inputs
                    X_eval = preprocess(filtered_df, label_encoders, model)
                    y_true = filtered_df["target_binary"]

                    y_prob = model.predict_proba(X_eval)[:,1]
                    y_pred = (y_prob >= threshold).astype(int)

                    # ==============================
                    # METRICS
                    # ==============================

                    accuracy = accuracy_score(y_true, y_pred)
                    precision = precision_score(y_true, y_pred)
                    recall = recall_score(y_true, y_pred)
                    f1 = f1_score(y_true, y_pred)

                    col1,col2,col3,col4=st.columns(4)

                    col1.metric(
                        "Accuracy",
                        f"{accuracy:.2f}",
                        help="Accuracy shows how often the model makes correct predictions overall."
                    )

                    col2.metric(
                        "Precision",
                        f"{precision:.2f}",
                        help="Precision means: out of all shipments predicted as risky, how many were actually risky."
                    )

                    col3.metric(
                        "Recall",
                        f"{recall:.2f}",
                        help="Recall means: out of all truly risky shipments, how many the model successfully detected."
                    )

                    col4.metric(
                        "F1 Score",
                        f"{f1:.2f}",
                        help="F1 Score balances Precision and Recall. Higher value means the model performs well overall."
                    )

                    # ==============================
                    # ROC CURVE
                    # ==============================

                    st.subheader("📈 ROC Curve")

                    fpr,tpr,_=roc_curve(y_true,y_prob)
                    roc_auc=auc(fpr,tpr)

                    roc_df=pd.DataFrame({
                        "False Positive Rate":fpr,
                        "True Positive Rate":tpr
                    })

                    fig_roc=px.line(
                        roc_df,
                        x="False Positive Rate",
                        y="True Positive Rate",
                        title=f"ROC Curve (AUC = {roc_auc:.2f})"
                    )

                    fig_roc.add_scatter(
                        x=[0,1],
                        y=[0,1],
                        mode="lines",
                        name="Random Model"
                    )

                    st.plotly_chart(fig_roc,use_container_width=True)

                    # ==============================
                    # CONFUSION MATRIX
                    # ==============================

                    st.subheader("📉 Confusion Matrix")

                    cm=confusion_matrix(y_true,y_pred)

                    cm_df=pd.DataFrame(
                        cm,
                        index=["Actual Safe","Actual Breach"],
                        columns=["Predicted Safe","Predicted Breach"]
                    )

                    fig_cm=px.imshow(
                        cm_df,
                        text_auto=True,
                        color_continuous_scale="Blues",
                        title="Confusion Matrix"
                    )

                    st.plotly_chart(fig_cm,use_container_width=True)

                else:

                    st.info("Model performance unavailable. 'target_binary' column not found in dataset.")

        except Exception as e:

            st.warning("Model evaluation unavailable.")
            st.text(str(e))
# ==============================
# TAB 3
# ==============================

    with tab3:

        st.subheader("⏳ Live Commitment Breach Control Tower")

        # ==============================
        # BUTTON STYLE CSS
        # ==============================

        st.markdown("""
        <style>
        div.stButton > button {
            width: 100%;
            border-radius: 10px;
            height: 48px;
            font-weight: 600;
            font-size: 15px;
        }
        </style>
        """, unsafe_allow_html=True)

        # Manual refresh button
        colA, colB = st.columns([1,5])

        with colA:
            if st.button("🔄 Refresh Now"):
                st.rerun()

        try:

            filtered_df["IST_svc_commit_tmstp"] = pd.to_datetime(
                filtered_df["IST_svc_commit_tmstp"], errors="coerce"
            )

            current_time = pd.Timestamp.now()

            filtered_df["Time_Remaining"] = (
                filtered_df["IST_svc_commit_tmstp"] - current_time
            )

            filtered_df["Remaining_Seconds"] = filtered_df["Time_Remaining"].dt.total_seconds()

            sla_df = filtered_df.copy()

            # ==============================
            # CREATE STATUS COLUMN
            # ==============================

            status_list = []

            for _, row in sla_df.iterrows():

                sec = row["Remaining_Seconds"]
                risk = row["Failure_Risk_%"]

                delivery_status = str(row.get("commit_status", "")).lower()
                last_scan = str(row.get("last_scan", "")).upper()

                delivered_codes = ["POD", "DEX04", "DDEX16"]

                if "deliver" in delivery_status or last_scan in delivered_codes:
                    status_list.append("Delivered")

                elif pd.isna(sec):
                    status_list.append("Unknown")

                elif sec < 0:
                    status_list.append("Breached")

                elif sec < 2 * 3600:
                    status_list.append("Critical")

                elif sec < 6 * 3600 and risk >= 70:
                    status_list.append("Critical")

                elif sec < 6 * 3600:
                    status_list.append("Warning")

                elif risk >= 90:
                    status_list.append("Warning")

                else:
                    status_list.append("Safe")

            sla_df["SLA_Status"] = status_list

            # ==============================
            # AI PRIORITY SCORE
            # ==============================

            sla_df["SLA_Urgency"] = sla_df["Remaining_Seconds"].apply(
                lambda x: 1 if x < 0 else 1 / (x/3600 + 1)
            )

            sla_df["Priority_Score"] = sla_df["Failure_Risk_%"] * sla_df["SLA_Urgency"]

            sla_df = sla_df.sort_values("Priority_Score", ascending=False)

            # ==============================
            # CONTROL TOWER STATUS BAR
            # ==============================

            breach_count = (sla_df["SLA_Status"]=="Breached").sum()
            critical_count = (sla_df["SLA_Status"]=="Critical").sum()
            warning_count = (sla_df["SLA_Status"]=="Warning").sum()
            safe_count = (sla_df["SLA_Status"]=="Safe").sum()
            delivered_count = (sla_df["SLA_Status"]=="Delivered").sum()

            st.markdown(
                f"""
                <div style="background:#111827;padding:10px;border-radius:8px;font-size:16px">
                🚨 <b>{breach_count}</b> Breached &nbsp;&nbsp; |
                🔴 <b>{critical_count}</b> Critical &nbsp;&nbsp; |
                🟠 <b>{warning_count}</b> Warning &nbsp;&nbsp; |
                🟢 <b>{safe_count}</b> Safe &nbsp;&nbsp; |
                📦 <b>{delivered_count}</b> Delivered
                </div>
                """,
                unsafe_allow_html=True
            )

            # ==============================
            # FILTER BUTTONS (OLD CODE - KEPT)
            # ==============================

            st.subheader("🎛 Shipment Filters")

            statuses = ["Breached","Critical","Warning","Safe","Delivered"]

            # Initial state → all inactive
            if "status_filter_state" not in st.session_state:
                st.session_state.status_filter_state = {
                    "Breached": False,
                    "Critical": False,
                    "Warning": False,
                    "Safe": False,
                    "Delivered": False
                }

            # ==============================
            # NEW DROPDOWN FILTER (ADDED)
            # ==============================

            status_filter = st.multiselect(
                "Filter by Shipment Status",
                statuses,
                default=statuses
            )

            if len(status_filter) > 0:
                sla_df = sla_df[sla_df["SLA_Status"].isin(status_filter)]
            else:
                sla_df = sla_df.iloc[0:0]

              # ==============================
            # CARD DISPLAY LIMIT
            # ==============================

            card_limit = st.slider(
                "Number of shipments to show in control room",
                min_value=1,
                max_value=100,
                value=10
            )

            # ==============================
            # URGENT SHIPMENTS (CARDS)
            # ==============================

            st.divider()
            st.subheader("🚨 Urgent Shipments (Control Room View)")

            urgent_df = sla_df.head(card_limit)

            for _, row in urgent_df.iterrows():

                shipment = row.get("Trk Nos", "Unknown")
                risk = row["Failure_Risk_%"]
                seconds_left = row["Remaining_Seconds"]
                status = row["SLA_Status"]

                # ==============================
                # COUNTDOWN (WITH NEGATIVE TIME)
                # ==============================

                if pd.isna(seconds_left):
                    countdown = "N/A"

                else:

                    sign = "-" if seconds_left < 0 else ""

                    seconds_left = abs(int(seconds_left))

                    hrs = seconds_left // 3600
                    mins = (seconds_left % 3600) // 60
                    secs = seconds_left % 60

                    countdown = f"{sign}{hrs:02}:{mins:02}:{secs:02}"

                # ==============================
                # STATUS ICONS
                # ==============================

                if status == "Breached":
                    status_icon = "🚨 Breached"
                elif status == "Critical":
                    status_icon = "🔴 Critical"
                elif status == "Warning":
                    status_icon = "🟠 Warning"
                elif status == "Safe":
                    status_icon = "🟢 Safe"
                elif status == "Delivered":
                    status_icon = "📦 Delivered"
                else:
                    status_icon = "⚪ Unknown"

                col1, col2, col3, col4 = st.columns(4)

                col1.markdown(f"**Shipment**  \n`{shipment}`")
                col2.metric("Remaining Time", countdown)
                col3.metric("Risk %", f"{risk:.2f}%")
                col4.metric("Status", status_icon)    

            # ==============================
            # FULL SHIPMENT TABLE
            # ==============================

            st.divider()
            st.subheader("📊 Full Shipment SLA Monitor")

            table_df = sla_df.copy()

            def format_time(x):

                if pd.isna(x):
                    return "N/A"

                sign = "-" if x < 0 else ""
                x = abs(int(x))

                hrs = x // 3600
                mins = (x % 3600) // 60
                secs = x % 60

                return f"{sign}{hrs:02}:{mins:02}:{secs:02}"

            table_df["Remaining Time"] = table_df["Remaining_Seconds"].apply(format_time)

            # Select required columns safely (only if they exist)
            cols = [
                "Trk Nos",
                "Remaining Time",
                "Failure_Risk_%",
                "SLA_Status",
                "Dest Loc",
                "recp_pstl_cd",
                "last_scan",
                "Last Scan Loc",
                "City name"
            ]

            cols = [c for c in cols if c in table_df.columns]

            display_table = table_df[cols].rename(
                columns={
                    "Trk Nos": "Shipment",
                    "Failure_Risk_%": "Risk %",
                    "SLA_Status": "Status"
                }
            )

            st.dataframe(display_table, use_container_width=True)

            st.write(f"Showing **{len(display_table)} shipments**")

        except Exception:
            st.info("SLA monitoring unavailable for this dataset.")
# ==============================
# TAB 4
# ==============================

    with tab4:

        st.subheader("🔎 Predict Shipment Risk Using Tracking Number")

        tracking_no = st.text_input("Enter Tracking Number")

        if st.button("Fetch Shipment & Predict Risk"):

            shipment_row = df[df["Trk Nos"] == tracking_no]

            if shipment_row.empty:
                st.error("Tracking number not found")

            else:

                st.success("Shipment Found")

                st.subheader("📦 Shipment Details")

                st.dataframe(shipment_row, use_container_width=True)

                X_single = preprocess(shipment_row, label_encoders, model)

                prob = model.predict_proba(X_single)[:,1][0]

                risk_pct = round(prob * 100, 2)

                category = categorize_risk(risk_pct)

                st.subheader("🚨 Prediction Result")

                col1, col2 = st.columns(2)

                col1.metric("Failure Risk %", f"{risk_pct:.2f}%")
                col2.metric("Risk Category", category)

                confidence = int(abs(risk_pct - 50) * 2)

                st.subheader("📊 Model Confidence Level")

                st.progress(confidence)

                st.write(f"Confidence Score: {confidence}/100")

                # ==============================
                # NEW ADDITION: SLA STATUS CHECK
                # ==============================

                try:

                    st.divider()
                    st.subheader("⏱ SLA Status")

                    if "IST_svc_commit_tmstp" in shipment_row.columns:

                        commit_time = pd.to_datetime(
                            shipment_row["IST_svc_commit_tmstp"].values[0],
                            errors="coerce"
                        )

                        current_time = pd.Timestamp.now()

                        if pd.notna(commit_time):

                            remaining_seconds = (commit_time - current_time).total_seconds()

                        else:

                            remaining_seconds = None

                    else:

                        remaining_seconds = None

                    delivery_status = str(shipment_row.get("commit_status", "").values[0]).lower() if "commit_status" in shipment_row.columns else ""
                    last_scan = str(shipment_row.get("last_scan", "").values[0]).upper() if "last_scan" in shipment_row.columns else ""

                    delivered_codes = ["POD", "DEX04", "DDEX16"]

                    if "deliver" in delivery_status or last_scan in delivered_codes:
                        sla_status = "Delivered"

                    elif remaining_seconds is None:
                        sla_status = "Unknown"

                    elif remaining_seconds < 0:
                        sla_status = "Breached"

                    elif remaining_seconds < 2 * 3600:
                        sla_status = "Critical"

                    elif remaining_seconds < 6 * 3600 and risk_pct >= 70:
                        sla_status = "Critical"

                    elif remaining_seconds < 6 * 3600:
                        sla_status = "Warning"

                    elif risk_pct >= 90:
                        sla_status = "Warning"

                    else:
                        sla_status = "Safe"

                    st.metric("Current SLA Status", sla_status)

                except Exception as e:

                    st.warning("SLA status unavailable.")
                    st.text(str(e))


                # ==============================
                # AI ROOT CAUSE PREDICTION (UPDATED)
                # ==============================

                try:

                    st.divider()
                    st.subheader("🧠 AI Root Cause Prediction")

                    st.caption("These operational factors contributed most to the predicted SLA breach risk.")

                    feature_names = {
                        "shp_pce_qty": "Shipment Volume",
                        "recp_pstl_cd": "Destination Postal Code",
                        "Dest Loc": "Destination Hub",
                        "Time_Remaining_Commit": "Remaining SLA Time",
                        "Trk Nos": "Tracking Number",
                        "last_scan": "Last Scan Status",
                        "last_scan_loc": "Last Scan Location",
                        "cntry_cd": "Country Code",
                        "Emp Nos": "Employee Count"
                    }

                    background_sample = filtered_df.sample(
                        min(100, len(filtered_df))
                    )

                    background = preprocess(
                        background_sample,
                        label_encoders,
                        model
                    )

                    explainer = shap.Explainer(model, background)

                    shap_values = explainer(X_single)

                    impact_values = shap_values.values[0]

                    feature_impact = pd.DataFrame({
                        "Feature": X_single.columns,
                        "Impact": impact_values
                    })

                    exclude_features = ["cntry_cd", "Trk Nos"]

                    feature_impact = feature_impact[
                        ~feature_impact["Feature"].isin(exclude_features)
                    ]

                    feature_impact["AbsImpact"] = feature_impact["Impact"].abs()

                    top_drivers = feature_impact.sort_values(
                        "AbsImpact",
                        ascending=False
                    ).head(5)

                    st.write("### 🔍 Top Risk Drivers")

                    for _, row in top_drivers.iterrows():

                        feature = feature_names.get(row["Feature"], row["Feature"])
                        impact = row["Impact"]

                        direction = "↑ increases risk" if impact > 0 else "↓ reduces risk"

                        st.write(
                            f"• **{feature}** → Impact Score {impact:.4f} ({direction})"
                        )

                    st.subheader("📊 Risk Driver Visualization")

                    top_drivers["Feature"] = top_drivers["Feature"].apply(
                        lambda x: feature_names.get(x, x)
                    )

                    top_drivers["Impact"] = top_drivers["Impact"] * 100

                    fig_root = px.bar(
                        top_drivers,
                        x="Impact",
                        y="Feature",
                        orientation="h",
                        color="Impact",
                        color_continuous_scale="Reds",
                        title="Top Factors Driving Shipment Risk",
                        height=400
                    )

                    fig_root.update_layout(
                        yaxis=dict(categoryorder="total ascending"),
                        xaxis_title="Impact Score (Scaled)",
                        yaxis_title="Operational Factor"
                    )

                    st.plotly_chart(fig_root, use_container_width=True)

                except Exception as e:

                    st.warning("Root cause explanation unavailable.")
                    st.text(str(e))