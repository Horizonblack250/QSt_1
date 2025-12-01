import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm

# --- Page Configuration ---
st.set_page_config(
    page_title="Sunfresh Pune Dashboard",
    page_icon="🥛",
    layout="wide"
)

# --- Data Loading Function ---
@st.cache_data
def load_data():
    # Try to load the file from the root directory
    try:
        df = pd.read_csv('df_new.csv')
        # Ensure timestamp is datetime if needed
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        return df
    except FileNotFoundError:
        st.error("File 'df_new.csv' not found. Please ensure it is in the repository.")
        return None

df = load_data()

# --- Helper Function for Bell Curves ---
def create_distribution_chart(data, column_name, title, color_seq, setpoint=None, usl=None, lsl=None):
    """
    Generates an interactive Plotly chart with Histogram + Bell Curve + Limits
    """
    mu, std = norm.fit(data)
    
    # 1. Histogram
    fig = px.histogram(
        data, 
        x=column_name, 
        nbins=40, 
        histnorm='probability density',
        title=title,
        color_discrete_sequence=[color_seq],
        opacity=0.7
    )

    # 2. Bell Curve (Normal Distribution)
    x_range = np.linspace(data.min(), data.max(), 100)
    pdf = norm.pdf(x_range, mu, std)
    
    fig.add_trace(go.Scatter(
        x=x_range, y=pdf, 
        mode='lines', 
        name='Normal Dist', 
        line=dict(color='red', width=2)
    ))

    # 3. Mean Line
    fig.add_vline(x=mu, line_width=2, line_dash="dash", line_color="black", annotation_text=f"Mean: {mu:.2f}")

    # 4. Setpoint (if provided)
    if setpoint is not None:
        fig.add_vline(x=setpoint, line_width=2, line_dash="dashdot", line_color="blue", annotation_text=f"SP: {setpoint}")

    # 5. USL/LSL (if provided)
    if usl is not None and lsl is not None:
        fig.add_vline(x=usl, line_width=2, line_color="red", annotation_text="USL")
        fig.add_vline(x=lsl, line_width=2, line_color="red", annotation_text="LSL")
        
        # Add Control Limits (3 Sigma)
        ucl = mu + 3 * std
        lcl = mu - 3 * std
        fig.add_vline(x=ucl, line_width=2, line_dash="dot", line_color="purple", annotation_text="UCL")
        fig.add_vline(x=lcl, line_width=2, line_dash="dot", line_color="purple", annotation_text="LCL")

    fig.update_layout(bargap=0.1, template="plotly_white", height=400)
    return fig

# --- Main Dashboard Layout ---

if df is not None:
    st.title("🏭 Sunfresh Pune: Process Dashboard")
    st.markdown("### QualSteam Temperature & Pressure Control Analysis")
    st.divider()

    # --- KPI Row (Key Performance Indicators) ---
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.metric(
            label="Avg Inlet Pressure", 
            value=f"{df['Inlet Steam Pressure'].mean():.2f} bar"
        )
    with kpi2:
        st.metric(
            label="Avg Valve Opening", 
            value=f"{df['QualSteam Valve Opening'].mean():.1f}%"
        )
    with kpi3:
        st.metric(
            label="Avg Outlet Pressure", 
            value=f"{df['Outlet Steam Pressure'].mean():.2f} bar",
            delta=f"{df['Outlet Steam Pressure'].mean() - 2.5:.2f} vs SP"
        )
    with kpi4:
        st.metric(
            label="Avg Process Temp", 
            value=f"{df['Process Temp'].mean():.2f} °C",
            delta=f"{df['Process Temp'].mean() - 60:.2f} vs SP"
        )

    st.divider()

    # --- Row 1: Input Variables ---
    st.subheader("1. Input Variables Analysis")
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        fig_p1 = create_distribution_chart(
            df['Inlet Steam Pressure'], 
            'Inlet Steam Pressure', 
            'Inlet Steam Pressure (P1) Distribution',
            'skyblue'
        )
        st.plotly_chart(fig_p1, use_container_width=True)

    with row1_col2:
        fig_valve = create_distribution_chart(
            df['QualSteam Valve Opening'], 
            'QualSteam Valve Opening', 
            'QualSteam Valve Opening Distribution',
            'lightgreen'
        )
        st.plotly_chart(fig_valve, use_container_width=True)

    # --- Row 2: Output & Control Variables ---
    st.subheader("2. Output & Control Analysis")
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        fig_outlet = create_distribution_chart(
            df['Outlet Steam Pressure'], 
            'Outlet Steam Pressure', 
            'Outlet Steam Pressure vs Setpoint (2.5)',
            'skyblue',
            setpoint=2.5
        )
        st.plotly_chart(fig_outlet, use_container_width=True)

    with row2_col2:
        # Define specifications for Process Temp
        SP_TEMP = 60.0
        USL_TEMP = SP_TEMP + 1
        LSL_TEMP = SP_TEMP - 1
        
        fig_temp = create_distribution_chart(
            df['Process Temp'], 
            'Process Temp', 
            'Process Temp Capability (SP=60, USL/LSL=±1)',
            'orange',
            setpoint=SP_TEMP,
            usl=USL_TEMP,
            lsl=LSL_TEMP
        )
        st.plotly_chart(fig_temp, use_container_width=True)

    # --- Data Preview Section ---
    with st.expander("View Raw Data"):
        st.dataframe(df)

else:
    st.warning("Please upload the dataset to the repository.")
