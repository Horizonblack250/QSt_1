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
    try:
        # Load the dataset
        df = pd.read_csv('df_new.csv')
        # Ensure timestamp is datetime
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        return df
    except FileNotFoundError:
        return None

df = load_data()

# --- Advanced Charting Function ---
def create_distribution_chart(data, column_name, title, color_hex, setpoint=None, usl=None, lsl=None, show_ucl_lcl=False):
    """
    Generates an interactive Plotly chart with Histogram, Bell Curve, and detailed Legend.
    Values are shown in the legend to prevent overlap on the chart.
    """
    mu, std = norm.fit(data)
    
    # Calculate Control Limits (3 Sigma)
    ucl = mu + 3 * std
    lcl = mu - 3 * std
    
    # 1. Base Histogram
    fig = px.histogram(
        data, 
        x=column_name, 
        nbins=40, 
        histnorm='probability density',
        title=title,
        color_discrete_sequence=[color_hex],
        opacity=0.6
    )
    
    # Calculate Y-axis max to scale the lines properly
    # (We estimate the peak of the PDF to set line heights)
    x_range = np.linspace(data.min(), data.max(), 100)
    pdf = norm.pdf(x_range, mu, std)
    y_max = max(pdf) * 1.1 # Add 10% headroom

    # 2. Add Bell Curve
    fig.add_trace(go.Scatter(
        x=x_range, y=pdf, 
        mode='lines', 
        name='Normal Dist', 
        line=dict(color='darkred', width=2)
    ))

    # --- HELPER: Function to add vertical lines to Legend ---
    def add_line(value, name, color, dash_style):
        fig.add_trace(go.Scatter(
            x=[value, value], 
            y=[0, y_max],
            mode='lines',
            name=f"{name}: {value:.2f}", # Value printed in Legend
            line=dict(color=color, width=2, dash=dash_style)
        ))

    # 3. Add Lines (Order determines legend order)
    
    # Mean (Always show)
    add_line(mu, "Mean", "black", "dash")
    
    # Setpoint
    if setpoint is not None:
        add_line(setpoint, "Setpoint", "blue", "dashdot")

    # USL / LSL
    if usl is not None and lsl is not None:
        add_line(usl, "USL", "red", "solid")
        add_line(lsl, "LSL", "red", "solid")

    # UCL / LCL (Calculated)
    if show_ucl_lcl:
        add_line(ucl, "UCL (3σ)", "purple", "dot")
        add_line(lcl, "LCL (3σ)", "purple", "dot")

    # Layout Updates
    fig.update_layout(
        bargap=0.1, 
        template="plotly_white", 
        height=450,
        legend=dict(
            orientation="v", # Vertical legend
            yanchor="top", 
            y=1, 
            xanchor="right", 
            x=1.15, # Move legend slightly outside to the right
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="Black",
            borderwidth=1
        ),
        margin=dict(r=150) # Add right margin to accommodate legend
    )
    return fig

# --- Main Dashboard Layout ---

if df is not None:
    st.title("🏭 Sunfresh Pune: Process Dashboard")
    st.markdown("### QualSteam Temperature & Pressure Control Analysis")
    st.divider()

    # --- KPI Row ---
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    # Calculate Cpk for Process Temp (Target 60, Tol +/- 1)
    sigma = df['Process Temp'].std()
    mu = df['Process Temp'].mean()
    usl_temp = 61.0
    lsl_temp = 59.0
    cpk = min((usl_temp - mu) / (3 * sigma), (mu - lsl_temp) / (3 * sigma))

    with kpi1:
        st.metric("Avg Inlet Pressure", f"{df['Inlet Steam Pressure'].mean():.2f} bar")
    with kpi2:
        st.metric("Avg Valve Opening", f"{df['QualSteam Valve Opening'].mean():.1f}%")
    with kpi3:
        st.metric("Avg Outlet Pressure", f"{df['Outlet Steam Pressure'].mean():.2f} bar")
    with kpi4:
        st.metric("Process Temp Cpk", f"{cpk:.2f}", delta="Target > 1.33")

    st.divider()

    # --- Row 1 ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Inlet Steam Pressure (P1)")
        fig1 = create_distribution_chart(
            df['Inlet Steam Pressure'], 
            'Inlet Steam Pressure', 
            'Inlet Pressure Distribution', 
            'skyblue'
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("2. QualSteam Valve Opening")
        fig2 = create_distribution_chart(
            df['QualSteam Valve Opening'], 
            'QualSteam Valve Opening', 
            'Valve Opening Distribution', 
            'lightgreen'
        )
        st.plotly_chart(fig2, use_container_width=True)

    # --- Row 2 ---
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("3. Outlet Steam Pressure")
        fig3 = create_distribution_chart(
            df['Outlet Steam Pressure'], 
            'Outlet Steam Pressure', 
            'Outlet Pressure Distribution', 
            'skyblue',
            setpoint=2.5
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("4. Process Temp Capability")
        # Spec Limits
        SP = 60.0
        USL = SP + 1
        LSL = SP - 1
        
        fig4 = create_distribution_chart(
            df['Process Temp'], 
            'Process Temp', 
            'Process Temp Capability Analysis', 
            'orange',
            setpoint=SP,
            usl=USL,
            lsl=LSL,
            show_ucl_lcl=True # Enable UCL/LCL for this plot
        )
        st.plotly_chart(fig4, use_container_width=True)

    # --- Data Table ---
    with st.expander("📝 View Raw Data"):
        st.dataframe(df.sort_values(by='Timestamp', ascending=False).head(100))

else:
    st.error("File 'df_new.csv' not found. Please upload it to your GitHub repository.")
