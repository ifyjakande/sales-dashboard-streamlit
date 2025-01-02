import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
import urllib.parse
import re
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Sales Analytics Dashboard",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Auto-refresh configuration to sidebar
st.sidebar.markdown("### Auto-Refresh Settings")
if 'refresh_interval' not in st.session_state:
    st.session_state.refresh_interval = 60

refresh_interval = st.sidebar.slider(
    "Refresh Interval (seconds)",
    min_value=5,
    max_value=300,
    value=st.session_state.refresh_interval,
    step=5,
    help="Set how often the dashboard should refresh"
)
st.session_state.refresh_interval = refresh_interval

if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True

auto_refresh = st.sidebar.checkbox(
    "Enable Auto-Refresh",
    value=st.session_state.auto_refresh,
    help="Toggle automatic dashboard refresh"
)
st.session_state.auto_refresh = auto_refresh

# Refresh status indicator
refresh_placeholder = st.sidebar.empty()

# Timestamp placeholder
timestamp_placeholder = st.sidebar.empty()

# Manual refresh button
if st.sidebar.button("Refresh Now"):
    st.experimental_rerun()

# Custom CSS with modern design and refresh indicator
st.markdown("""
<style>
    /* Main Layout */
    .main {
        padding: 2rem;
        background-color: #f8f9fa;
    }
    
    /* Title Styling */
    .title {
        text-align: center;
        font-size: 2.8em;
        background: linear-gradient(45deg, #2E3192, #1BFFFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 30px;
        font-weight: 800;
        font-family: 'Helvetica Neue', sans-serif;
        padding: 20px 0;
    }
    
    /* Subtitle Styling */
    .subtitle {
        text-align: center;
        font-size: 1.6em;
        color: #4a4a4a;
        margin-bottom: 30px;
        font-weight: 300;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.8em;
        color: #2E3192;
        margin: 30px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid #e0e0e0;
        font-weight: 600;
    }
    
    /* Card Styling */
    .stcard {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background-color: #f1f3f4;
        padding: 20px;
    }
    
    .sidebar .sidebar-content {
        background-color: #f1f3f4;
    }
    
    /* Chart Container */
    .chart-container {
        background-color: white;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Dataframe Styling */
    .dataframe {
        font-family: 'Helvetica Neue', sans-serif;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Streamlit Elements Override */
    .stSelectbox {
        background-color: white;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    
    .stNumberInput {
        background-color: white;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    
    /* Plotly Chart Styling */
    .js-plotly-plot {
        border-radius: 15px;
        overflow: hidden;
    }
    
    /* Recommendations Section */
    .recommendations {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    
    /* Refresh Status Indicator */
    .refresh-status {
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
        text-align: center;
        font-size: 0.9em;
    }
    
    .refresh-active {
        background-color: #e3f2fd;
        color: #1976d2;
        border: 1px solid #bbdefb;
    }
    
    .refresh-inactive {
        background-color: #ffebee;
        color: #c62828;
        border: 1px solid #ffcdd2;
    }
    
    /* Last Updated Timestamp */
    .timestamp {
        font-size: 0.8em;
        color: #666;
        text-align: center;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

def update_refresh_status(active=True):
    """Update the refresh status indicator"""
    status_class = "refresh-active" if active else "refresh-inactive"
    status_text = "Auto-Refresh Active" if active else "Auto-Refresh Inactive"
    refresh_placeholder.markdown(
        f'<div class="refresh-status {status_class}">{status_text}</div>',
        unsafe_allow_html=True
    )

def update_timestamp():
    """Update the last refresh timestamp"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp_placeholder.markdown(
        f'<div class="timestamp">Last refreshed: {current_time}</div>',
        unsafe_allow_html=True
    )

def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    # Initialize chart configuration state
    if 'num_scorecards' not in st.session_state:
        st.session_state.num_scorecards = 1
    if 'num_charts' not in st.session_state:
        st.session_state.num_charts = 1
        
    # Initialize scorecard configurations
    if 'scorecard_metrics' not in st.session_state:
        st.session_state.scorecard_metrics = []
    if 'scorecard_functions' not in st.session_state:
        st.session_state.scorecard_functions = []
        
    # Initialize chart configurations
    if 'chart_x_cols' not in st.session_state:
        st.session_state.chart_x_cols = []
    if 'chart_y_cols' not in st.session_state:
        st.session_state.chart_y_cols = []
    if 'chart_agg_funcs' not in st.session_state:
        st.session_state.chart_agg_funcs = []
    if 'chart_types' not in st.session_state:
        st.session_state.chart_types = []

def save_chart_config():
    """Save current chart configuration to session state"""
    # Save number of charts/scorecards
    st.session_state.num_scorecards = num_scorecards
    st.session_state.num_charts = num_charts
    
    # Save scorecard configurations
    st.session_state.scorecard_metrics = [
        st.session_state.get(f"agg_col_{i}", "")
        for i in range(num_scorecards)
    ]
    st.session_state.scorecard_functions = [
        st.session_state.get(f"agg_func_{i}", "")
        for i in range(num_scorecards)
    ]
    
    # Save chart configurations
    st.session_state.chart_x_cols = [
        st.session_state.get(f"x_col_{i}", "")
        for i in range(num_charts)
    ]
    st.session_state.chart_y_cols = [
        st.session_state.get(f"y_col_{i}", "")
        for i in range(num_charts)
    ]
    st.session_state.chart_agg_funcs = [
        st.session_state.get(f"agg_func_chart_{i}", "")
        for i in range(num_charts)
    ]
    st.session_state.chart_types = [
        st.session_state.get(f"chart_type_{i}", "")
        for i in range(num_charts)
    ]

def connect_to_synapse():
    try:

        params = urllib.parse.quote_plus(
            'DRIVER={ODBC Driver 17 for SQL Server};'
            f'SERVER={st.secrets["server"]};'
            f'DATABASE={st.secrets["database"]};'
            f'UID={st.secrets["username"]};'
            f'PWD={st.secrets["password"]};'
            'Connection Timeout=0'
        )
        
        engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
        return engine
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def get_data():
    engine = connect_to_synapse()
    if engine:
        try:
            with engine.connect() as conn:
                query = text("SELECT * FROM dbo.sales_data")
                df = pd.read_sql(query, conn)
            return df
        except Exception as e:
            st.error(f"Data Fetch Error: {str(e)}")
            return None

def recommend_chart_columns(df):
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    time_based_columns = identify_time_based_columns(df)
    numeric_y_columns = [col for col in numeric_columns if col not in time_based_columns]
    
    recommendations = []
    recommendations.append("Chart Column Recommendations:")
    
    if time_based_columns and numeric_y_columns:
        recommendations.append("Line Chart:")
        recommendations.append("  Possible X-axis columns (datetime):")
        recommendations.extend(f"    - {col}" for col in time_based_columns)
        recommendations.append("  Possible Y-axis columns (numeric):")
        recommendations.extend(f"    - {col}" for col in numeric_y_columns)
    
    categorical_with_few_unique = [col for col in categorical_columns if df[col].nunique() <= 20]
    if categorical_with_few_unique and numeric_y_columns:
        recommendations.append("Bar/Doughnut Chart:")
        recommendations.append("  Possible X-axis columns (categorical):")
        recommendations.extend(f"    - {col}" for col in categorical_with_few_unique)
        recommendations.append("  Possible Y-axis columns (numeric):")
        recommendations.extend(f"    - {col}" for col in numeric_y_columns)
    
    return "\n".join(recommendations)

def get_pure_numeric_columns(df):
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    return [col for col in numeric_columns if not is_datetime_related(col)]

def identify_time_based_columns(df):
    time_based_columns = []
    for column in df.columns:
        if is_datetime_related(column):
            time_based_columns.append(column)
    return time_based_columns

def is_datetime_related(column_name):
    datetime_pattern = r'(year|month|day|date|time|quarter|week|hour|minute|second)'
    return bool(re.search(datetime_pattern, column_name, re.IGNORECASE))

def identify_datetime_columns(df):
    datetime_pattern = r'(year|month|day|date|time|quarter|week|hour|minute|second)'
    datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
    datetime_like_columns = [col for col in df.columns if re.search(datetime_pattern, col, re.IGNORECASE)]
    return list(set(datetime_columns + datetime_like_columns))

def aggregate_data(df, group_col, agg_col, agg_func):
    if agg_func == 'Average':
        return df.groupby(group_col)[agg_col].mean().reset_index()
    elif agg_func == 'Total':
        return df.groupby(group_col)[agg_col].sum().reset_index()
    return df

def create_scorecard(df, agg_col, agg_func):
    if agg_func == 'Total':
        value = df[agg_col].sum()
    elif agg_func == 'Average':
        value = df[agg_col].mean()
    elif agg_func == 'Count':
        value = df[agg_col].count()
    else:
        return None
    return {"label": f"{agg_func} of {agg_col}", "value": f"{value:,.2f}"}

def create_scorecard_chart(scorecards):
    fig = go.Figure()
    colors = ['#2E3192', '#1BFFFF', '#00B4D8', '#0077B6', '#023E8A']
    
    for i, scorecard in enumerate(scorecards):
        fig.add_trace(go.Indicator(
            mode="number",
            value=float(scorecard["value"].replace(",", "")),
            title={'text': scorecard["label"], 'font': {'size': 18, 'color': colors[i % len(colors)]}},
            number={'font': {'size': 36, 'color': colors[i % len(colors)]}},
            domain={'x': [0.2 * i, 0.2 * i + 0.2], 'y': [0.5, 1]}
        ))
    
    fig.update_layout(
        grid=dict(rows=1, columns=len(scorecards)),
        template="plotly_white",
        height=200,
        margin=dict(t=50, b=30, l=30, r=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def format_x_axis(fig, x_col, df):
    if df[x_col].dtype in ['int64', 'float64']:
        if df[x_col].between(1900, 2100).all():
            fig.update_xaxes(tickvals=df[x_col].unique(), ticktext=[str(int(year)) for year in df[x_col].unique()])
    elif df[x_col].dtype == 'datetime64[ns]':
        fig.update_xaxes(tickformat='%Y-%m-%d')
    return fig

def create_doughnut_chart(df, x_col, y_col, threshold):
    try:
        unique_values = df[x_col].nunique()
        if unique_values > threshold:
            raise ValueError(f"The number of unique values in '{x_col}' exceeds the threshold of {threshold}. Doughnut chart not displayed for readability.")
        
        if x_col in df.select_dtypes(include=['object']).columns and y_col in df.select_dtypes(include=['number']).columns:
            aggregated_df = df.groupby(x_col)[y_col].sum().reset_index()
            fig = go.Figure(data=[go.Pie(
                labels=aggregated_df[x_col],
                values=aggregated_df[y_col],
                hole=0.4,
                marker=dict(colors=['#2E3192', '#1BFFFF', '#00B4D8', '#0077B6', '#023E8A'])
            )])
            fig.update_layout(
                title=f"Doughnut Chart of {x_col}",
                hovermode="closest",
                template="plotly_white",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Helvetica Neue"),
                margin=dict(t=50, b=30, l=30, r=30)
            )
        else:
            raise ValueError("Doughnut chart requires a categorical x-axis and a numeric aggregation value.")
        
        return fig
    except Exception as e:
        st.error(f"Chart Error: {str(e)}")
        return None

def create_bar_chart(df, x_col, y_col):
    try:
        if df[y_col].dtype not in ['int64', 'float64']:
            raise ValueError(f"Bar chart requires a numeric y-axis. '{y_col}' is not numeric.")
        
        if x_col not in df.select_dtypes(include=['object']).columns:
            raise ValueError("Bar chart requires a categorical x-axis.")
        
        fig = px.bar(
            df,
            x=x_col,
            y=y_col,
            title=f"Bar Chart of {y_col} by {x_col}",
            hover_data=[x_col, y_col],
            color_discrete_sequence=['#2E3192']
        )
        
        fig.update_layout(
            title_font=dict(size=20),
            xaxis_title_font=dict(size=14),
            yaxis_title_font=dict(size=14),
            template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Helvetica Neue"),
            margin=dict(t=50, b=30, l=30, r=30)
        )
        
        return fig
    except ValueError as e:
        st.error(f"Chart Error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected Error: {str(e)}")
        return None

def create_line_chart(df, x_col, y_col):
    try:
        if x_col not in identify_datetime_columns(df):
            raise ValueError("Line chart requires a datetime column for the x-axis.")
        
        if y_col not in df.select_dtypes(include=['number']).columns:
            st.error(f"Chart Error: Line chart requires a numeric column for the y-axis. '{y_col}' is not numeric.")
            return None
        
        fig = px.line(
            df,
            x=x_col,
            y=y_col,
            title=f"Line Chart of {y_col} by {x_col}",
            hover_data=[x_col, y_col],
            line_shape="spline",
            color_discrete_sequence=['#2E3192']
        )
        
        fig = format_x_axis(fig, x_col, df)
        
        fig.update_layout(
            title_font=dict(size=20),
            xaxis_title_font=dict(size=14),
            yaxis_title_font=dict(size=14),
            template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Helvetica Neue"),
            margin=dict(t=50, b=30, l=30, r=30)
        )
        
        fig.update_traces(
            line=dict(width=3),
            mode='lines+markers',
            marker=dict(size=8)
        )
        
        return fig
    except ValueError as e:
        st.error(f"Chart Error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected Error: {str(e)}")
        return None

def create_chart(df, x_col, y_col, chart_type, agg_func=None):
    if agg_func:
        df = aggregate_data(df, x_col, y_col, agg_func)
    
    if chart_type == 'Bar':
        return create_bar_chart(df, x_col, y_col)
    elif chart_type == 'Line':
        return create_line_chart(df, x_col, y_col)
    elif chart_type == 'Doughnut':
        return create_doughnut_chart(df, x_col, y_col, threshold=20)
    return None

def main():
    st.markdown("<div class='title'>Sales Analytics Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Comprehensive Sales Performance Analysis</div>", unsafe_allow_html=True)
    
    # Initialize session state for configurations if not exists
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    if 'num_scorecards' not in st.session_state:
        st.session_state.num_scorecards = 1
    if 'num_charts' not in st.session_state:
        st.session_state.num_charts = 1
    if 'chart_configs' not in st.session_state:
        st.session_state.chart_configs = {}
    if 'scorecard_configs' not in st.session_state:
        st.session_state.scorecard_configs = {}
    
    # Check if it's time to refresh
    current_time = datetime.now()
    if 'last_refresh' in st.session_state:
        time_elapsed = (current_time - st.session_state.last_refresh).total_seconds()
        
        # Perform refresh if auto-refresh is enabled and interval has elapsed
        if st.session_state.auto_refresh and time_elapsed >= st.session_state.refresh_interval:
            st.session_state.last_refresh = current_time
            st.experimental_rerun()
    
    # Update refresh status indicator
    update_refresh_status(st.session_state.auto_refresh)
    
    # Update timestamp
    update_timestamp()
    
    # Load data
    df = get_data()
    if df is None:
        st.error("Failed to load data from Synapse. Please check your connection.")
        return
    
    # Display dataset preview in a card
    st.markdown("<div class='section-header'>Data Overview</div>", unsafe_allow_html=True)
    with st.container():
        st.markdown("<div class='stcard'>", unsafe_allow_html=True)
        st.dataframe(df.head())
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Display chart recommendations in a card
    st.markdown("<div class='section-header'>Chart Recommendations</div>", unsafe_allow_html=True)
    with st.container():
        st.markdown("<div class='recommendations'>", unsafe_allow_html=True)
        recommendations = recommend_chart_columns(df)
        st.text(recommendations)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Sidebar styling
    st.sidebar.markdown("""
        <style>
        .sidebar .sidebar-content {
            background-image: linear-gradient(180deg, #f1f3f4 0%, #e2e6ea 100%);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Visualization configurations with persistent state
    st.sidebar.markdown("### Dashboard Configuration")
    columns = df.columns.tolist()
    num_scorecards = st.sidebar.number_input(
        "Number of Scorecards", 
        min_value=0, 
        max_value=10, 
        value=st.session_state.num_scorecards
    )
    st.session_state.num_scorecards = num_scorecards

    num_charts = st.sidebar.number_input(
        "Number of Charts", 
        min_value=1, 
        max_value=10, 
        value=st.session_state.num_charts
    )
    st.session_state.num_charts = num_charts
    
    # Scorecard configuration with persistent state
    scorecards = []
    if num_scorecards > 0:
        st.sidebar.markdown("### Scorecard Settings")
        for i in range(num_scorecards):
            st.sidebar.markdown(f"#### Scorecard {i + 1}")
            pure_numeric_columns = get_pure_numeric_columns(df)
            
            # Get previous selections from session state
            prev_metric = st.session_state.scorecard_configs.get(f"metric_{i}", "")
            prev_func = st.session_state.scorecard_configs.get(f"func_{i}", "")
            
            agg_col = st.sidebar.selectbox(
                f"Metric {i + 1}:", 
                [""] + pure_numeric_columns,
                index=pure_numeric_columns.index(prev_metric) + 1 if prev_metric in pure_numeric_columns else 0,
                key=f"metric_{i}"
            )
            
            agg_func = st.sidebar.selectbox(
                f"Function {i + 1}:", 
                [""] + ["Total", "Average", "Count"],
                index=["", "Total", "Average", "Count"].index(prev_func) if prev_func else 0,
                key=f"func_{i}"
            )
            
            # Save selections to session state
            st.session_state.scorecard_configs[f"metric_{i}"] = agg_col
            st.session_state.scorecard_configs[f"func_{i}"] = agg_func
            
            if agg_col and agg_func:
                scorecard = create_scorecard(df, agg_col, agg_func)
                if scorecard:
                    scorecards.append(scorecard)
    
    # Chart configuration with persistent state
    st.sidebar.markdown("### Chart Settings")
    charts = []
    for i in range(num_charts):
        st.sidebar.markdown(f"#### Chart {i + 1}")
        
        # Get previous selections from session state
        prev_x = st.session_state.chart_configs.get(f"x_{i}", "")
        prev_y = st.session_state.chart_configs.get(f"y_{i}", "")
        prev_agg = st.session_state.chart_configs.get(f"agg_{i}", "")
        prev_type = st.session_state.chart_configs.get(f"type_{i}", "")
        
        x_col = st.sidebar.selectbox(
            f"X-axis {i + 1}:", 
            [""] + columns,
            index=columns.index(prev_x) + 1 if prev_x in columns else 0,
            key=f"x_{i}"
        )
        
        y_col = st.sidebar.selectbox(
            f"Y-axis {i + 1}:", 
            [""] + columns,
            index=columns.index(prev_y) + 1 if prev_y in columns else 0,
            key=f"y_{i}"
        )
        
        agg_func = st.sidebar.selectbox(
            f"Aggregation {i + 1}:", 
            [""] + ["Average", "Total"],
            index=["", "Average", "Total"].index(prev_agg) if prev_agg else 0,
            key=f"agg_{i}"
        )
        
        chart_type = st.sidebar.selectbox(
            f"Chart Type {i + 1}:", 
            [""] + ["Bar", "Line", "Doughnut"],
            index=["", "Bar", "Line", "Doughnut"].index(prev_type) if prev_type else 0,
            key=f"type_{i}"
        )
        
        # Save selections to session state
        st.session_state.chart_configs[f"x_{i}"] = x_col
        st.session_state.chart_configs[f"y_{i}"] = y_col
        st.session_state.chart_configs[f"agg_{i}"] = agg_func
        st.session_state.chart_configs[f"type_{i}"] = chart_type
        
        if x_col and y_col and chart_type:
            fig = create_chart(df, x_col, y_col, chart_type, agg_func)
            if fig:
                charts.append({"fig": fig, "type": chart_type})
    
    # Display visualizations
    if scorecards:
        st.markdown("<div class='section-header'>Key Metrics</div>", unsafe_allow_html=True)
        with st.container():
            st.markdown("<div class='stcard'>", unsafe_allow_html=True)
            scorecard_fig = create_scorecard_chart(scorecards)
            st.plotly_chart(scorecard_fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    if charts:
        st.markdown("<div class='section-header'>Data Visualization</div>", unsafe_allow_html=True)
        for i in range(0, len(charts), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j < len(charts):
                    with col:
                        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                        st.plotly_chart(charts[i + j]["fig"], use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    
    # Auto-refresh mechanism
    if st.session_state.auto_refresh:
        time.sleep(1)  # Small delay to prevent excessive CPU usage
        time_elapsed = (datetime.now() - st.session_state.last_refresh).total_seconds()
        if time_elapsed >= st.session_state.refresh_interval:
            st.experimental_rerun()