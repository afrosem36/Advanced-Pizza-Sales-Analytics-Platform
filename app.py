# =========================================================
# ADVANCED PIZZA SALES ANALYTICS PLATFORM
# =========================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import io
import requests
import os

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Pizza Sales Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)
px.defaults.template = "plotly_dark"

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .insight-box {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🍕 Advanced Pizza Sales Analytics Platform")
st.markdown("### Comprehensive Business Intelligence & Predictive Insights")

# ---------------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------------
st.subheader("📁 Data Input")

GITHUB_SAMPLE_URL = (
    "https://raw.githubusercontent.com/"
    "afrosem36/Advanced-Pizza-Sales-Analytics-Platform/"
    "main/Sample%20Data1.xlsx"
)

sample_data = None
try:
    response = requests.get(GITHUB_SAMPLE_URL)
    response.raise_for_status()
    sample_data = response.content
except:
    sample_data = b""

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Step 1️⃣: Download Sample File")
    st.markdown("Click the button below to download the sample Excel file.")
    if sample_data:
        st.download_button(
            label="📥 Download Sample Excel",
            data=sample_data,
            file_name="Sample_Pizza_Sales.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.error("Could not fetch sample file.")

with col2:
    st.markdown("### Step 2️⃣: Upload File")
    st.markdown("Upload the downloaded file (or your own CSV/Excel).")

st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload your pizza sales file (CSV / Excel)",
    type=["csv", "xlsx"]
)

if uploaded_file is None:
    st.info("👆 Please upload a file to begin analysis")
    st.stop()

df = None
if uploaded_file.name.endswith(".csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

st.success("✅ File uploaded successfully! Analysis below...")

# ---------------------------------------------------------
# LOAD & PREP DATA
# ---------------------------------------------------------
@st.cache_data
def prep_data(df):
    df.columns = df.columns.str.strip().str.lower()
    
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    
    for col in ["quantity", "unit_price", "total_price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df["year"] = df["order_date"].dt.year.astype(str)
    df["month"] = df["order_date"].dt.month
    df["month_name"] = df["order_date"].dt.month_name()
    df["day_name"] = df["order_date"].dt.day_name()
    df["hour"] = df["order_date"].dt.hour
    df["quarter"] = df["order_date"].dt.quarter
    df["week"] = df["order_date"].dt.isocalendar().week
    df["is_weekend"] = df["day_name"].isin(["Saturday", "Sunday"])
    
    return df

df = prep_data(df)

# ---------------------------------------------------------
# SIDEBAR FILTERS & ANALYSIS SELECTOR
# ---------------------------------------------------------
st.sidebar.header("🎛️ Filters")

years = sorted(df["year"].unique())
selected_years = st.sidebar.multiselect(
    "Select Year(s)",
    options=years,
    default=years
)

categories = sorted(df["pizza_category"].unique())
selected_categories = st.sidebar.multiselect(
    "Select Category",
    options=categories,
    default=categories
)

if "pizza_size" in df.columns:
    sizes = sorted(df["pizza_size"].unique())
    selected_sizes = st.sidebar.multiselect(
        "Select Size",
        options=sizes,
        default=sizes
    )
else:
    selected_sizes = None

filtered_df = df[
    (df["year"].isin(selected_years)) &
    (df["pizza_category"].isin(selected_categories))
]

if selected_sizes and "pizza_size" in df.columns:
    filtered_df = filtered_df[filtered_df["pizza_size"].isin(selected_sizes)]

st.sidebar.divider()
st.sidebar.header("📊 Analysis Sections")
show_overview = st.sidebar.checkbox("Overview Dashboard", value=True)
show_time = st.sidebar.checkbox("Time Series Analysis", value=True)
show_product = st.sidebar.checkbox("Product Performance", value=True)
show_customer = st.sidebar.checkbox("Customer Behavior", value=True)
show_advanced = st.sidebar.checkbox("Advanced Analytics", value=True)
show_predictions = st.sidebar.checkbox("Trends & Forecasts", value=True)

# ---------------------------------------------------------
# OVERVIEW DASHBOARD
# ---------------------------------------------------------
if show_overview:
    st.header("📊 Overview Dashboard")
    
    total_revenue = filtered_df["total_price"].sum()
    total_orders = filtered_df["order_id"].nunique()
    total_pizzas = filtered_df["quantity"].sum()
    avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
    avg_pizzas_per_order = total_pizzas / total_orders if total_orders > 0 else 0
    unique_pizzas = filtered_df["pizza_name"].nunique()
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("💰 Total Revenue", f"${total_revenue:,.0f}")
    with col2:
        st.metric("📦 Total Orders", f"{total_orders:,}")
    with col3:
        st.metric("🍕 Pizzas Sold", f"{total_pizzas:,}")
    with col4:
        st.metric("💵 Avg Order Value", f"${avg_order_value:.2f}")
    with col5:
        st.metric("📊 Avg Pizzas/Order", f"{avg_pizzas_per_order:.2f}")
    with col6:
        st.metric("🎯 Unique Pizzas", f"{unique_pizzas}")
    
    st.divider()
    
    # Revenue distribution charts
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("🍕 Revenue by Category")
        category_revenue = (
            filtered_df.groupby("pizza_category", as_index=False)["total_price"]
            .sum()
            .sort_values("total_price", ascending=False)
        )
        fig_pie = px.pie(
            category_revenue,
            values="total_price",
            names="pizza_category",
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Plasma
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_right:
        st.subheader("📅 Weekday vs Weekend Sales")
        weekend_sales = (
            filtered_df.groupby("is_weekend", as_index=False)["total_price"]
            .sum()
        )
        weekend_sales["type"] = weekend_sales["is_weekend"].map({True: "Weekend", False: "Weekday"})
        fig_weekend = px.bar(
            weekend_sales,
            x="type",
            y="total_price",
            color="type",
            text_auto=".2s"
        )
        st.plotly_chart(fig_weekend, use_container_width=True)
    
    st.divider()

# ---------------------------------------------------------
# TIME SERIES ANALYSIS
# ---------------------------------------------------------
if show_time:
    st.header("⏰ Time Series Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📅 Daily", "📊 Weekly", "📆 Monthly", "🕐 Hourly"])
    
    with tab1:
        st.subheader("Daily Sales Trend")
        daily_sales = (
            filtered_df.groupby("order_date", as_index=False)
            .agg({"total_price": "sum", "order_id": "nunique", "quantity": "sum"})
        )
        fig_daily = px.line(
            daily_sales,
            x="order_date",
            y="total_price",
            markers=True,
            title="Daily Revenue Over Time"
        )
        fig_daily.add_scatter(
            x=daily_sales["order_date"],
            y=daily_sales["total_price"].rolling(window=7).mean(),
            mode='lines',
            name='7-Day Moving Avg',
            line=dict(color='red', dash='dash')
        )
        st.plotly_chart(fig_daily, use_container_width=True)
    
    with tab2:
        st.subheader("Weekly Performance")
        weekly_sales = (
            filtered_df.groupby(["year", "week"], as_index=False)["total_price"]
            .sum()
        )
        weekly_sales["year_week"] = weekly_sales["year"] + "-W" + weekly_sales["week"].astype(str)
        fig_weekly = px.bar(
            weekly_sales,
            x="year_week",
            y="total_price",
            color="year",
            title="Weekly Revenue Comparison"
        )
        st.plotly_chart(fig_weekly, use_container_width=True)
    
    with tab3:
        st.subheader("Monthly Trends")
        monthly_revenue = (
            filtered_df.groupby(["year", "month", "month_name"], as_index=False)["total_price"]
            .sum()
            .sort_values(["year", "month"])
        )
        fig_monthly = px.line(
            monthly_revenue,
            x="month_name",
            y="total_price",
            color="year",
            markers=True,
            title="Monthly Revenue by Year"
        )
        st.plotly_chart(fig_monthly, use_container_width=True)
        
        # Quarter comparison
        st.subheader("Quarterly Performance")
        quarterly = (
            filtered_df.groupby(["year", "quarter"], as_index=False)["total_price"]
            .sum()
        )
        quarterly["quarter_label"] = "Q" + quarterly["quarter"].astype(str)
        fig_quarter = px.bar(
            quarterly,
            x="quarter_label",
            y="total_price",
            color="year",
            barmode="group",
            text_auto=".2s"
        )
        st.plotly_chart(fig_quarter, use_container_width=True)
    
    with tab4:
        st.subheader("Hourly Sales Heatmap")
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        hourly_pattern = (
            filtered_df.groupby(["day_name", "hour"], as_index=False)["total_price"]
            .sum()
        )
        heatmap_data = hourly_pattern.pivot(index="day_name", columns="hour", values="total_price")
        heatmap_data = heatmap_data.reindex(day_order)
        
        fig_heatmap = px.imshow(
            heatmap_data,
            labels=dict(x="Hour of Day", y="Day of Week", color="Revenue"),
            aspect="auto",
            color_continuous_scale="YlOrRd"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Peak hours analysis
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🔥 Peak Hours**")
            peak_hours = (
                filtered_df.groupby("hour")["total_price"]
                .sum()
                .sort_values(ascending=False)
                .head(5)
            )
            for hour, revenue in peak_hours.items():
                st.write(f"**{hour}:00** - ${revenue:,.0f}")
        
        with col2:
            st.markdown("**😴 Slowest Hours**")
            slow_hours = (
                filtered_df.groupby("hour")["total_price"]
                .sum()
                .sort_values(ascending=True)
                .head(5)
            )
            for hour, revenue in slow_hours.items():
                st.write(f"**{hour}:00** - ${revenue:,.0f}")
    
    st.divider()

# ---------------------------------------------------------
# PRODUCT PERFORMANCE
# ---------------------------------------------------------
if show_product:
    st.header("🎯 Product Performance Analysis")
    
    tab1, tab2, tab3 = st.tabs(["🏆 Top Performers", "📉 Low Performers", "💎 Product Mix"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 10 Pizzas by Revenue")
            top_pizzas = (
                filtered_df.groupby("pizza_name", as_index=False)
                .agg({"total_price": "sum", "quantity": "sum", "order_id": "nunique"})
                .sort_values("total_price", ascending=False)
                .head(10)
            )
            fig_top = px.bar(
                top_pizzas,
                x="total_price",
                y="pizza_name",
                orientation="h",
                color="total_price",
                color_continuous_scale="Blues"
            )
            fig_top.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_top, use_container_width=True)
        
        with col2:
            st.subheader("Top 10 by Quantity Sold")
            top_qty = (
                filtered_df.groupby("pizza_name", as_index=False)["quantity"]
                .sum()
                .sort_values("quantity", ascending=False)
                .head(10)
            )
            fig_qty = px.bar(
                top_qty,
                x="quantity",
                y="pizza_name",
                orientation="h",
                color="quantity",
                color_continuous_scale="Greens"
            )
            fig_qty.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_qty, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Bottom 10 Pizzas by Revenue")
            bottom_pizzas = (
                filtered_df.groupby("pizza_name", as_index=False)["total_price"]
                .sum()
                .sort_values("total_price", ascending=True)
                .head(10)
            )
            fig_bottom = px.bar(
                bottom_pizzas,
                x="total_price",
                y="pizza_name",
                orientation="h",
                color="total_price",
                color_continuous_scale="Reds"
            )
            st.plotly_chart(fig_bottom, use_container_width=True)
        
        with col2:
            st.subheader("Least Sold Pizzas")
            least_sold = (
                filtered_df.groupby("pizza_name", as_index=False)["quantity"]
                .sum()
                .sort_values("quantity", ascending=True)
                .head(10)
            )
            fig_least = px.bar(
                least_sold,
                x="quantity",
                y="pizza_name",
                orientation="h",
                color="quantity",
                color_continuous_scale="Oranges"
            )
            st.plotly_chart(fig_least, use_container_width=True)
    
    with tab3:
        if "pizza_size" in filtered_df.columns:
            st.subheader("Product Mix Analysis")
            
            # Size distribution
            size_dist = (
                filtered_df.groupby("pizza_size", as_index=False)
                .agg({
                    "total_price": "sum",
                    "quantity": "sum",
                    "order_id": "nunique"
                })
            )
            
            fig_size = make_subplots(
                rows=1, cols=3,
                subplot_titles=("Revenue by Size", "Quantity by Size", "Orders by Size"),
                specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
            )
            
            fig_size.add_trace(
                go.Bar(x=size_dist["pizza_size"], y=size_dist["total_price"], 
                       marker_color='lightblue', name="Revenue"),
                row=1, col=1
            )
            fig_size.add_trace(
                go.Bar(x=size_dist["pizza_size"], y=size_dist["quantity"],
                       marker_color='lightgreen', name="Quantity"),
                row=1, col=2
            )
            fig_size.add_trace(
                go.Bar(x=size_dist["pizza_size"], y=size_dist["order_id"],
                       marker_color='lightsalmon', name="Orders"),
                row=1, col=3
            )
            
            st.plotly_chart(fig_size, use_container_width=True)
            
            # Category-Size matrix
            st.subheader("Category × Size Revenue Matrix")
            cat_size = (
                filtered_df.groupby(["pizza_category", "pizza_size"], as_index=False)["total_price"]
                .sum()
            )
            matrix = cat_size.pivot(index="pizza_category", columns="pizza_size", values="total_price")
            
            fig_matrix = px.imshow(
                matrix,
                labels=dict(x="Size", y="Category", color="Revenue"),
                aspect="auto",
                color_continuous_scale="Blues",
                text_auto=".2s"
            )
            st.plotly_chart(fig_matrix, use_container_width=True)
    
    st.divider()

# ---------------------------------------------------------
# CUSTOMER BEHAVIOR ANALYSIS
# ---------------------------------------------------------
if show_customer:
    st.header("👥 Customer Behavior Insights")
    
    # Order size distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Order Size Distribution")
        order_sizes = (
            filtered_df.groupby("order_id", as_index=False)["quantity"]
            .sum()
        )
        fig_dist = px.histogram(
            order_sizes,
            x="quantity",
            nbins=20,
            title="Number of Pizzas per Order"
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        st.subheader("Order Value Distribution")
        order_values = (
            filtered_df.groupby("order_id", as_index=False)["total_price"]
            .sum()
        )
        fig_value = px.histogram(
            order_values,
            x="total_price",
            nbins=30,
            title="Revenue Distribution per Order"
        )
        st.plotly_chart(fig_value, use_container_width=True)
    
    # Customer segments
    st.subheader("📊 Order Segmentation")
    order_summary = (
        filtered_df.groupby("order_id", as_index=False)
        .agg({"total_price": "sum", "quantity": "sum"})
    )
    
    # Define segments
    revenue_quantiles = order_summary["total_price"].quantile([0.33, 0.67])
    def segment_order(price):
        if price <= revenue_quantiles.iloc[0]:
            return "Low Value"
        elif price <= revenue_quantiles.iloc[1]:
            return "Medium Value"
        else:
            return "High Value"
    
    order_summary["segment"] = order_summary["total_price"].apply(segment_order)
    segment_counts = order_summary["segment"].value_counts()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🟢 Low Value Orders", f"{segment_counts.get('Low Value', 0):,}")
    with col2:
        st.metric("🟡 Medium Value Orders", f"{segment_counts.get('Medium Value', 0):,}")
    with col3:
        st.metric("🔴 High Value Orders", f"{segment_counts.get('High Value', 0):,}")
    
    # Segment analysis
    segment_analysis = (
        order_summary.groupby("segment", as_index=False)
        .agg({
            "total_price": ["sum", "mean"],
            "quantity": "mean"
        })
    )
    segment_analysis.columns = ["Segment", "Total Revenue", "Avg Order Value", "Avg Pizzas"]
    st.dataframe(segment_analysis, use_container_width=True)
    
    st.divider()

# ---------------------------------------------------------
# ADVANCED ANALYTICS
# ---------------------------------------------------------
if show_advanced:
    st.header("🔬 Advanced Analytics")
    
    tab1, tab2, tab3 = st.tabs(["💰 Revenue Analysis", "📈 Growth Metrics", "🎯 Performance Ratios"])
    
    with tab1:
        st.subheader("Revenue Concentration (Pareto Analysis)")
        
        # Calculate cumulative revenue
        pizza_rev = (
            filtered_df.groupby("pizza_name", as_index=False)["total_price"]
            .sum()
            .sort_values("total_price", ascending=False)
        )
        pizza_rev["cumulative_revenue"] = pizza_rev["total_price"].cumsum()
        pizza_rev["cumulative_pct"] = (pizza_rev["cumulative_revenue"] / pizza_rev["total_price"].sum()) * 100
        pizza_rev["pizza_rank"] = range(1, len(pizza_rev) + 1)
        
        fig_pareto = go.Figure()
        fig_pareto.add_trace(go.Bar(
            x=pizza_rev["pizza_rank"],
            y=pizza_rev["total_price"],
            name="Revenue",
            marker_color='lightblue'
        ))
        fig_pareto.add_trace(go.Scatter(
            x=pizza_rev["pizza_rank"],
            y=pizza_rev["cumulative_pct"],
            name="Cumulative %",
            yaxis="y2",
            line=dict(color='red', width=2)
        ))
        fig_pareto.update_layout(
            title="Pareto Chart: Revenue Concentration",
            yaxis=dict(title="Revenue"),
            yaxis2=dict(title="Cumulative %", overlaying="y", side="right"),
            hovermode='x unified'
        )
        st.plotly_chart(fig_pareto, use_container_width=True)
        
        # 80/20 insight
        top_20_pct = int(len(pizza_rev) * 0.2)
        revenue_from_top_20 = pizza_rev.head(top_20_pct)["total_price"].sum()
        pct_from_top_20 = (revenue_from_top_20 / pizza_rev["total_price"].sum()) * 100
        
        st.markdown(f"""
        <div class="insight-box">
        💡 <strong>Pareto Insight:</strong> Top 20% of pizzas ({top_20_pct} products) generate 
        <strong>{pct_from_top_20:.1f}%</strong> of total revenue
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Growth Rate Analysis")
        
        if len(selected_years) > 1:
            # Year-over-year growth
            yearly_rev = (
                filtered_df.groupby("year", as_index=False)["total_price"]
                .sum()
                .sort_values("year")
            )
            yearly_rev["yoy_growth"] = yearly_rev["total_price"].pct_change() * 100
            
            fig_growth = go.Figure()
            fig_growth.add_trace(go.Bar(
                x=yearly_rev["year"],
                y=yearly_rev["total_price"],
                name="Revenue",
                marker_color='lightblue'
            ))
            fig_growth.add_trace(go.Scatter(
                x=yearly_rev["year"],
                y=yearly_rev["yoy_growth"],
                name="YoY Growth %",
                yaxis="y2",
                mode='lines+markers',
                line=dict(color='green', width=3)
            ))
            fig_growth.update_layout(
                yaxis=dict(title="Revenue"),
                yaxis2=dict(title="Growth %", overlaying="y", side="right")
            )
            st.plotly_chart(fig_growth, use_container_width=True)
        else:
            # Month-over-month for single year
            monthly = (
                filtered_df.groupby(["month"], as_index=False)["total_price"]
                .sum()
                .sort_values("month")
            )
            monthly["mom_growth"] = monthly["total_price"].pct_change() * 100
            
            fig_mom = px.line(
                monthly,
                x="month",
                y="mom_growth",
                markers=True,
                title="Month-over-Month Growth %"
            )
            st.plotly_chart(fig_mom, use_container_width=True)
    
    with tab3:
        st.subheader("Key Performance Ratios")
        
        # Calculate various ratios
        avg_unit_price = filtered_df["unit_price"].mean()
        revenue_per_pizza = filtered_df["total_price"].sum() / filtered_df["quantity"].sum()
        order_frequency = len(filtered_df) / filtered_df["order_id"].nunique()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💵 Avg Unit Price", f"${avg_unit_price:.2f}")
        with col2:
            st.metric("📊 Revenue/Pizza", f"${revenue_per_pizza:.2f}")
        with col3:
            st.metric("🔢 Items/Order", f"{order_frequency:.2f}")
        with col4:
            if "pizza_size" in filtered_df.columns:
                most_popular_size = filtered_df["pizza_size"].mode()[0]
                st.metric("⭐ Popular Size", most_popular_size)
        
        # Category performance comparison
        st.subheader("Category Performance Comparison")
        cat_metrics = (
            filtered_df.groupby("pizza_category", as_index=False)
            .agg({
                "total_price": ["sum", "mean"],
                "quantity": "sum",
                "order_id": "nunique"
            })
        )
        cat_metrics.columns = ["Category", "Total Revenue", "Avg Order Value", "Units Sold", "Num Orders"]
        cat_metrics["Revenue per Unit"] = cat_metrics["Total Revenue"] / cat_metrics["Units Sold"]
        
        st.dataframe(cat_metrics, use_container_width=True)
    
    st.divider()

# ---------------------------------------------------------
# PREDICTIONS & TRENDS
# ---------------------------------------------------------
if show_predictions:
    st.header("🔮 Trends & Forecasting")
    
    st.subheader("📈 Sales Trend with Moving Averages")
    
    daily_trend = (
        filtered_df.groupby("order_date", as_index=False)["total_price"]
        .sum()
        .sort_values("order_date")
    )
    
    # Calculate moving averages
    daily_trend["MA_7"] = daily_trend["total_price"].rolling(window=7).mean()
    daily_trend["MA_30"] = daily_trend["total_price"].rolling(window=30).mean()
    
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(
        x=daily_trend["order_date"],
        y=daily_trend["total_price"],
        name="Actual",
        mode='lines',
        line=dict(color='lightblue', width=1)
    ))
    fig_forecast.add_trace(go.Scatter(
        x=daily_trend["order_date"],
        y=daily_trend["MA_7"],
        name="7-Day MA",
        line=dict(color='yellow', width=2)
    ))
    fig_forecast.add_trace(go.Scatter(
        x=daily_trend["order_date"],
        y=daily_trend["MA_30"],
        name="30-Day MA",
        line=dict(color='red', width=2)
    ))
    fig_forecast.update_layout(
        title="Revenue Trend with Moving Averages",
        hovermode='x unified'
    )
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Trend insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Recent Performance")
        last_7_days = daily_trend.tail(7)["total_price"].mean()
        last_30_days = daily_trend.tail(30)["total_price"].mean()
        overall_avg = daily_trend["total_price"].mean()
        
        st.metric("Last 7 Days Avg", f"${last_7_days:,.0f}", 
                 delta=f"{((last_7_days/overall_avg - 1) * 100):.1f}% vs overall")
        st.metric("Last 30 Days Avg", f"${last_30_days:,.0f}",
                 delta=f"{((last_30_days/overall_avg - 1) * 100):.1f}% vs overall")
    
    with col2:
        st.subheader("🎯 Seasonality Insights")
        month_avg = (
            filtered_df.groupby("month_name")["total_price"]
            .sum()
        )
        best_month = month_avg.idxmax()
        worst_month = month_avg.idxmin()
        
        st.write(f"**Best Month:** {best_month} (${month_avg.max():,.0f})")
        st.write(f"**Worst Month:** {worst_month} (${month_avg.min():,.0f})")
        
        day_avg = (
            filtered_df.groupby("day_name")["total_price"]
            .sum()
        )
        best_day = day_avg.idxmax()
        st.write(f"**Best Day:** {best_day}")
    
    st.divider()

# ---------------------------------------------------------
# DATA EXPORT & SUMMARY
# ---------------------------------------------------------
st.header("📥 Export & Summary Report")

col1, col2, col3 = st.columns(3)

with col1:
    summary_stats = pd.DataFrame({
        "Metric": [
            "Total Revenue", "Total Orders", "Total Pizzas",
            "Avg Order Value", "Avg Pizzas/Order", "Unique Products"
        ],
        "Value": [
            f"${total_revenue:,.2f}",
            f"{total_orders:,}",
            f"{total_pizzas:,}",
            f"${avg_order_value:.2f}",
            f"{avg_pizzas_per_order:.2f}",
            f"{unique_pizzas}"
        ]
    })
    
    csv_summary = summary_stats.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📊 Download Summary Stats",
        data=csv_summary,
        file_name="pizza_summary_stats.csv",
        mime="text/csv"
    )

with col2:
    # Top products report
    top_products = (
        filtered_df.groupby("pizza_name", as_index=False)
        .agg({
            "total_price": "sum",
            "quantity": "sum",
            "order_id": "nunique"
        })
        .sort_values("total_price", ascending=False)
        .head(20)
    )
    
    csv_products = top_products.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="🏆 Download Top Products",
        data=csv_products,
        file_name="top_products_report.csv",
        mime="text/csv"
    )

with col3:
    csv_filtered = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📁 Download Filtered Data",
        data=csv_filtered,
        file_name="pizza_filtered_data.csv",
        mime="text/csv"
    )

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.divider()
st.caption("🍕 Advanced Pizza Sales Analytics Platform | Comprehensive BI & Predictive Analytics | Powered by Streamlit & Plotly")
