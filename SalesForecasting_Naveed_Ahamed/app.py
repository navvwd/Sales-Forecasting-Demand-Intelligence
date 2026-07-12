from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "train.csv"

st.set_page_config(
    page_title="Sales Forecasting & Demand Intelligence",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"train.csv was not found at {DATA_PATH}. Keep it beside app.py."
        )

    df = pd.read_csv(DATA_PATH)
    required = {
        "Order Date", "Ship Date", "Order ID", "Category",
        "Sub-Category", "Region", "Sales"
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["Order Date"] = pd.to_datetime(
        df["Order Date"], dayfirst=True, errors="coerce"
    )
    df["Ship Date"] = pd.to_datetime(
        df["Ship Date"], dayfirst=True, errors="coerce"
    )
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df = df.dropna(subset=["Order Date", "Sales"]).copy()
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Shipping Days"] = (
        df["Ship Date"] - df["Order Date"]
    ).dt.days
    return df


def monthly_sales(data: pd.DataFrame) -> pd.Series:
    return (
        data.set_index("Order Date")["Sales"]
        .resample("MS")
        .sum()
        .asfreq("MS", fill_value=0)
        .astype(float)
    )


def build_prophet_model() -> Prophet:
    return Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        interval_width=0.95,
    )


@st.cache_data(show_spinner=False)
def forecast_segment(segment_type: str, segment_value: str, horizon: int):
    df = load_data()
    column = "Category" if segment_type == "Category" else "Region"
    series = monthly_sales(df[df[column] == segment_value].copy())

    if len(series) < 18:
        raise ValueError("Not enough monthly observations for forecasting.")

    train = series.iloc[:-6]
    test = series.iloc[-6:]

    train_df = train.rename("y").reset_index()
    train_df.columns = ["ds", "y"]

    evaluation_model = build_prophet_model()
    evaluation_model.fit(train_df)
    evaluation_output = evaluation_model.predict(
        pd.DataFrame({"ds": test.index})
    )
    test_prediction = evaluation_output["yhat"].to_numpy()

    mae = mean_absolute_error(test.to_numpy(), test_prediction)
    rmse = np.sqrt(mean_squared_error(test.to_numpy(), test_prediction))

    full_df = series.rename("y").reset_index()
    full_df.columns = ["ds", "y"]
    final_model = build_prophet_model()
    final_model.fit(full_df)

    future_dates = pd.DataFrame({
        "ds": pd.date_range(
            start=series.index.max() + pd.offsets.MonthBegin(1),
            periods=horizon,
            freq="MS",
        )
    })
    output = final_model.predict(future_dates)
    forecast = output[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    forecast["yhat"] = forecast["yhat"].clip(lower=0)
    forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)

    history = series.rename("Sales").reset_index()
    history.columns = ["Month", "Sales"]
    return history, forecast, float(mae), float(rmse)


@st.cache_data(show_spinner=False)
def build_anomaly_data() -> pd.DataFrame:
    df = load_data()
    weekly = (
        df.set_index("Order Date")["Sales"]
        .resample("W")
        .sum()
        .reset_index()
        .rename(columns={"Order Date": "Week", "Sales": "Weekly Sales"})
        .sort_values("Week")
        .reset_index(drop=True)
    )

    model = IsolationForest(
        n_estimators=300,
        contamination=0.05,
        random_state=42,
    )
    weekly["Isolation Anomaly"] = (
        model.fit_predict(weekly[["Weekly Sales"]]) == -1
    )
    weekly["Rolling Mean"] = (
        weekly["Weekly Sales"].shift(1).rolling(8).mean()
    )
    weekly["Rolling Std"] = (
        weekly["Weekly Sales"].shift(1).rolling(8).std()
    )
    weekly["Rolling Z Score"] = (
        weekly["Weekly Sales"] - weekly["Rolling Mean"]
    ) / weekly["Rolling Std"]
    weekly["ZScore Anomaly"] = weekly["Rolling Z Score"].abs() > 2
    weekly["Both Methods"] = (
        weekly["Isolation Anomaly"] & weekly["ZScore Anomaly"]
    )
    weekly["Detection Result"] = np.select(
        [
            weekly["Both Methods"],
            weekly["Isolation Anomaly"],
            weekly["ZScore Anomaly"],
        ],
        ["Both Methods", "Isolation Forest Only", "Z-Score Only"],
        default="Normal",
    )
    return weekly


@st.cache_data(show_spinner=False)
def build_cluster_data():
    df = load_data()

    total_sales = df.groupby("Sub-Category")["Sales"].sum().rename("Total Sales")
    yearly = (
        df.groupby(["Sub-Category", "Year"])["Sales"]
        .sum()
        .unstack(fill_value=0)
    )
    growth = (
        yearly.replace(0, np.nan)
        .pct_change(axis=1)
        .replace([np.inf, -np.inf], np.nan)
        .mean(axis=1)
        .fillna(0)
        .mul(100)
        .rename("Average YoY Growth")
    )
    monthly = (
        df.groupby([
            "Sub-Category",
            pd.Grouper(key="Order Date", freq="MS"),
        ])["Sales"]
        .sum()
        .unstack(fill_value=0)
    )
    volatility = monthly.std(axis=1).rename("Monthly Sales Volatility")
    order_sales = (
        df.groupby(["Sub-Category", "Order ID"])["Sales"]
        .sum()
        .reset_index()
    )
    average_order = (
        order_sales.groupby("Sub-Category")["Sales"]
        .mean()
        .rename("Average Order Value")
    )

    cluster_df = pd.concat(
        [total_sales, growth, volatility, average_order], axis=1
    ).reset_index()
    cluster_df = cluster_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    features = [
        "Total Sales",
        "Average YoY Growth",
        "Monthly Sales Volatility",
        "Average Order Value",
    ]
    scaled = StandardScaler().fit_transform(cluster_df[features])
    model = KMeans(n_clusters=4, random_state=42, n_init=20)
    cluster_df["Cluster"] = model.fit_predict(scaled)

    profiles = cluster_df.groupby("Cluster")[features].mean()
    remaining = list(profiles.index)
    labels = {}

    high_value = profiles["Average Order Value"].idxmax()
    labels[high_value] = "High Value, High Volatility"
    remaining.remove(high_value)

    high_volume = profiles.loc[remaining, "Total Sales"].idxmax()
    labels[high_volume] = "High Volume, Core Demand"
    remaining.remove(high_volume)

    irregular = profiles.loc[remaining, "Average YoY Growth"].idxmax()
    labels[irregular] = "Irregular or Rebound Demand"
    remaining.remove(irregular)

    for cluster in remaining:
        labels[cluster] = "Low Volume, Moderate Growth"

    strategies = {
        "High Value, High Volatility":
            "Maintain limited safety stock, monitor demand frequently, and avoid excessive capital lock-up.",
        "High Volume, Core Demand":
            "Maintain higher safety stock, replenish frequently, and closely monitor stock-out risk.",
        "Irregular or Rebound Demand":
            "Use cautious purchasing and manually review recent demand before large stock increases.",
        "Low Volume, Moderate Growth":
            "Use lean inventory, smaller purchase quantities, and regular replenishment.",
    }

    cluster_df["Demand Segment"] = cluster_df["Cluster"].map(labels)
    cluster_df["Stocking Strategy"] = (
        cluster_df["Demand Segment"].map(strategies)
    )

    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(scaled)
    cluster_df["PCA 1"] = components[:, 0]
    cluster_df["PCA 2"] = components[:, 1]
    return cluster_df, pca.explained_variance_ratio_


def show_overview(df: pd.DataFrame):
    st.title("📊 Sales Overview Dashboard")
    st.caption("Historical sales, regional performance, category mix, and filters.")

    categories = sorted(df["Category"].dropna().unique())
    regions = sorted(df["Region"].dropna().unique())
    selected_categories = st.multiselect(
        "Filter categories", categories, default=categories
    )
    selected_regions = st.multiselect(
        "Filter regions", regions, default=regions
    )

    filtered = df[
        df["Category"].isin(selected_categories)
        & df["Region"].isin(selected_regions)
    ].copy()
    if filtered.empty:
        st.warning("The selected filters returned no records.")
        return

    total_sales = filtered["Sales"].sum()
    orders = filtered["Order ID"].nunique()
    average_order = filtered.groupby("Order ID")["Sales"].sum().mean()
    average_shipping = filtered["Shipping Days"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sales", f"{total_sales:,.2f}")
    c2.metric("Unique Orders", f"{orders:,}")
    c3.metric("Average Order Value", f"{average_order:,.2f}")
    c4.metric("Average Shipping Days", f"{average_shipping:.2f}")

    yearly = filtered.groupby("Year", as_index=False)["Sales"].sum()
    monthly = (
        filtered.set_index("Order Date")["Sales"]
        .resample("MS")
        .sum()
        .reset_index()
    )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            px.bar(yearly, x="Year", y="Sales", title="Total Sales by Year", text_auto=".3s"),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            px.line(monthly, x="Order Date", y="Sales", title="Monthly Sales Trend", markers=True),
            use_container_width=True,
        )

    category_region = (
        filtered.groupby(["Region", "Category"], as_index=False)["Sales"].sum()
    )
    st.plotly_chart(
        px.bar(
            category_region,
            x="Region",
            y="Sales",
            color="Category",
            barmode="group",
            title="Sales by Region and Category",
        ),
        use_container_width=True,
    )


def show_forecast(df: pd.DataFrame):
    st.title("🔮 Forecast Explorer")
    st.caption("Select a category or region and forecast one to three months ahead.")

    c1, c2, c3 = st.columns(3)
    with c1:
        segment_type = st.selectbox("Segment type", ["Category", "Region"])
    with c2:
        segment_value = st.selectbox(
            f"Select {segment_type.lower()}",
            sorted(df[segment_type].dropna().unique()),
        )
    with c3:
        horizon = st.slider("Forecast horizon (months)", 1, 3, 3)

    with st.spinner("Training Prophet model..."):
        history, forecast, mae, rmse = forecast_segment(
            segment_type, segment_value, horizon
        )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history["Month"], y=history["Sales"], mode="lines", name="Historical Sales"
    ))
    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat_upper"], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat_lower"], mode="lines",
        fill="tonexty", line=dict(width=0), name="95% Confidence Interval", hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat"], mode="lines+markers", name="Prophet Forecast"
    ))
    fig.update_layout(
        title=f"{segment_value}: Prophet Sales Forecast",
        xaxis_title="Month",
        yaxis_title="Sales",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    m1, m2 = st.columns(2)
    m1.metric("Holdout MAE", f"{mae:,.2f}")
    m2.metric("Holdout RMSE", f"{rmse:,.2f}")

    table = forecast.rename(columns={
        "ds": "Month",
        "yhat": "Forecasted Sales",
        "yhat_lower": "Lower Confidence Limit",
        "yhat_upper": "Upper Confidence Limit",
    })
    st.dataframe(
        table.style.format({
            "Forecasted Sales": "{:,.2f}",
            "Lower Confidence Limit": "{:,.2f}",
            "Upper Confidence Limit": "{:,.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )


def show_anomalies():
    st.title("🚨 Weekly Sales Anomaly Report")
    st.caption("Isolation Forest and rolling Z-score comparison.")

    weekly = build_anomaly_data()
    anomalies = weekly[weekly["Detection Result"] != "Normal"].copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Isolation Forest", int(weekly["Isolation Anomaly"].sum()))
    c2.metric("Rolling Z-Score", int(weekly["ZScore Anomaly"].sum()))
    c3.metric("Both Methods", int(weekly["Both Methods"].sum()))
    c4.metric("Unique Anomaly Weeks", len(anomalies))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weekly["Week"], y=weekly["Weekly Sales"], mode="lines", name="Weekly Sales"
    ))
    marker_map = {
        "Both Methods": "x",
        "Isolation Forest Only": "square",
        "Z-Score Only": "triangle-up",
    }
    for detection, marker in marker_map.items():
        points = anomalies[anomalies["Detection Result"] == detection]
        fig.add_trace(go.Scatter(
            x=points["Week"], y=points["Weekly Sales"], mode="markers",
            marker=dict(size=11, symbol=marker), name=detection
        ))
    fig.update_layout(
        title="Weekly Sales Anomaly Detection",
        xaxis_title="Week",
        yaxis_title="Weekly Sales",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    method = st.selectbox(
        "Show anomalies detected by",
        ["All Methods", "Both Methods", "Isolation Forest Only", "Z-Score Only"],
    )
    displayed = anomalies if method == "All Methods" else anomalies[anomalies["Detection Result"] == method]
    displayed = displayed[[
        "Week", "Weekly Sales", "Rolling Mean", "Rolling Z Score", "Detection Result"
    ]].sort_values("Week")
    st.dataframe(
        displayed.style.format({
            "Weekly Sales": "{:,.2f}",
            "Rolling Mean": "{:,.2f}",
            "Rolling Z Score": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )


def show_segments():
    st.title("📦 Product Demand Segments")
    st.caption("K-Means clustering using sales, growth, volatility, and order value.")

    cluster_df, variance = build_cluster_data()
    c1, c2, c3 = st.columns(3)
    c1.metric("Sub-Categories", len(cluster_df))
    c2.metric("Clusters", cluster_df["Cluster"].nunique())
    c3.metric("PCA Variance Explained", f"{variance.sum() * 100:.2f}%")

    fig = px.scatter(
        cluster_df,
        x="PCA 1",
        y="PCA 2",
        color="Demand Segment",
        text="Sub-Category",
        size="Total Sales",
        hover_data={
            "Average YoY Growth": ":.2f",
            "Monthly Sales Volatility": ":,.2f",
            "Average Order Value": ":,.2f",
            "Cluster": True,
        },
        title="Product Demand Segments by Sub-Category",
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(height=650)
    st.plotly_chart(fig, use_container_width=True)

    segments = sorted(cluster_df["Demand Segment"].unique())
    selected = st.multiselect("Filter demand segments", segments, default=segments)
    table = cluster_df[cluster_df["Demand Segment"].isin(selected)][[
        "Sub-Category", "Demand Segment", "Total Sales", "Average YoY Growth",
        "Monthly Sales Volatility", "Average Order Value", "Stocking Strategy"
    ]].sort_values(["Demand Segment", "Total Sales"], ascending=[True, False])

    st.dataframe(
        table.style.format({
            "Total Sales": "{:,.2f}",
            "Average YoY Growth": "{:.2f}%",
            "Monthly Sales Volatility": "{:,.2f}",
            "Average Order Value": "{:,.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )


def main():
    try:
        df = load_data()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    with st.sidebar:
        st.title("Sales Intelligence")
        page = st.radio(
            "Navigate",
            ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Demand Segments"],
        )
        st.divider()
        st.caption(
            f"Dataset period: {df['Order Date'].min():%b %Y} to {df['Order Date'].max():%b %Y}"
        )
        st.caption(f"Rows: {len(df):,}")

    if page == "Sales Overview":
        show_overview(df)
    elif page == "Forecast Explorer":
        show_forecast(df)
    elif page == "Anomaly Report":
        show_anomalies()
    else:
        show_segments()


if __name__ == "__main__":
    main()
