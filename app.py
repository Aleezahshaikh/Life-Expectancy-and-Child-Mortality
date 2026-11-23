from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# -------------------------------------------------------------------
# Page setup and data
# -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "wdi_clean_final_reference.csv"

st.set_page_config(
    page_title="Life Expectancy & Child Mortality",
    page_icon="🌍",
    layout="wide",
)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    return df


df = load_data()

required_columns = [
    "country", "year", "income_group", "region", "life_exp", "u5_mort",
    "health_pc", "gdp_pc", "water", "avg_immunization", "fertility", "code"
]
missing = [column for column in required_columns if column not in df.columns]
if missing:
    st.error(
        "The dataset is missing columns required by this dashboard: "
        + ", ".join(missing)
    )
    st.stop()

YEARS = sorted(df["year"].dropna().unique())
MIN_YEAR, MAX_YEAR = int(min(YEARS)), int(max(YEARS))

INCOME_GROUPS = [
    group for group in
    ["Low income", "Lower middle income", "Upper middle income", "High income"]
    if group in df["income_group"].dropna().unique()
]

INDICATORS = {
    "life_exp": "Life expectancy (years)",
    "u5_mort": "Under-5 mortality (deaths per 1,000 live births)",
    "health_pc": "Health spending per person (current US$)",
    "gdp_pc": "GDP per person (current US$)",
    "water": "Basic drinking-water access (%)",
    "avg_immunization": "Average DPT and measles immunisation (%)",
    "fertility": "Fertility (births per woman)",
}
if "health_gdp" in df.columns:
    INDICATORS["health_gdp"] = "Health spending (% of GDP)"
if "urban_pct" in df.columns:
    INDICATORS["urban_pct"] = "Urban population (%)"

COLORS = {
    "Low income": "#c8452f",
    "Lower middle income": "#e39a2d",
    "Upper middle income": "#2b9a8a",
    "High income": "#2a4f9e",
}




def average_by_year(data, indicator, group_column=None):
    grouping = ["year"] if group_column is None else ["year", group_column]
    return data.groupby(grouping, as_index=False)[indicator].mean()


# -------------------------------------------------------------------
# Navigation (World Map is deliberately the LAST section)
# -------------------------------------------------------------------
st.sidebar.title("Project Navigation")
page = st.sidebar.radio(
    "Choose a section",
    [
        "Overview",
        "Data Explorer",
        "Trends",
        "Relationships",
        "Model Results",
        "Transformer",
        "Country Profile",
        "Major Events",
        "World Map",
    ],
)
st.sidebar.caption("World Bank WDI | Country-year analysis | 2000–2023")


# -------------------------------------------------------------------
# Overview
# -------------------------------------------------------------------
if page == "Overview":
    st.title("Determinants of Life Expectancy and Child Mortality")
    st.write(
        "A cross-country analysis using World Bank World Development Indicators (WDI)."
    )
    st.markdown(
        "This project examines how health spending, income, access to clean water, "
        "immunisation, fertility and other development indicators are associated "
        "with life expectancy and under-5 mortality."
    )

    a, b, c = st.columns(3)
    a.metric("Country-year records", f"{len(df):,}")
    b.metric("Countries", f"{df['country'].nunique():,}")
    c.metric("Years covered", f"{MIN_YEAR}–{MAX_YEAR}")

    st.subheader("What we investigated")
    st.markdown(
        "- Which factors are associated with longer life expectancy?\n"
        "- How do child-mortality levels differ across countries and income groups?\n"
        "- How accurately can life expectancy and child-mortality risk be predicted?\n"
        "- Does a more complex Transformer model improve prediction?"
    )

    st.subheader("Main findings")
    st.markdown(
        "- Life expectancy and under-5 mortality have a strong negative relationship.\n"
        "- Income group is strongly associated with differences in life expectancy.\n"
        "- Health spending, income, water access and fertility are useful indicators to examine together.\n"
        "- Linear Regression performed slightly better than the Transformer in the reported comparison."
    )

    st.caption(
        "Source: World Bank, World Development Indicators. The analysis period "
        "ends in 2023 because several important indicators are incomplete for 2024."
    )


# -------------------------------------------------------------------
# Data Explorer
# -------------------------------------------------------------------
elif page == "Data Explorer":
    st.title("Data Explorer")
    st.write("Filter the country-year dataset and inspect the values behind the analysis.")

    countries = ["All"] + sorted(df["country"].dropna().unique().tolist())
    selected_country = st.selectbox("Country", countries)
    year_range = st.slider(
        "Year range",
        MIN_YEAR,
        MAX_YEAR,
        (MIN_YEAR, MAX_YEAR),
    )

    view = df[df["year"].between(year_range[0], year_range[1])].copy()
    if selected_country != "All":
        view = view[view["country"] == selected_country]

    st.metric("Rows in current view", f"{len(view):,}")

    columns_to_show = [
        column for column in [
            "country", "year", "region", "income_group", "life_exp", "u5_mort",
            "health_pc", "gdp_pc", "water", "avg_immunization", "fertility"
        ] if column in view.columns
    ]
    st.dataframe(
        view[columns_to_show].sort_values(["country", "year"]),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download filtered data as CSV",
        view.to_csv(index=False).encode("utf-8"),
        file_name="filtered_wdi_data.csv",
        mime="text/csv",
    )


# -------------------------------------------------------------------
# Trends: outcome and income-group filters
# -------------------------------------------------------------------
elif page == "Trends":
    st.title("Trends Over Time")
    st.write("Compare changes in health outcomes across the selected income groups.")

    left, right = st.columns(2)
    indicator = left.selectbox(
        "Choose an indicator",
        ["life_exp", "u5_mort"] + [
            key for key in INDICATORS
            if key not in ["life_exp", "u5_mort"]
        ],
        format_func=lambda key: INDICATORS[key],
    )
    selected_groups = right.multiselect(
        "Income groups",
        INCOME_GROUPS,
        default=INCOME_GROUPS,
    )

    trend_data = df[df["income_group"].isin(selected_groups)].copy()
    trend = average_by_year(trend_data, indicator, "income_group")

    fig = px.line(
        trend,
        x="year",
        y=indicator,
        color="income_group",
        markers=True,
        color_discrete_map=COLORS,
        labels={"year": "Year", indicator: INDICATORS[indicator],
                "income_group": "Income group"},
        title=f"{INDICATORS[indicator]} over time by income group",
    )
    fig.add_vrect(
        x0=2020, x1=2021,
        fillcolor="grey", opacity=0.15, line_width=0,
        annotation_text="COVID-19 period",
        annotation_position="top left",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Lines show unweighted averages of available country observations in each "
        "income group. The shaded band marks 2020–2021; it does not estimate the "
        "causal effect of COVID-19."
    )


# -------------------------------------------------------------------
# Relationships
# -------------------------------------------------------------------
elif page == "Relationships":
    st.title("Relationships Between Indicators")
    st.write("Explore how life expectancy varies alongside other indicators.")

    left, right = st.columns(2)
    choices = [key for key in INDICATORS if key != "life_exp"]
    x_choice = left.selectbox(
        "Compare life expectancy with",
        choices,
        format_func=lambda key: INDICATORS[key],
    )
    group_choice = right.selectbox(
        "Colour points by",
        ["income_group", "region"],
    )

    plot_data = df[
        ["life_exp", x_choice, group_choice, "country", "year"]
    ].dropna().copy()

    if len(plot_data) > 2500:
        plot_data = plot_data.sample(2500, random_state=42)

    fig = px.scatter(
        plot_data,
        x=x_choice,
        y="life_exp",
        color=group_choice,
        hover_name="country",
        hover_data=["year"],
        labels={
            x_choice: INDICATORS[x_choice],
            "life_exp": INDICATORS["life_exp"],
            group_choice: group_choice.replace("_", " ").title(),
        },
        title=f"Life expectancy compared with {INDICATORS[x_choice].lower()}",
        opacity=0.65,
        color_discrete_map=COLORS if group_choice == "income_group" else None,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Each dot represents a country-year observation. A random sample is used "
        "when there are many observations so the chart remains readable."
    )


# -------------------------------------------------------------------
# Model Results
# -------------------------------------------------------------------
elif page == "Model Results":
    st.title("Prediction Model Results")
    st.write(
        "The project compared regression models for life expectancy and "
        "classification models for child-mortality risk."
    )

    st.subheader("Life expectancy regression")
    reg = pd.DataFrame({
        "Model": ["Linear Regression", "Gradient Boosting", "Random Forest"],
        "Mean R²": [0.778, 0.776, 0.757],
        "Mean MAE (years)": [2.901, 2.910, 2.943],
    })
    st.dataframe(reg, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_r2 = px.bar(
            reg, x="Model", y="Mean R²", range_y=[0, 1],
            text_auto=".3f", title="Regression comparison: R²",
        )
        st.plotly_chart(fig_r2, use_container_width=True)
    with c2:
        fig_mae = px.bar(
            reg, x="Model", y="Mean MAE (years)",
            text_auto=".2f", title="Regression comparison: MAE (years)",
        )
        st.plotly_chart(fig_mae, use_container_width=True)


    st.divider()
    st.subheader("Child-mortality risk classification")
    cls = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest"],
        "Accuracy": [0.887, 0.878],
        "F1 score": [0.860, 0.854],
    })
    st.dataframe(cls, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_acc = px.bar(
            cls, x="Model", y="Accuracy", range_y=[0, 1],
            text_auto=".3f", title="Classification comparison: Accuracy",
        )
        st.plotly_chart(fig_acc, use_container_width=True)
    with c2:
        fig_f1 = px.bar(
            cls, x="Model", y="F1 score", range_y=[0, 1],
            text_auto=".3f", title="Classification comparison: F1 score",
        )
        st.plotly_chart(fig_f1, use_container_width=True)

    st.caption(
        "Values are the results recorded in the analysis notebook. They are "
        "evaluation measures, not guarantees for future data."
    )


# -------------------------------------------------------------------
# Transformer
# -------------------------------------------------------------------
elif page == "Transformer":
    st.title("Transformer Model: What We Tested")
    st.write(
        "A tabular Transformer was tested to see whether a more complex model "
        "could improve life-expectancy prediction."
    )

    transformer = pd.DataFrame({
        "Model": [
            "Linear Regression", "Gradient Boosting",
            "Random Forest", "Tabular Transformer"
        ],
        "Mean R²": [0.778, 0.776, 0.757, 0.768],
        "Mean MAE (years)": [2.901, 2.910, 2.943, 2.965],
    })
    st.dataframe(transformer, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_r2 = px.bar(
            transformer, x="Model", y="Mean R²", range_y=[0, 1],
            text_auto=".3f", title="Model comparison: R²",
        )
        st.plotly_chart(fig_r2, use_container_width=True)
    with c2:
        fig_mae = px.bar(
            transformer, x="Model", y="Mean MAE (years)",
            text_auto=".2f", title="Model comparison: MAE (years)",
        )
        st.plotly_chart(fig_mae, use_container_width=True)

    st.markdown(
        "**Simple answer if asked:** We tested the Transformer to explore a more "
        "advanced approach. It did not improve the results enough to justify "
        "using it as our main model."
    )
    st.caption(
        "Transformer result: mean R² 0.768 ± 0.044; it performed better than "
        "Linear Regression in 3 of 10 splits."
    )


# -------------------------------------------------------------------
# Country Profile (borrowed from the second dashboard)
# -------------------------------------------------------------------
elif page == "Country Profile":
    st.title("Country Profile")
    st.write(
        "Compare one country's historical trend with its income-group average "
        "and the overall country average."
    )

    left, right = st.columns(2)
    countries = sorted(df["country"].dropna().unique().tolist())
    default_country = "Pakistan" if "Pakistan" in countries else countries[0]
    country = left.selectbox(
        "Select a country",
        countries,
        index=countries.index(default_country),
    )
    indicator = right.selectbox(
        "Select an indicator",
        list(INDICATORS),
        format_func=lambda key: INDICATORS[key],
        key="profile_indicator",
    )

    country_data = df[df["country"] == country].copy()
    if country_data.empty:
        st.warning("No data is available for this country.")
    else:
        income_group = country_data["income_group"].dropna().iloc[0]
        country_line = country_data[["year", indicator]].rename(
            columns={indicator: country}
        )
        income_line = (
            df[df["income_group"] == income_group]
            .groupby("year", as_index=False)[indicator]
            .mean()
            .rename(columns={indicator: f"{income_group} average"})
        )
        overall_line = (
            df.groupby("year", as_index=False)[indicator]
            .mean()
            .rename(columns={indicator: "All countries average"})
        )

        profile = country_line.merge(income_line, on="year", how="outer")
        profile = profile.merge(overall_line, on="year", how="outer")
        profile = profile.melt(
            id_vars="year", var_name="Series", value_name="Value"
        )

        fig = px.line(
            profile,
            x="year",
            y="Value",
            color="Series",
            markers=True,
            title=f"{INDICATORS[indicator]}: {country} compared with averages",
            labels={"Value": INDICATORS[indicator], "year": "Year"},
        )
        st.plotly_chart(fig, use_container_width=True)

        latest = country_data.sort_values("year").iloc[-1]
        a, b, c = st.columns(3)
        a.metric("Income group", income_group)
        if pd.notna(latest["life_exp"]):
            b.metric("Latest life expectancy", f"{latest['life_exp']:.1f} years")
        if pd.notna(latest["u5_mort"]):
            c.metric("Latest under-5 mortality", f"{latest['u5_mort']:.1f} per 1,000")



# -------------------------------------------------------------------
# Major events timeline
# -------------------------------------------------------------------
elif page == "Major Events":
    st.title("Major World Events During the Study Period")
    st.write(
        "These events provide historical context for interpreting changes in "
        "economic and health indicators between 2000 and 2023."
    )

    events = pd.DataFrame([
        {
            "Period": "2008–2009",
            "Event": "Global financial crisis",
            "Possible relevance": (
                "Economic slowdown and pressure on public budgets may affect "
                "household welfare and the resources available for health services."
            ),
        },
        {
            "Period": "2014–2016",
            "Event": "Ebola outbreak in West Africa",
            "Possible relevance": (
                "The outbreak disrupted health services in affected countries "
                "and placed additional pressure on health systems."
            ),
        },
        {
            "Period": "2020 onward",
            "Event": "COVID-19 pandemic",
            "Possible relevance": (
                "The pandemic disrupted routine healthcare, vaccination and "
                "economic activity in many countries."
            ),
        },
        {
            "Period": "2022 onward",
            "Event": "Russia–Ukraine war",
            "Possible relevance": (
                "The war contributed to wider food and energy price pressures, "
                "which can affect household living costs and public budgets."
            ),
        },
        {
            "Period": "2023 onward",
            "Event": "Gaza conflict",
            "Possible relevance": (
                "The conflict caused severe disruption to health infrastructure "
                "and access to healthcare in Gaza."
            ),
        },
    ])

    for _, event in events.iterrows():
        with st.container(border=True):
            st.subheader(f"{event['Period']} · {event['Event']}")
            st.write(event["Possible relevance"])

    st.warning(
        "Important limitation: these events are shown as historical context. "
        "The project does not separately estimate or prove the causal effect "
        "of any event on life expectancy or child mortality."
    )
    st.caption(
        "The trend chart in this dashboard shades 2020–2021 as the COVID-19 "
        "period for visual reference. The timeline includes broader event dates."
    )


# -------------------------------------------------------------------
# World Map (deliberately the final page in the navigation)
# -------------------------------------------------------------------
elif page == "World Map":
    st.title("World Map")
    st.write(
        "Explore how health and development indicators differ across countries "
        "in a selected year."
    )

    left, right = st.columns(2)
    map_year = left.select_slider(
        "Select year",
        options=YEARS,
        value=MAX_YEAR,
    )
    map_indicator = right.selectbox(
        "Select indicator",
        list(INDICATORS),
        format_func=lambda key: INDICATORS[key],
        key="map_indicator",
    )

    map_data = df[df["year"] == map_year].dropna(
        subset=["code", map_indicator]
    ).copy()

    if map_data.empty:
        st.warning("No map data is available for these selections.")
    else:
        reverse_scale = map_indicator == "u5_mort"
        color_scale = "RdYlGn_r" if reverse_scale else "RdYlGn"

        fig = px.choropleth(
            map_data,
            locations="code",
            color=map_indicator,
            hover_name="country",
            color_continuous_scale=color_scale,
            labels={map_indicator: INDICATORS[map_indicator]},
            title=f"{INDICATORS[map_indicator]} by country ({map_year})",
        )
        fig.update_layout(height=600, margin={"r": 0, "t": 55, "l": 0, "b": 0})
        st.plotly_chart(fig, use_container_width=True)

        best = map_data.nlargest(10, map_indicator)[
            ["country", map_indicator]
        ].rename(columns={map_indicator: INDICATORS[map_indicator]})
        lowest = map_data.nsmallest(10, map_indicator)[
            ["country", map_indicator]
        ].rename(columns={map_indicator: INDICATORS[map_indicator]})

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("10 highest values")
            st.dataframe(best, use_container_width=True, hide_index=True)
        with c2:
            st.subheader("10 lowest values")
            st.dataframe(lowest, use_container_width=True, hide_index=True)

    st.caption(
        "For under-5 mortality, lower values represent better outcomes. "
        "Country coverage may vary because some indicators have missing observations."
    )
