from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "wdi_clean_final_reference.csv"

st.set_page_config(
    page_title="Life Expectancy & Child Mortality",
    page_icon="🌍",
    layout="wide"
)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    return df

df = load_data()

st.sidebar.title("Project Navigation")
page = st.sidebar.radio(
    "Choose a section",
    ["Overview", "Data Explorer", "Trends", "Relationships", "Model Results", "Transformer"]
)
st.sidebar.caption("World Bank WDI | Country-year analysis | 2000–2023")

def explain(text):
    st.info("**How to explain this:** " + text)

if page == "Overview":
    st.title("Determinants of Life Expectancy and Child Mortality")
    st.write("A cross-country analysis using World Bank World Development Indicators (WDI).")
    st.markdown(
        "This project studies how health spending, income, access to clean water, "
        "immunisation and other development indicators are associated with life expectancy "
        "and under-5 mortality."
    )
    a, b, c = st.columns(3)
    a.metric("Country-year records", f"{len(df):,}")
    b.metric("Countries", f"{df['country'].nunique():,}")
    c.metric("Years covered", f"{int(df['year'].min())}–{int(df['year'].max())}")
    st.subheader("What we investigated")
    st.markdown(
        "- Which factors are associated with longer life expectancy?\n"
        "- How do child mortality levels differ across countries and income groups?\n"
        "- How well can life expectancy and child-mortality risk be predicted?\n"
        "- Does a more complex Transformer model improve prediction?"
    )
    st.subheader("Main findings")
    st.markdown(
        "- Life expectancy and under-5 mortality have a strong negative relationship.\n"
        "- Income group is strongly associated with differences in life expectancy.\n"
        "- Health spending, income, water access and fertility are useful indicators to examine together.\n"
        "- The simpler Linear Regression model performed slightly better than the Transformer in the reported comparison."
    )
    explain("The project uses country-level observations over time. The results show associations and predictive patterns; they do not prove that one factor directly causes a health outcome.")
    st.caption("Source: World Bank, World Development Indicators. The analysis period ends in 2023 because several important indicators are incomplete for 2024.")

elif page == "Data Explorer":
    st.title("Data Explorer")
    st.write("Use the filters to view a smaller part of the dataset.")
    countries = ["All"] + sorted(df["country"].dropna().unique().tolist())
    selected_country = st.selectbox("Country", countries)
    years = st.slider("Year range", int(df.year.min()), int(df.year.max()),
                      (int(df.year.min()), int(df.year.max())))
    view = df[df.year.between(years[0], years[1])]
    if selected_country != "All":
        view = view[view.country == selected_country]
    st.metric("Rows in current view", f"{len(view):,}")
    cols = ["country", "year", "income_group", "life_exp", "u5_mort",
            "health_pc", "gdp_pc", "water", "avg_immunization"]
    st.dataframe(view[cols].sort_values(["country", "year"]), use_container_width=True, hide_index=True)
    st.download_button(
        "Download this filtered data as CSV",
        view.to_csv(index=False).encode("utf-8"),
        file_name="filtered_wdi_data.csv",
        mime="text/csv"
    )
    explain("Each row represents one country in one year. The table lets us inspect the original values behind the charts.")

elif page == "Trends":
    st.title("Trends Over Time")
    st.write("The charts show average outcomes across the countries available in each year.")
    trend = df.groupby("year", as_index=False).agg(
        life_exp=("life_exp", "mean"),
        u5_mort=("u5_mort", "mean")
    )
    metric = st.radio("Choose an outcome", ["Life expectancy", "Under-5 mortality"], horizontal=True)
    if metric == "Life expectancy":
        fig = px.line(trend, x="year", y="life_exp", markers=True,
                      labels={"year": "Year", "life_exp": "Average life expectancy (years)"},
                      title="Average life expectancy over time")
        st.plotly_chart(fig, use_container_width=True)
        explain("Each point is the average life expectancy across available country observations in that year. The line helps us see the overall direction over time.")
    else:
        fig = px.line(trend, x="year", y="u5_mort", markers=True,
                      labels={"year": "Year", "u5_mort": "Under-5 deaths per 1,000"},
                      title="Average under-5 mortality over time")
        st.plotly_chart(fig, use_container_width=True)
        explain("A lower value means fewer children die before reaching age five. The line shows whether the overall average is rising or falling.")
    st.caption("Yearly averages can be affected by which countries have data available in a given year.")

elif page == "Relationships":
    st.title("Relationships Between Indicators")
    st.write("These simple comparisons help us see how two indicators move together.")
    left, right = st.columns(2)
    with left:
        x_choice = st.selectbox("Compare life expectancy with",
            ["health_pc", "gdp_pc", "water", "avg_immunization", "fertility", "u5_mort"])
    with right:
        group_choice = st.selectbox("Colour points by", ["income_group", "region"])
    labels = {
        "health_pc": "Health spending per person (current US$)",
        "gdp_pc": "GDP per person (current US$)",
        "water": "Basic drinking-water access (%)",
        "avg_immunization": "Average DPT and measles immunisation (%)",
        "fertility": "Births per woman",
        "u5_mort": "Under-5 deaths per 1,000 live births"
    }
    plot_data = df[["life_exp", x_choice, group_choice, "country", "year"]].dropna()
    # Keep the chart readable and responsive while preserving a representative sample.
    if len(plot_data) > 1800:
        plot_data = plot_data.sample(1800, random_state=42)
    fig = px.scatter(
        plot_data, x=x_choice, y="life_exp", color=group_choice,
        hover_name="country", hover_data=["year"],
        labels={x_choice: labels[x_choice], "life_exp": "Life expectancy (years)"},
        title=f"Life expectancy compared with {labels[x_choice].lower()}",
        opacity=0.65
    )
    st.plotly_chart(fig, use_container_width=True)
    if x_choice == "u5_mort":
        explain("The points generally slope downwards: countries with higher under-5 mortality tend to have lower life expectancy.")
    elif x_choice == "fertility":
        explain("This chart helps us check whether higher fertility rates tend to appear alongside lower life expectancy in the dataset.")
    else:
        explain("Look at the overall direction of the points. An upward pattern means the indicators tend to increase together; a downward pattern means one tends to be lower when the other is higher. This is association, not proof of cause.")
    st.caption("Each dot represents a country-year observation. A sample is used when there are many observations so the chart stays easy to read.")

elif page == "Model Results":
    st.title("Prediction Model Results")
    st.write("The project compared models for predicting life expectancy and classifying child-mortality risk.")
    st.subheader("Life expectancy regression")
    reg = pd.DataFrame({
        "Model": ["Linear Regression", "Gradient Boosting", "Random Forest"],
        "Mean R²": [0.778, 0.776, 0.757],
        "Mean absolute error (years)": [2.90, 2.91, 2.94]
    })
    st.dataframe(reg, use_container_width=True, hide_index=True)
    fig = px.bar(reg, x="Model", y="Mean R²", range_y=[0, 1],
                 title="Life expectancy model comparison",
                 text="Mean R²")
    st.plotly_chart(fig, use_container_width=True)
    explain("R² measures how much variation in life expectancy the model explains; closer to 1 is better. Mean absolute error is the average prediction difference in years; lower is better. Linear Regression had the strongest overall result in this comparison.")
    st.subheader("Child-mortality risk classification")
    cls = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest"],
        "Accuracy": [0.887, 0.878],
        "F1 score": [0.860, 0.854]
    })
    st.dataframe(cls, use_container_width=True, hide_index=True)
    fig2 = px.bar(cls, x="Model", y=["Accuracy", "F1 score"], barmode="group",
                  range_y=[0, 1], title="Child-mortality classification scores")
    st.plotly_chart(fig2, use_container_width=True)
    explain("Accuracy is the share of classifications that were correct. F1 balances precision and recall, which is useful when we want to identify the higher-risk group. Logistic Regression performed slightly better in the reported results.")
    st.caption("Values shown are the project results recorded in the analysis notebook. They are model-performance measures, not guarantees for future data.")

elif page == "Transformer":
    st.title("Transformer Model: What We Tested")
    st.write("We also tested a tabular Transformer to see whether a more complex model improved life-expectancy prediction.")
    transformer = pd.DataFrame({
        "Model": ["Linear Regression", "Gradient Boosting", "Random Forest", "Tabular Transformer"],
        "Mean R²": [0.778, 0.776, 0.757, 0.768],
        "Mean absolute error (years)": [2.901, 2.910, 2.943, 2.965]
    })
    st.dataframe(transformer, use_container_width=True, hide_index=True)
    fig = px.bar(transformer, x="Model", y="Mean R²", range_y=[0, 1],
                 title="Transformer compared with other models", text="Mean R²")
    st.plotly_chart(fig, use_container_width=True)
    explain("The Transformer was included as an experiment. Its R² was slightly lower and its average error slightly higher than Linear Regression, so the simpler model was retained. A more complex model is not automatically a better model.")
    st.markdown("**Simple answer if asked:** We tested the Transformer to explore a more advanced approach. It did not improve the results enough to justify using it as our main model.")
    st.caption("Transformer result: mean R² 0.768 ± 0.044; it performed better than Linear Regression in 3 of 10 splits.")
