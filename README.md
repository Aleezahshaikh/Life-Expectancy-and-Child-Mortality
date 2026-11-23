# Data Analytics Project

## Project

**Determinants of Life Expectancy and Child Mortality: A Cross-Country Analysis Using World Bank Data**

This project analyzes country-level health and development indicators using the World Bank World Development Indicators (WDI). It explores the factors associated with life expectancy and under-5 child mortality through data cleaning, exploratory data analysis, statistical analysis, PCA, regression, classification, a tabular Transformer experiment, and data visualizations.

## Dashboard link 

The project dashboard is available, You can open it directly using the link below:

**(PASTE_YOUR_COLAB_NOTEBOOK_LINK_HERE)**



To run the project:

1. You can upload the colab file on Google Colab.
2. Sign in to your Google account if required.
3. Follow the notebook cells in order, or select **Runtime → Run all** to execute the complete analysis.
4. When prompted, upload the raw data file named **DA PROJ DATA**
5. Allow the notebook to complete execution and generate the outputs.

## Data and Analysis Period

- The original dataset was obtained from the World Bank World Development Indicators (WDI).
- The source data contains observations through 2024.
- The analysis covers **2000–2023**. The year 2024 was excluded because several key indicators have not been updated by WHo.
- Country region and income-group classifications are included in the project data.
- The cleaned dataset is also provided as a reference copy.

## Project Analysis

The project includes:

- **Data Cleaning and Preprocessing** — handling missing values, preparing indicators, and organizing country-year observations.
- **Exploratory Data Analysis** — examining distributions, trends, and differences across countries and income groups.
- **Statistical Analysis** — investigating relationships between health, economic, and development indicators.
- **Principal Component Analysis (PCA)** — exploring patterns and reducing dimensionality.
- **Regression Analysis** — estimating life expectancy using selected development indicators.
- **Classification** — predicting child mortality categories using machine learning models.
- **Transformer Experiment** — comparing a tabular Transformer model with conventional machine learning approaches.
- **Geographical Analysis** — presenting country-level health outcomes through maps and visualizations.

## Outputs and Visualizations

The project outputs are presented primarily through **figures, charts, plots, and maps**.

The `outputs/figures/` folder contains the visual outputs generated during the analysis, including relevant trend charts, relationship plots, model comparisons, and geographical visualizations.

## Transformer Experiment

The tabular Transformer experiment is included as an additional model comparison. It is evaluated alongside simpler machine learning models to assess whether the more complex approach improves predictive performance.

The Transformer is retained as part of the experimental analysis, while model selection is based on the comparison of evaluation results.

## Main Project Files

- `DA_PROJECT_FINAL.ipynb` — main project notebook.
- `DA_PROJECT_FINAL_original_backup.ipynb` — backup of the original notebook.
- `data/raw/` — original World Bank source dataset.
- `data/processed/` — cleaned reference dataset and country metadata.
- `outputs/figures/` — generated charts, plots, and visualizations.

## Data Source

World Bank, World Development Indicators (WDI):

https://databank.worldbank.org/source/world-development-indicators
