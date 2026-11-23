# Data Analytics Project

## Project
**Determinants of Life Expectancy and Child Mortality: A Cross-Country Analysis Using World Bank Data**

This project analyzes country-level health and development indicators using the World Bank World Development Indicators (WDI). It includes data cleaning, exploratory analysis, statistical analysis, PCA, regression, classification, a tabular-transformer experiment, and 2023 country maps.

## Data and analysis period
- `data/raw/DA PROJ DATA.csv` is the downloaded source dataset. It contains source observations through 2024.
- The analysis uses **2000–2023**. 2024 is excluded because several key indicators have high missingness.
- `data/processed/country_metadata.csv` contains the country region and income-group classifications used by the notebook. It is bundled locally so the notebook does not need to call the World Bank API during execution.
- `data/processed/wdi_clean_final_reference.csv` is a reference copy of the cleaned dataset supplied with the project. The notebook regenerates its own cleaned output.


## Run the interactive Streamlit app

The Streamlit app is a simple, browser-based way to explore the project. It uses the
cleaned dataset already included in `data/processed/` and does not need to download data.

From a terminal opened in the project folder:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

A browser tab will open automatically. If it does not, copy the local address printed
in the terminal (usually `http://localhost:8501`) into your browser.

The app contains:
- **Overview** — project purpose and key findings
- **Data Explorer** — filter and inspect country-year records
- **Trends** — simple line charts over time
- **Relationships** — easy-to-read scatter plots
- **Model Results** — regression and classification scores
- **Transformer** — the Transformer experiment and why the simpler model was retained

The charts are deliberately kept simple. Their explanations describe what to look for,
and remind readers that association does not prove causation.

## Requirements
Python 3.10 or later is recommended. Install packages from the project folder:

```bash
python -m pip install -r requirements.txt
```

## Run the project
1. Extract the complete project folder. Keep the `data` folder beside the notebook.
2. Open a terminal in the project folder.
3. Install the requirements.
4. Start Jupyter Lab or Jupyter Notebook:

```bash
jupyter lab
```

5. Open `DA_PROJECT_FINAL.ipynb` and choose **Run All**.
6. Generated cleaned data, result tables and figures are saved under `outputs/`.

The notebook uses relative paths and the bundled country metadata, so no World Bank API access is required. An internet connection is not required for the data-processing and analysis steps.

## Transformer experiment
The tabular transformer experiment is intentionally retained as an additional model comparison. PyTorch is included in `requirements.txt`. The experiment is not presented as the preferred model; its results are compared with the simpler regression models.

## Main files
- `DA_PROJECT_FINAL.ipynb` — simplified, runnable notebook.
- `DA_PROJECT_FINAL_original_backup.ipynb` — unchanged backup of the submitted notebook.
- `data/raw/` — original source dataset.
- `data/processed/` — country metadata and reference cleaned data.
- `outputs/figures/` — supplied plots and interactive map files; regenerated figures are saved here.
- `outputs/tables/` — generated analysis tables.

## Data source
World Bank, World Development Indicators: https://databank.worldbank.org/source/world-development-indicators
