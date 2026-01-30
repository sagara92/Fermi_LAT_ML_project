# Machine Learning Classification of Fermi-LAT Blazars

## 1. Problem Statement
Classify uncertain blazars of the type BCUs from *Fermi*-LAT 4FGL catalog into two known classes: **BL Lacertae objects (BLL)** and **Flat Spectrum Radio Quasars (FSRQ)**.

The classification pipeline followed these systematic steps:

1.  **Data Preprocessing:** 
    * The catalog `gll_psc_v35.fits` file was obtained from [link](https://fermi.gsfc.nasa.gov/ssc/data/access/lat/14yr_catalog/) and converted to a `csv` file containing numeric parameters (features) and identifying names (labels).
    * The dataset (`blazar_catalog_v35_features.csv`) was filtered to isolate labeled data ('bll' and 'fsrq').
    * Numerical features were cleaned using **Median Imputation** to handle missing values.

2.  **Stratified Splitting:** 
    * The data was split into **Training (60%)**, **Validation (20%)**, and **Testing (20%)** sets.
    * Stratification preserves the natural imbalance ratio of BLL vs. FSRQ across all subsets, preventing sampling bias.

3.  **Model Selection:** Three distinct supervised learning algorithms were implemented:
    * **Decision Tree (DT):** A baseline interpretable model.
    * **Random Forest (RF):** An ensemble bagging method used to reduce variance and overfitting.
    * **XGBoost (GBDT):** A gradient boosting method utilized for its optimized performance on tabular data and ability to handle non-linear decision boundaries.
4.  **Hyperparameter Optimization:** 
    * A **GridSearchCV** was performed for each model using 5-fold cross-validation.
    * Parameters such as `max_depth`, `n_estimators`, and splitting criteria were tuned to maximize classification accuracy.

## 2. How ML is helping with performance metrics
ML classification methods are significant upgrade over traditional spectral cuts, such as separating/classifying sources based on optical features (e.g., Pivot Energy), as ML methods analyzes multi-dimensional correlations between all the numeric features simultaneously.

* **Confusion Matrix:** This was the primary visual metric which breaks down performance not just by overall accuracy, but by specific class errors.
* **F1-Score:** Due to the potential imbalance in blazar datasets, accuracy alone can be misleading. The F1-score (the harmonic mean of Precision and Recall) was calculated to ensure the model did not simply achieve high accuracy by guessing the majority class.