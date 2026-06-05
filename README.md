# Liver Disease Prediction using KNN and Logistic Regression

A machine learning project that predicts whether a patient has liver disease based on clinical blood test features, using K-Nearest Neighbours (KNN) and Logistic Regression models. The project includes a Tkinter desktop GUI for training the model and making predictions.

---

## Dataset

**File:** `indian_liver_patient.csv`  
**Source:** Indian Liver Patient Dataset (ILPD)  
**Samples:** 583 patients | **Features:** 10 + 1 target

| Column | Description |
|---|---|
| Age | Age of the patient |
| Gender | Male / Female |
| Total_Bilirubin | Total bilirubin level |
| Direct_Bilirubin | Direct bilirubin level |
| Alkaline_Phosphotase | Alkaline phosphatase enzyme level |
| Alamine_Aminotransferase | ALT enzyme level |
| Aspartate_Aminotransferase | AST enzyme level |
| Total_Protiens | Total protein level |
| Albumin | Albumin level |
| Albumin_and_Globulin_Ratio | Ratio of albumin to globulin |
| Dataset | Target — `1` = Liver disease, `2` = No disease |

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
scipy
Pillow
tkinter (built into Python on Windows/macOS)
```

Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels scipy Pillow
```

---

## How to Run

1. Place `indian_liver_patient.csv` in the same directory as the notebook.
2. Open `Major_Project_Fixed.ipynb` in Jupyter Notebook or JupyterLab.
3. Run all cells — this will launch the Tkinter GUI window.

### Using the GUI

| Button | Action |
|---|---|
| **Browse Files** | Select the `indian_liver_patient.csv` dataset file |
| **Train the Model** | Preprocesses data and trains both models; progress bar tracks stages |
| **Generate Results** | Displays the ROC curve and model summary as images |
| **Predict Output** | Enter patient values and click to get a liver disease prediction |

---

## Project Pipeline

1. **Data Loading** — reads CSV, renames columns to short codes
2. **Preprocessing**
   - Maps target column: `1 → Yes`, `2 → No`
   - Engineers features: `DB/TB Percentage`, `Globulin`
   - Drops redundant columns: `DB`, `TB`, `A/G Ratio`
   - Imputes 4 missing `Globulin` values using OLS regression on `Albumin`
3. **EDA** — correlation heatmaps, pair plots, box plots
4. **Upsampling** — minority class (`No disease`) upsampled to balance the dataset
5. **Logistic Regression** — four variants compared (all features, without Gender, log-transformed, sqrt-transformed); best model selected by F1 score
6. **KNN** — evaluated across odd values of k with both Min-Max and Standard scaling
7. **Evaluation** — ROC curves, AUC scores, confusion matrix metrics (Accuracy, Recall, Precision, F1, MCC)

---

## Results

| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression (log-transformed) | ~68.8% | ~64.1% |
| KNN (k=137, Min-Max scaled) | ~66.8% | ~59.7% |

Logistic Regression with log-transformed skewed features (ALP, ALT, AST) was selected as the final model.

---

## Bug Fixes Applied

The following fixes were made to the original notebook for compatibility with modern library versions:

| # | Fix |
|---|---|
| 1 | `dtype=np.bool` → `dtype=bool` (removed in NumPy 1.24+) |
| 2 | `pd.options.display.max_colwidth = -1` → `None` (removed in pandas 1.0+) |
| 3 | `liver.loc[... == 1, "Disease"] = "Yes"` → `.map({1: "Yes", 2: "No"})` (dtype mismatch) |
| 4 | `liver[-missing]` → `liver[~missing]` (wrong boolean negation operator) |
| 5 | Added `.astype(float)` / `.astype(int)` on `X` / `y` before statsmodels fitting |
| 6 | `plotCorrelationHeatmap(liver)` → `plotCorrelationHeatmap(liver.select_dtypes('number'))` (`.corr()` no longer drops non-numeric columns silently) |
| 7 | Missing `Display result bg.png` — replaced with auto-generated placeholder image |
