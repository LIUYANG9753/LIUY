import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# ====================== 1. Style Configuration ======================
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'axes.labelsize': 10.5,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'figure.dpi': 600,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.4,
    'figure.autolayout': True,
})

# ====================== 2. Data Loading and Preprocessing ======================
data_path = r"E:\BaiduSyncdisk\1_paper\1_paper2\excel\analysis_new.xlsx"
try:
    df = pd.read_excel(data_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Excel file not found at: {data_path}")

# Print available columns for debugging
print("Available columns in dataset:", df.columns.tolist())

# Clean column names to remove leading/trailing whitespace
df.columns = df.columns.str.strip()

# Define X and Y variables
x_cols = ['CE', 'PADcv', 'PAI', 'GF']
y_cols = ['LAcv', 'LAmean']

# Verify columns exist
missing_cols = [col for col in x_cols + y_cols if col not in df.columns]
if missing_cols:
    raise ValueError(
        f"Missing columns in dataset: {missing_cols}. Please check column names in dataset: {df.columns.tolist()}")

# Select relevant columns and drop rows with missing values
data = df[x_cols + y_cols].dropna()

# Check for sufficient data
if len(data) < len(x_cols) + 1:
    raise ValueError(f"Insufficient data points for regression analysis. Only {len(data)} observations available.")

# Print data summary for debugging
print(f"Data shape after preprocessing: {data.shape}")
print("Data summary:")
print(data[x_cols + y_cols].describe())

# Split X and Y
X = data[x_cols]
y_LAcv = data['LAcv']
y_lamean = data['LAmean']

# Standardize features for better model stability
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=x_cols, index=X.index)

# Check for multicollinearity using VIF
vif_data = pd.DataFrame()
vif_data['Feature'] = x_cols
vif_data['VIF'] = [variance_inflation_factor(X_scaled.values, i) for i in range(X_scaled.shape[1])]
print("Variance Inflation Factors (VIF):")
print(vif_data)
if any(vif_data['VIF'] > 10):
    print("Warning: High VIF detected. Consider removing correlated features or using regularization.")


# ====================== 3. OLS Modeling Function ======================
def fit_and_evaluate_ols(data, y_data, y_name, x_cols):
    """Fit OLS regression model and evaluate performance."""
    try:
        X = data[x_cols]
        y = y_data
    except KeyError as e:
        raise KeyError(f"Column {e} not found in data. Available columns: {data.columns.tolist()}")

    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()

    # Predictions
    y_pred = model.predict(X_with_const)

    # Metrics
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # Print results
    print(f"\nResults for {y_name}:")
    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("Coefficients:")
    for feature, coef, pval in zip(x_cols, model.params[1:], model.pvalues[1:]):
        print(f"{feature}: {coef:.4f} (p-value: {pval:.4f})")
    print(model.summary())

    return model, y_pred, r2, rmse


# ====================== 4. Fit OLS Models for LAcv and LAmean ======================
# Fit OLS for LAcv
model_LAcv, y_pred_LAcv, r2_LAcv, rmse_LAcv = fit_and_evaluate_ols(X_scaled, y_LAcv, 'LAcv', x_cols)

# Fit OLS for LAmean
model_lamean, y_pred_lamean, r2_lamean, rmse_lamean = fit_and_evaluate_ols(X_scaled, y_lamean, 'LAmean', x_cols)

# ====================== 5. Visualization ======================
# Set journal-style configuration
plt.rcParams.update({
    'font.family': 'Times New Roman',  # Use Times New Roman as requested
    'font.size': 22,  # Base font size for readability in print
    'axes.labelsize': 10,  # Axis labels slightly larger
    'axes.titlesize': 10,  # Subplot titles
    'xtick.labelsize': 8,  # Tick labels smaller
    'ytick.labelsize': 8,
    'figure.dpi': 600,  # High resolution for preview
    'axes.linewidth': 0.8,  # Thinner axis lines
    'grid.linewidth': 0.3,  # Subtle grid
    'axes.grid': False,  # Disable grid by default
    'figure.autolayout': True,
})

# Colorblind-friendly palette (inspired by ColorBrewer)
colors = {
    'LAcv': '#1b9e77',  # Teal for LAcv
    'LAmean': '#d95f02',  # Orange for LAmean
    'fit_line': '#7570b3',  # Purple for regression lines
}

# Create figure with subplots
fig, axes = plt.subplots(2, len(x_cols), figsize=(len(x_cols) * 2.2, 4), sharex='col', sharey='row')
axes = axes.reshape(2, -1) if len(x_cols) > 1 else axes.reshape(2, 1)

# Plot each subplot
for i, x_col in enumerate(x_cols):
    # LAcv scatter plot (top row)
    axes[0, i].scatter(X[x_col], y_LAcv, c=colors['LAcv'], alpha=0.6, s=20, edgecolors='none')
    axes[0, i].set_xlabel(x_col, labelpad=2)
    if i == 0:
        axes[0, i].set_ylabel('LAcv', labelpad=2)
    # Fit simple linear regression for visualization
    slope, intercept = np.polyfit(X[x_col], y_LAcv, 1)
    x_range = np.linspace(X[x_col].min(), X[x_col].max(), 100)
    axes[0, i].plot(x_range, slope * x_range + intercept, c=colors['fit_line'], lw=1.2)
    # Calculate p-value for simple linear regression
    X_simple = sm.add_constant(X[x_col])
    sm_model_LAcv = sm.OLS(y_LAcv, X_simple).fit()
    p_value = sm_model_LAcv.pvalues[1]  # p-value for slope
    # Add coefficient and p-value annotation (R² removed)
    axes[0, i].text(0.05, 0.95, f'β = {slope:.2f}\np = {p_value:.3f}',
                    transform=axes[0, i].transAxes,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1), fontsize=10)

    # LAmean scatter plot (bottom row)
    axes[1, i].scatter(X[x_col], y_lamean, c=colors['LAmean'], alpha=0.6, s=20, edgecolors='none')
    axes[1, i].set_xlabel(x_col, labelpad=2)
    if i == 0:
        axes[1, i].set_ylabel('LAmean', labelpad=2)
    # Fit simple linear regression for visualization
    slope, intercept = np.polyfit(X[x_col], y_lamean, 1)
    axes[1, i].plot(x_range, slope * x_range + intercept, c=colors['fit_line'], lw=1.2)
    # Calculate p-value for simple linear regression
    X_simple = sm.add_constant(X[x_col])
    sm_model_lamean = sm.OLS(y_lamean, X_simple).fit()
    p_value = sm_model_lamean.pvalues[1]  # p-value for slope
    # Add coefficient and p-value annotation (R² removed)
    axes[1, i].text(0.05, 0.95, f'β = {slope:.2f}\np = {p_value:.3f}',
                    transform=axes[1, i].transAxes,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1), fontsize=10)

    # Customize spines and ticks
    for ax in [axes[0, i], axes[1, i]]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.tick_params(axis='both', which='both', length=3, pad=2)

# Adjust layout to prevent overlap
plt.tight_layout(pad=0.5, h_pad=0.8, w_pad=0.8)

# Save figure in required formats
plt.savefig('Regression_Scatter.tif', format='tiff', dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.savefig('Regression_Scatter.pdf', format='pdf')
plt.savefig('Regression_Scatter.eps', format='eps')  # EPS for vector submission
plt.show()

# ====================== 6. Interpretation and Table Generation ======================
print("\nInterpretation:")
print("1. LAcv (Light Attenuation Coefficient of Variation):")
print(f"   - R²={r2_LAcv:.4f} indicates {r2_LAcv * 100:.1f}% of variance explained.")
print(f"   - Significant predictors (p<0.05): {[x_cols[i] for i, p in enumerate(model_LAcv.pvalues[1:]) if p < 0.05]}")
print("2. LAmean (Mean Light Attenuation):")
print(f"   - R²={r2_lamean:.4f} indicates {r2_lamean * 100:.1f}% of variance explained.")
print(
    f"   - Significant predictors (p<0.05): {[x_cols[i] for i, p in enumerate(model_lamean.pvalues[1:]) if p < 0.05]}")

# Generate table for paper
table_data = {
    'Response': [],
    'Predictor': [],
    'Coefficient': [],
    'Standard Error': [],
    'p-value': [],
    'R²': [],
    'RMSE': []
}

for response, model, r2, rmse in [('LAmean', model_lamean, r2_lamean, rmse_lamean),
                                  ('LAcv', model_LAcv, r2_LAcv, rmse_LAcv)]:
    table_data['Response'].extend([response] * (len(x_cols) + 1))
    table_data['Predictor'].extend(['Intercept'] + x_cols)
    table_data['Coefficient'].extend([f"{coef:.3f}" for coef in model.params])
    table_data['Standard Error'].extend([f"{se:.3f}" for se in model.bse])
    table_data['p-value'].extend([f"{p:.3f}" if p >= 0.001 else "<0.001" for p in model.pvalues])
    table_data['R²'].extend([f"{r2:.4f}" if i == 0 else "" for i in range(len(x_cols) + 1)])
    table_data['RMSE'].extend([f"{rmse:.4f}" if i == 0 else "" for i in range(len(x_cols) + 1)])

table_df = pd.DataFrame(table_data)
print("\nTable for Paper:")
print(table_df.to_string(index=False))

# Save table as LaTeX for journal submission
latex_table = table_df.to_latex(index=False, float_format="%.3f",
                                caption="Fixed Effects from OLS Regression Models for LAmean and LAcv",
                                label="tab:ols_results",
                                column_format="llcccll")
with open('ols_results_table.tex', 'w') as f:
    f.write(latex_table)