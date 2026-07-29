
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# ==========================================
# Settings
# ==========================================
# If True, the visualizations will show all rows in the dataset (about 16k points).
# The printed accuracy/recall still come from the test set only.
USE_ALL_DATA_POINTS_IN_VISUALIZATION = True

# Numeric axis range for visual encoding of categories
MIN_VIS_CODE = 0.10
MAX_VIS_CODE = 1.00

# Jitter controls how much the categorical points spread visually.
# Increase slightly if points still look too stacked.
PUBLISHER_JITTER = 0.0025
PLATFORM_JITTER = 0.0100

# Grid resolution for red/green background regions.
# Higher = more detail but slower.
GRID_SIZE = 260

# ==========================================
# 1. Load the dataset
# ==========================================
dataset = pd.read_csv('vgsales.csv')

# Create folders for outputs
os.makedirs('cleaning_visualizations', exist_ok=True)
os.makedirs('model_visualizations', exist_ok=True)

# ==========================================
# 2. Data Preprocessing / Cleaning
# ==========================================

missing_before = dataset[['Year', 'Publisher']].isnull().sum()

print("Missing values before cleaning:")
print(missing_before)

# Publisher is categorical, so missing values are replaced with Unknown.
dataset['Publisher'] = dataset['Publisher'].fillna('Unknown')

# Year is numerical, so missing values are replaced with the average Year of the same Platform.
platform_year_mean = dataset.groupby('Platform')['Year'].transform('mean')
dataset['Year'] = dataset['Year'].fillna(platform_year_mean)

# Fallback if any Year values are still missing
dataset['Year'] = dataset['Year'].fillna(dataset['Year'].mean())
dataset['Year'] = dataset['Year'].round().astype(int)

missing_after = dataset[['Year', 'Publisher']].isnull().sum()

print("\nMissing values after cleaning:")
print(missing_after)

# ==========================================
# 3. Missing Values Visualization
# ==========================================

plt.figure(figsize=(7, 5))
columns = ['Year', 'Publisher']
x = np.arange(len(columns))
width = 0.35

plt.bar(x - width / 2, missing_before.values, width, label='Before Cleaning')
plt.bar(x + width / 2, missing_after.values, width, label='After Cleaning')

plt.xticks(x, columns)
plt.ylabel('Missing Values Count')
plt.title('Missing Values Before and After Cleaning')
plt.legend()

for i, value in enumerate(missing_before.values):
    plt.text(i - width / 2, value + 2, str(value), ha='center')

for i, value in enumerate(missing_after.values):
    plt.text(i + width / 2, value + 2, str(value), ha='center')

plt.tight_layout()
plt.savefig('cleaning_visualizations/missing_values_before_after.png', dpi=200)
plt.show()

# ==========================================
# 4. Outlier Detection Summary (IQR Method)
# ==========================================
# Outliers are detected using the IQR method, but they are NOT removed because
# extreme sales values are part of the real market behavior in video game sales.
# Instead of plotting separate scatter plots for each sales column, a single
# clean summary visualization is created.

sales_columns = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
outlier_summary = []
total_rows = len(dataset)

for col in sales_columns:
    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_flag = (dataset[col] < lower_bound) | (dataset[col] > upper_bound)
    outlier_count = int(outlier_flag.sum())
    outlier_percent = (outlier_count / total_rows) * 100

    outlier_summary.append({
        'Column': col,
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'Lower Bound': lower_bound,
        'Upper Bound': upper_bound,
        'Outlier Count': outlier_count,
        'Outlier Percentage': outlier_percent
    })

outlier_summary = pd.DataFrame(outlier_summary)

print("\nOutlier summary using IQR method:")
print(outlier_summary)
print("Note: Outliers were detected only. No rows were removed.")

outlier_summary.to_csv('outlier_summary.csv', index=False)

# Visual summary of detected outliers
plt.figure(figsize=(9, 5))
bar_positions = np.arange(len(outlier_summary))
bar_values = outlier_summary['Outlier Count'].values
bar_labels = outlier_summary['Column'].values

plt.bar(bar_positions, bar_values)
plt.xticks(bar_positions, bar_labels)
plt.ylabel('Detected Outlier Count')
plt.title('Outlier Summary Using IQR Method')
plt.grid(True, axis='y', alpha=0.3)

for i, row in outlier_summary.iterrows():
    plt.text(
        i,
        row['Outlier Count'] + max(bar_values) * 0.02,
        f"{int(row['Outlier Count'])}\n({row['Outlier Percentage']:.1f}%)",
        ha='center',
        va='bottom',
        fontsize=9
    )

plt.tight_layout()
plt.savefig('cleaning_visualizations/outlier_summary_iqr_visualization.png', dpi=200)
plt.show()

# ==========================================
# 5. Feature Engineering: Define Target Variable
# ==========================================

dataset['IsItHit'] = (dataset['Global_Sales'] >= 1.0).astype(int)

features_df = dataset[['Platform', 'Year', 'Genre', 'Publisher']].copy()
y = dataset['IsItHit'].values

# Encode categorical data
X_encoded_df = pd.get_dummies(features_df, drop_first=True)
X = X_encoded_df.values

# ==========================================
# 6. Splitting the Dataset into Training and Test Sets
# ==========================================
from sklearn.model_selection import train_test_split

# Keep original row indices so visualization can use the same test samples if needed.
row_indices = np.arange(len(dataset))

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X,
    y,
    row_indices,
    test_size=0.30,
    random_state=42
)

# ==========================================
# 7. Feature Scaling
# ==========================================
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_all_scaled = sc.transform(X)

# ==========================================
# 8. Train and Evaluate Models
# ==========================================
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

model_results = []

def save_model_result(model_name, acc, recall, precision, f1, cm):
    model_results.append({
        'Model': model_name,
        'Accuracy': acc,
        'Hit Recall': recall,
        'Hit Precision': precision,
        'Hit F1-score': f1,
        'TN': cm[0, 0],
        'FP': cm[0, 1],
        'FN': cm[1, 0],
        'TP': cm[1, 1]
    })

# ==========================================
# Model 1: Logistic Regression
# ==========================================
classifier_lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
classifier_lr.fit(X_train, y_train)

y_pred_lr = classifier_lr.predict(X_test)
y_pred_lr_all = classifier_lr.predict(X_all_scaled)

cm_lr = confusion_matrix(y_test, y_pred_lr)
acc_lr = accuracy_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)

print("\nLogistic Regression Results:")
print("Confusion Matrix:")
print(cm_lr)
print("Accuracy =", acc_lr)
print("Hit Recall =", recall_lr)
print("Hit Precision =", precision_lr)
print("Hit F1-score =", f1_lr)

save_model_result('Logistic Regression', acc_lr, recall_lr, precision_lr, f1_lr, cm_lr)

# ==========================================
# Model 2: Random Forest
# ==========================================
classifier_rf = RandomForestClassifier(n_estimators=50, criterion='entropy', random_state=42)
classifier_rf.fit(X_train, y_train)

# Lower threshold to detect more Hit games
rf_hit_threshold = 0.25
y_pred_rf = (classifier_rf.predict_proba(X_test)[:, 1] >= rf_hit_threshold).astype(int)
y_pred_rf_all = (classifier_rf.predict_proba(X_all_scaled)[:, 1] >= rf_hit_threshold).astype(int)

cm_rf = confusion_matrix(y_test, y_pred_rf)
acc_rf = accuracy_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print("\nRandom Forest Results:")
print("Confusion Matrix:")
print(cm_rf)
print("Hit threshold =", rf_hit_threshold)
print("Accuracy =", acc_rf)
print("Hit Recall =", recall_rf)
print("Hit Precision =", precision_rf)
print("Hit F1-score =", f1_rf)

save_model_result('Random Forest', acc_rf, recall_rf, precision_rf, f1_rf, cm_rf)

# ==========================================
# Model 3: SVM Linear Kernel
# ==========================================
classifier_linear = SVC(kernel='linear', class_weight='balanced', random_state=42, max_iter=2000)
classifier_linear.fit(X_train, y_train)

y_pred_linear = classifier_linear.predict(X_test)
y_pred_linear_all = classifier_linear.predict(X_all_scaled)

cm_linear = confusion_matrix(y_test, y_pred_linear)
acc_linear = accuracy_score(y_test, y_pred_linear)
recall_linear = recall_score(y_test, y_pred_linear)
precision_linear = precision_score(y_test, y_pred_linear)
f1_linear = f1_score(y_test, y_pred_linear)

print("\nSVM (Linear Kernel) Results:")
print("Confusion Matrix:")
print(cm_linear)
print("Accuracy =", acc_linear)
print("Hit Recall =", recall_linear)
print("Hit Precision =", precision_linear)
print("Hit F1-score =", f1_linear)

save_model_result('SVM Linear', acc_linear, recall_linear, precision_linear, f1_linear, cm_linear)

# ==========================================
# Model 4: SVM RBF Kernel
# ==========================================
classifier_rbf = SVC(kernel='rbf', class_weight='balanced', random_state=42, max_iter=2000)
classifier_rbf.fit(X_train, y_train)

y_pred_rbf = classifier_rbf.predict(X_test)
y_pred_rbf_all = classifier_rbf.predict(X_all_scaled)

cm_rbf = confusion_matrix(y_test, y_pred_rbf)
acc_rbf = accuracy_score(y_test, y_pred_rbf)
recall_rbf = recall_score(y_test, y_pred_rbf)
precision_rbf = precision_score(y_test, y_pred_rbf)
f1_rbf = f1_score(y_test, y_pred_rbf)

print("\nSVM (RBF Kernel) Results:")
print("Confusion Matrix:")
print(cm_rbf)
print("Accuracy =", acc_rbf)
print("Hit Recall =", recall_rbf)
print("Hit Precision =", precision_rbf)
print("Hit F1-score =", f1_rbf)

save_model_result('SVM RBF', acc_rbf, recall_rbf, precision_rbf, f1_rbf, cm_rbf)

# ==========================================
# Model 5: SVM Polynomial Kernel
# ==========================================
classifier_poly = SVC(kernel='poly', degree=3, class_weight='balanced', random_state=42, max_iter=2000)
classifier_poly.fit(X_train, y_train)

y_pred_poly = classifier_poly.predict(X_test)
y_pred_poly_all = classifier_poly.predict(X_all_scaled)

cm_poly = confusion_matrix(y_test, y_pred_poly)
acc_poly = accuracy_score(y_test, y_pred_poly)
recall_poly = recall_score(y_test, y_pred_poly)
precision_poly = precision_score(y_test, y_pred_poly)
f1_poly = f1_score(y_test, y_pred_poly)

print("\nSVM (Polynomial Kernel) Results:")
print("Confusion Matrix:")
print(cm_poly)
print("Accuracy =", acc_poly)
print("Hit Recall =", recall_poly)
print("Hit Precision =", precision_poly)
print("Hit F1-score =", f1_poly)

save_model_result('SVM Polynomial', acc_poly, recall_poly, precision_poly, f1_poly, cm_poly)

# Save model results
model_results_df = pd.DataFrame(model_results)
model_results_df.to_csv('model_results_summary.csv', index=False)

# ==========================================
# 9. PCA Visualization Only
# ==========================================
# IMPORTANT:
# - This section does NOT change the train/test split.
# - This section does NOT retrain the original classification models.
# - This section does NOT recalculate accuracy, recall, precision, or F1.
# - It only creates 2D visualizations for the already-trained models.
#
# Visualization meaning:
# - Red background  = model decision region: Predicted Not Hit (0)
# - Green background = model decision region: Predicted Hit (1)
# - Red points      = actual data: Actual Not Hit (0)
# - Green points    = actual data: Actual Hit (1)
#
# Special note for Random Forest:
# Random Forest is visualized using a 2D surrogate surface fitted only for drawing.
# The original Random Forest model and its printed results remain unchanged.

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Create output folder
os.makedirs('model_visualizations_pca_clean', exist_ok=True)

# ------------------------------------------
# Visualization settings only
# ------------------------------------------
USE_ALL_POINTS_FOR_SCATTER = USE_ALL_DATA_POINTS_IN_VISUALIZATION

GRID_RESOLUTION = 300
POINT_SIZE_NOT_HIT = 5
POINT_SIZE_HIT = 7
POINT_ALPHA_NOT_HIT = 0.24
POINT_ALPHA_HIT = 0.36

JITTER_RATIO_X = 0.008
JITTER_RATIO_Y = 0.008

ZOOM_LOW = 2
ZOOM_HIGH = 98
RANDOM_STATE_VIS = 42

# ------------------------------------------
# PCA projection for visualization only
# ------------------------------------------
pca = PCA(n_components=2, random_state=RANDOM_STATE_VIS)
pca.fit(X_train)  # fitted only on already-scaled training data

X_test_2d = pca.transform(X_test)
X_all_2d = pca.transform(X_all_scaled)

explained_variance = pca.explained_variance_ratio_.sum() * 100
print(f"\nPCA explained variance for visualization only = {explained_variance:.2f}%")

# ------------------------------------------
# Choose which real points to display
# ------------------------------------------
if USE_ALL_POINTS_FOR_SCATTER:
    scatter_xy_base = X_all_2d
    scatter_y_true = y
    visual_label = "All Dataset Points"
else:
    scatter_xy_base = X_test_2d
    scatter_y_true = y_test
    visual_label = "Test Set Points"

# ------------------------------------------
# Zoom to the main cloud of points
# ------------------------------------------
x_min = np.percentile(X_all_2d[:, 0], ZOOM_LOW)
x_max = np.percentile(X_all_2d[:, 0], ZOOM_HIGH)
y_min = np.percentile(X_all_2d[:, 1], ZOOM_LOW)
y_max = np.percentile(X_all_2d[:, 1], ZOOM_HIGH)

x_pad = (x_max - x_min) * 0.08
y_pad = (y_max - y_min) * 0.08

x_min -= x_pad
x_max += x_pad
y_min -= y_pad
y_max += y_pad

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, GRID_RESOLUTION),
    np.linspace(y_min, y_max, GRID_RESOLUTION)
)

grid_2d = np.c_[xx.ravel(), yy.ravel()]

# Convert PCA grid back to original scaled feature space.
# This is used only to create approximate decision regions for visualization.
grid_high_dim = pca.inverse_transform(grid_2d)

# ------------------------------------------
# Visual jitter for points only
# ------------------------------------------
rng = np.random.default_rng(RANDOM_STATE_VIS)
jitter_x = (x_max - x_min) * JITTER_RATIO_X
jitter_y = (y_max - y_min) * JITTER_RATIO_Y

scatter_x = scatter_xy_base[:, 0] + rng.normal(0, jitter_x, size=len(scatter_xy_base))
scatter_y = scatter_xy_base[:, 1] + rng.normal(0, jitter_y, size=len(scatter_xy_base))

scatter_x = np.clip(scatter_x, x_min, x_max)
scatter_y = np.clip(scatter_y, y_min, y_max)

actual_not_hit = scatter_y_true == 0
actual_hit = scatter_y_true == 1

# ------------------------------------------
# Shared legend
# ------------------------------------------
def get_common_legend_handles():
    return [
        Patch(facecolor='#f7c6c6', edgecolor='none', alpha=0.55, label='Region: Predicted Not Hit (0)'),
        Patch(facecolor='#c9f2c4', edgecolor='none', alpha=0.55, label='Region: Predicted Hit (1)'),
        Line2D([0], [0], color='blue', linewidth=1.5, label='Decision Boundary'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=6, label='Actual Not Hit (0)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=6, label='Actual Hit (1)')
    ]

# ------------------------------------------
# General PCA decision-region plot
# Used for Logistic Regression and SVM models
# ------------------------------------------
def plot_pca_region_actual_data(
    model,
    model_name,
    acc,
    recall,
    precision,
    f1,
    cm,
    filename
):
    # Model decisions on the PCA grid approximation
    grid_pred = model.predict(grid_high_dim)
    Z = grid_pred.reshape(xx.shape)

    plt.figure(figsize=(10, 7))

    # Background regions
    plt.contourf(
        xx,
        yy,
        Z,
        levels=[-0.5, 0.5, 1.5],
        colors=['#f7c6c6', '#c9f2c4'],
        alpha=0.45
    )

    # Decision boundary, if both classes appear in the grid
    if len(np.unique(Z)) > 1:
        plt.contour(
            xx,
            yy,
            Z,
            levels=[0.5],
            colors='blue',
            linewidths=1.3
        )

    # Actual data points
    plt.scatter(
        scatter_x[actual_not_hit],
        scatter_y[actual_not_hit],
        c='red',
        s=POINT_SIZE_NOT_HIT,
        alpha=POINT_ALPHA_NOT_HIT,
        edgecolors='none'
    )

    plt.scatter(
        scatter_x[actual_hit],
        scatter_y[actual_hit],
        c='green',
        s=POINT_SIZE_HIT,
        alpha=POINT_ALPHA_HIT,
        edgecolors='none'
    )

    tn, fp, fn, tp = cm.ravel()

    summary_text = (
        f'Plotted Points: {len(scatter_xy_base)}\n'
        f'Actual Not Hit: {(scatter_y_true == 0).sum()}\n'
        f'Actual Hit: {(scatter_y_true == 1).sum()}\n'
        f'TP={tp}, TN={tn}, FP={fp}, FN={fn}'
    )

    plt.text(
        0.985,
        0.02,
        summary_text,
        transform=plt.gca().transAxes,
        ha='right',
        va='bottom',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.88)
    )

    plt.title(
        f'{model_name} PCA Decision Region Visualization\n'
        f'Accuracy={acc:.4f}, Recall={recall:.4f}, Precision={precision:.4f}, F1={f1:.4f} | {visual_label}'
    )
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(True, alpha=0.20)
    plt.legend(handles=get_common_legend_handles(), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(f'model_visualizations_pca_clean/{filename}', dpi=200)
    plt.show()

# ------------------------------------------
# Random Forest special visualization
# ------------------------------------------
def plot_random_forest_fixed_visualization():
    # Original RF probabilities on real high-dimensional data
    # This uses the original trained Random Forest model only for probability output.
    rf_prob_all = classifier_rf.predict_proba(X_all_scaled)[:, 1]

    # 2D surrogate surface for visualization only.
    # This does NOT replace or retrain the original Random Forest classifier.
    rf_visual_surrogate = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=30,
        random_state=RANDOM_STATE_VIS,
        n_jobs=-1
    )
    rf_visual_surrogate.fit(X_all_2d, rf_prob_all)

    grid_scores = rf_visual_surrogate.predict(grid_2d)
    grid_scores = np.clip(grid_scores, 0, 1).reshape(xx.shape)

    plt.figure(figsize=(10, 7))

    # Background based on RF hit probability threshold.
    # Red = predicted Not Hit, Green = predicted Hit.
    plt.contourf(
        xx,
        yy,
        grid_scores,
        levels=[0.0, rf_hit_threshold, 1.0],
        colors=['#f7c6c6', '#c9f2c4'],
        alpha=0.45
    )

    # Decision boundary at the same threshold used in RF evaluation.
    if grid_scores.min() <= rf_hit_threshold <= grid_scores.max():
        plt.contour(
            xx,
            yy,
            grid_scores,
            levels=[rf_hit_threshold],
            colors='blue',
            linewidths=1.5
        )

    # Actual data points
    plt.scatter(
        scatter_x[actual_not_hit],
        scatter_y[actual_not_hit],
        c='red',
        s=POINT_SIZE_NOT_HIT,
        alpha=POINT_ALPHA_NOT_HIT,
        edgecolors='none'
    )

    plt.scatter(
        scatter_x[actual_hit],
        scatter_y[actual_hit],
        c='green',
        s=POINT_SIZE_HIT,
        alpha=POINT_ALPHA_HIT,
        edgecolors='none'
    )

    tn, fp, fn, tp = cm_rf.ravel()

    summary_text = (
        f'Plotted Points: {len(scatter_xy_base)}\n'
        f'Actual Not Hit: {(scatter_y_true == 0).sum()}\n'
        f'Actual Hit: {(scatter_y_true == 1).sum()}\n'
        f'TP={tp}, TN={tn}, FP={fp}, FN={fn}'
    )

    plt.text(
        0.985,
        0.02,
        summary_text,
        transform=plt.gca().transAxes,
        ha='right',
        va='bottom',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.88)
    )

    plt.title(
        f'Random Forest PCA Decision Region Visualization\n'
        f'Accuracy={acc_rf:.4f}, Recall={recall_rf:.4f}, Precision={precision_rf:.4f}, F1={f1_rf:.4f} | {visual_label}'
    )
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(True, alpha=0.20)
    plt.legend(handles=get_common_legend_handles(), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig('model_visualizations_pca_clean/random_forest_pca_fixed_visualization.png', dpi=200)
    plt.show()

# ==========================================
# Generate the five model visualizations
# ==========================================

plot_pca_region_actual_data(
    classifier_lr,
    'Logistic Regression',
    acc_lr,
    recall_lr,
    precision_lr,
    f1_lr,
    cm_lr,
    'logistic_regression_pca_visualization.png'
)

plot_random_forest_fixed_visualization()

plot_pca_region_actual_data(
    classifier_linear,
    'SVM Linear Kernel',
    acc_linear,
    recall_linear,
    precision_linear,
    f1_linear,
    cm_linear,
    'svm_linear_pca_visualization.png'
)

plot_pca_region_actual_data(
    classifier_rbf,
    'SVM RBF Kernel',
    acc_rbf,
    recall_rbf,
    precision_rbf,
    f1_rbf,
    cm_rbf,
    'svm_rbf_pca_visualization.png'
)

plot_pca_region_actual_data(
    classifier_poly,
    'SVM Polynomial Kernel',
    acc_poly,
    recall_poly,
    precision_poly,
    f1_poly,
    cm_poly,
    'svm_polynomial_pca_visualization.png'
)

print("\nAll PCA visualizations were saved in: model_visualizations_pca_clean")
print("Random Forest was visualized using a separate 2D surrogate surface for visualization only.")
print("Cleaning visualization was saved in: cleaning_visualizations")
print("Outlier summary visualization was saved in: cleaning_visualizations/outlier_summary_iqr_visualization.png")
print("Model results were saved in: model_results_summary.csv")
print("Outlier summary was saved in: outlier_summary.csv")
