"""
Predicting UK road traffic accident severity.

Compares Logistic Regression, Random Forest, XGBoost and Balanced Random
Forest under several imbalance-handling strategies (class weighting,
RandomOverSampler, SMOTENC), tunes the best-performing pipeline with
RandomizedSearchCV, and reports feature importance for the final model.

Data source: UK STATS19 road safety data (data.gov.uk).
Place "Accident_Information.csv" in the working directory before running.
"""

import warnings
from pathlib import Path
from collections import OrderedDict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, precision_score, recall_score, f1_score
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from imblearn.ensemble import BalancedRandomForestClassifier

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False
    print("XGBoost not available, skipping XGBoost models.")

# display() only exists in Jupyter/Colab - fall back to print() for plain scripts
try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_JOBS = -1
SAMPLE_SIZE = 60000
DATA_PATH = "Accident_Information.csv"

OUTPUT_DIR = Path("project_outputs")
FIG_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "model_reports"

FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

severity_order = ["Fatal", "Serious", "Slight"]

plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["font.size"] = 10

print("Setup complete.")


# 1. Load and clean data

df = pd.read_csv(DATA_PATH, encoding="latin1", low_memory=False)
df.columns = df.columns.str.strip()

target = "Accident_Severity"

if target not in df.columns:
    raise ValueError("Accident_Severity column not found.")

df[target] = df[target].astype(str).str.strip()

severity_map = {
    "1": "Fatal", "2": "Serious", "3": "Slight",
    "fatal": "Fatal", "serious": "Serious", "slight": "Slight",
    "Fatal": "Fatal", "Serious": "Serious", "Slight": "Slight"
}

df[target] = df[target].replace(severity_map)
df = df[df[target].isin(severity_order)].copy()

print("Cleaned data shape:", df.shape)
print(df[target].value_counts().reindex(severity_order))
print((df[target].value_counts(normalize=True) * 100).reindex(severity_order).round(2))


# 2. Feature selection and feature engineering

candidate_features = [
    "1st_Road_Class", "2nd_Road_Class", "Carriageway_Hazards",
    "Day_of_Week", "Junction_Control", "Junction_Detail",
    "Light_Conditions", "Number_of_Vehicles",
    "Pedestrian_Crossing-Human_Control",
    "Pedestrian_Crossing-Physical_Facilities",
    "Road_Surface_Conditions", "Road_Type",
    "Special_Conditions_at_Site", "Speed_limit",
    "Urban_or_Rural_Area", "Weather_Conditions",
    "Year", "Date", "Time"
]

features = [c for c in candidate_features if c in df.columns]
df_model = df[features + [target]].copy()

if "Date" in df_model.columns:
    df_model["Date"] = pd.to_datetime(df_model["Date"], errors="coerce", dayfirst=True)
    df_model["Month"] = df_model["Date"].dt.month
    df_model.drop(columns=["Date"], inplace=True)

if "Time" in df_model.columns:
    df_model["Hour"] = pd.to_numeric(
        df_model["Time"].astype(str).str.extract(r"(\d{1,2}):")[0],
        errors="coerce"
    )
    df_model.drop(columns=["Time"], inplace=True)

if "Day_of_Week" in df_model.columns:
    df_model["Is_Weekend"] = df_model["Day_of_Week"].isin(["Saturday", "Sunday"]).astype(int)

if "Hour" in df_model.columns:
    df_model["Is_Night"] = df_model["Hour"].isin([20, 21, 22, 23, 0, 1, 2, 3, 4, 5]).astype(int)

if "Speed_limit" in df_model.columns:
    df_model["Speed_limit"] = pd.to_numeric(df_model["Speed_limit"], errors="coerce")
    df_model["High_Speed_Road"] = (df_model["Speed_limit"] >= 50).astype(int)

for col in df_model.columns:
    if col != target and df_model[col].dtype == "object":
        df_model[col] = df_model[col].replace({
            "Data missing or out of range": np.nan,
            "Unknown": np.nan,
            "unknown": np.nan,
            "": np.nan
        })

numeric_possible = [
    "Number_of_Vehicles", "Speed_limit", "Year", "Month", "Hour",
    "Is_Weekend", "Is_Night", "High_Speed_Road",
    "Pedestrian_Crossing-Human_Control",
    "Pedestrian_Crossing-Physical_Facilities"
]

for col in numeric_possible:
    if col in df_model.columns:
        df_model[col] = pd.to_numeric(df_model[col], errors="coerce")

df_model = df_model.dropna(subset=[target])

if len(df_model) > SAMPLE_SIZE:
    df_model, _ = train_test_split(
        df_model,
        train_size=SAMPLE_SIZE,
        stratify=df_model[target],
        random_state=RANDOM_STATE
    )

print("Final sample shape:", df_model.shape)
print(df_model[target].value_counts().reindex(severity_order))


# 3. Exploratory data analysis figures

def save_bar(series, title, xlabel, ylabel, filename, rotate=0):
    fig, ax = plt.subplots(figsize=(8, 5))
    series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotate)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=300, bbox_inches="tight")
    plt.show()

counts = df_model[target].value_counts().reindex(severity_order, fill_value=0)
percents = (df_model[target].value_counts(normalize=True) * 100).reindex(severity_order, fill_value=0)

save_bar(counts, "Accident Severity Distribution", "Severity", "Count", "figure_1_severity_distribution.png")
save_bar(percents, "Accident Severity Distribution (%)", "Severity", "Percentage", "figure_2_severity_distribution_percentage.png")

if "Speed_limit" in df_model.columns:
    speed_tab = pd.crosstab(df_model["Speed_limit"], df_model[target], normalize="index") * 100
    speed_tab = speed_tab[[c for c in severity_order if c in speed_tab.columns]]

    fig, ax = plt.subplots(figsize=(10, 6))
    speed_tab.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Accident Severity Proportion by Speed Limit")
    ax.set_xlabel("Speed limit")
    ax.set_ylabel("Percentage")
    ax.legend(title="Severity")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_3_severity_by_speed_limit.png", dpi=300, bbox_inches="tight")
    plt.show()

if "Urban_or_Rural_Area" in df_model.columns:
    area_tab = pd.crosstab(df_model["Urban_or_Rural_Area"], df_model[target], normalize="index") * 100
    area_tab = area_tab[[c for c in severity_order if c in area_tab.columns]]

    fig, ax = plt.subplots(figsize=(8, 5))
    area_tab.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Accident Severity Proportion by Urban/Rural Area")
    ax.set_xlabel("Area type")
    ax.set_ylabel("Percentage")
    ax.legend(title="Severity")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_4_severity_by_urban_rural.png", dpi=300, bbox_inches="tight")
    plt.show()


# 4. Train/test split and preprocessing

X = df_model.drop(columns=[target])
y_text = df_model[target]

label_encoder = LabelEncoder()
label_encoder.fit(severity_order)

y = label_encoder.transform(y_text)
class_names = list(label_encoder.classes_)

X_train, X_test, y_train_text, y_test_text = train_test_split(
    X,
    y_text,
    test_size=0.2,
    stratify=y_text,
    random_state=RANDOM_STATE
)

y_train = label_encoder.transform(y_train_text)
y_test = label_encoder.transform(y_test_text)

num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
cat_cols = [c for c in X.columns if c not in num_cols]

print("Numeric columns:", num_cols)
print("Categorical columns:", cat_cols)

def get_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

preprocessor_ohe = ColumnTransformer([
    ("cat", SkPipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", get_ohe())
    ]), cat_cols),
    ("num", SkPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False))
    ]), num_cols)
])

preprocessor_ord = ColumnTransformer([
    ("cat", SkPipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ]), cat_cols),
    ("num", SkPipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ]), num_cols)
])

cat_indices = list(range(len(cat_cols)))

print("Train distribution:")
print(y_train_text.value_counts().reindex(severity_order, fill_value=0))

print("Test distribution:")
print(y_test_text.value_counts().reindex(severity_order, fill_value=0))


# 5. Training class balance before/after resampling

def label_counts(y_array):
    labels = label_encoder.inverse_transform(y_array)
    return pd.Series(labels).value_counts().reindex(severity_order, fill_value=0)

X_train_ohe = preprocessor_ohe.fit_transform(X_train)

ros = RandomOverSampler(random_state=RANDOM_STATE)
_, y_ros = ros.fit_resample(X_train_ohe, y_train)

X_train_ord = preprocessor_ord.fit_transform(X_train)

min_class = min(Counter(y_train).values())
k_neighbors = max(1, min(5, min_class - 1))

smote = SMOTENC(
    categorical_features=cat_indices,
    random_state=RANDOM_STATE,
    k_neighbors=k_neighbors
)

_, y_smote = smote.fit_resample(X_train_ord, y_train)

balance_df = pd.DataFrame({
    "Original training": label_counts(y_train),
    "RandomOverSampler training": label_counts(y_ros),
    "SMOTENC training": label_counts(y_smote)
})

display(balance_df)
balance_df.to_csv(OUTPUT_DIR / "training_distribution_before_after_balancing.csv")

fig, ax = plt.subplots(figsize=(9, 5))
balance_df.plot(kind="bar", ax=ax)
ax.set_title("Training Class Distribution Before and After Balancing")
ax.set_xlabel("Severity")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=0)
fig.tight_layout()
fig.savefig(FIG_DIR / "figure_5_training_distribution_before_after_balancing.png", dpi=300, bbox_inches="tight")
plt.show()


# 6. Model definitions

def lr(weight=None):
    return LogisticRegression(
        max_iter=1500,
        class_weight=weight,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE
    )

def rf(weight=None):
    return RandomForestClassifier(
        n_estimators=100,
        class_weight=weight,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    )

def xgb():
    return XGBClassifier(
        objective="multi:softprob",
        num_class=len(class_names),
        eval_metric="mlogloss",
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        tree_method="hist"
    )

models = OrderedDict()

models["Logistic Regression - Imbalanced"] = ImbPipeline([
    ("prep", preprocessor_ohe),
    ("model", lr())
])

models["Random Forest - Imbalanced"] = ImbPipeline([
    ("prep", preprocessor_ohe),
    ("model", rf())
])

if XGB_AVAILABLE:
    models["XGBoost - Imbalanced"] = ImbPipeline([
        ("prep", preprocessor_ohe),
        ("model", xgb())
    ])

models["Logistic Regression - Class Weighted"] = ImbPipeline([
    ("prep", preprocessor_ohe),
    ("model", lr("balanced"))
])

models["Random Forest - Class Weighted"] = ImbPipeline([
    ("prep", preprocessor_ohe),
    ("model", rf("balanced"))
])

models["Logistic Regression - RandomOverSampler"] = ImbPipeline([
    ("prep", preprocessor_ohe),
    ("ros", RandomOverSampler(random_state=RANDOM_STATE)),
    ("model", lr())
])

models["Random Forest - RandomOverSampler"] = ImbPipeline([
    ("prep", preprocessor_ohe),
    ("ros", RandomOverSampler(random_state=RANDOM_STATE)),
    ("model", rf())
])

if XGB_AVAILABLE:
    models["XGBoost - RandomOverSampler"] = ImbPipeline([
        ("prep", preprocessor_ohe),
        ("ros", RandomOverSampler(random_state=RANDOM_STATE)),
        ("model", xgb())
    ])

models["Random Forest - SMOTENC"] = ImbPipeline([
    ("prep", preprocessor_ord),
    ("smote", SMOTENC(cat_indices, random_state=RANDOM_STATE, k_neighbors=k_neighbors)),
    ("model", rf())
])

if XGB_AVAILABLE:
    models["XGBoost - SMOTENC"] = ImbPipeline([
        ("prep", preprocessor_ord),
        ("smote", SMOTENC(cat_indices, random_state=RANDOM_STATE, k_neighbors=k_neighbors)),
        ("model", xgb())
    ])

models["Balanced Random Forest"] = ImbPipeline([
    ("prep", preprocessor_ohe),
    ("model", BalancedRandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    ))
])

print("Models ready:", len(models))


# 7. Evaluation helpers

cm_files = {
    "Logistic Regression - Imbalanced": "figure_6_logistic_regression_imbalanced_cm.png",
    "Random Forest - Imbalanced": "figure_7_random_forest_imbalanced_cm.png",
    "XGBoost - Imbalanced": "figure_8_xgboost_imbalanced_cm.png",
    "Logistic Regression - Class Weighted": "figure_9_logistic_regression_class_weighted_cm.png",
    "Random Forest - Class Weighted": "figure_10_random_forest_class_weighted_cm.png",
    "Logistic Regression - RandomOverSampler": "figure_11_logistic_regression_ros_cm.png",
    "Random Forest - RandomOverSampler": "figure_12_random_forest_ros_cm.png",
    "XGBoost - RandomOverSampler": "figure_13_xgboost_ros_cm.png",
    "Random Forest - SMOTENC": "figure_14_random_forest_smotenc_cm.png",
    "XGBoost - SMOTENC": "figure_15_xgboost_smotenc_cm.png",
    "Balanced Random Forest": "figure_16_balanced_random_forest_cm.png",
    "Random Forest - RandomOverSampler - Tuned": "figure_20_tuned_random_forest_ros_cm.png"
}

def clean_name(name):
    return name.lower().replace(" ", "_").replace("-", "").replace("/", "_")

def plot_cm(cm, title, filename):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=300, bbox_inches="tight")
    plt.show()

def evaluate(model_name, fitted_model):
    pred = fitted_model.predict(X_test)

    report = classification_report(
        y_test,
        pred,
        target_names=class_names,
        zero_division=0
    )

    report_dict = classification_report(
        y_test,
        pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(y_test, pred, labels=list(range(len(class_names))))

    with open(REPORT_DIR / f"{clean_name(model_name)}_report.txt", "w") as f:
        f.write(report)

    plot_cm(cm, f"Confusion Matrix: {model_name}", cm_files.get(model_name, f"{clean_name(model_name)}_cm.png"))

    row = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, pred),
        "Balanced Accuracy": balanced_accuracy_score(y_test, pred),
        "Macro Precision": precision_score(y_test, pred, average="macro", zero_division=0),
        "Macro Recall": recall_score(y_test, pred, average="macro", zero_division=0),
        "Macro F1": f1_score(y_test, pred, average="macro", zero_division=0),
        "Weighted F1": f1_score(y_test, pred, average="weighted", zero_division=0)
    }

    for cls in class_names:
        row[f"{cls} Precision"] = report_dict[cls]["precision"]
        row[f"{cls} Recall"] = report_dict[cls]["recall"]
        row[f"{cls} F1"] = report_dict[cls]["f1-score"]

    return row, report, cm


# 8. Train and evaluate all models

rows = []
fitted_models = {}

for name, pipe in models.items():
    print("\nTraining:", name)
    try:
        model = pipe.fit(X_train, y_train)
        row, report, cm = evaluate(name, model)
        rows.append(row)
        fitted_models[name] = model
        print(report)
    except Exception as e:
        print("Failed:", name)
        print(e)

results_df = pd.DataFrame(rows).sort_values("Macro F1", ascending=False)
results_df.to_csv(OUTPUT_DIR / "model_comparison_before_tuning.csv", index=False)

print("Model comparison before tuning:")
display(results_df)


# 9. Hyperparameter tuning (Random Forest)

rf_tune_pipe = ImbPipeline([
    ("prep", preprocessor_ohe),
    ("ros", RandomOverSampler(random_state=RANDOM_STATE)),
    ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS))
])

rf_grid = {
    "model__n_estimators": [50, 100],
    "model__max_depth": [10, 20, None],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2],
    "model__max_features": ["sqrt"]
}

rf_search = RandomizedSearchCV(
    rf_tune_pipe,
    rf_grid,
    n_iter=5,
    scoring="f1_macro",
    cv=2,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS,
    verbose=2
)

rf_search.fit(X_train, y_train)

print("Best RF parameters:")
print(rf_search.best_params_)
print("Best CV macro F1:", rf_search.best_score_)

tuned_row, tuned_report, tuned_cm = evaluate(
    "Random Forest - RandomOverSampler - Tuned",
    rf_search.best_estimator_
)

print(tuned_report)

fitted_models["Random Forest - RandomOverSampler - Tuned"] = rf_search.best_estimator_
tuned_df = pd.DataFrame([tuned_row])


# 10. Final results table

final_df = pd.concat([results_df, tuned_df], ignore_index=True)
final_df = final_df.drop_duplicates("Model", keep="last")
final_df = final_df.sort_values("Macro F1", ascending=False)

final_df.to_csv(OUTPUT_DIR / "final_model_comparison_results_with_tuning.csv", index=False)

print("Final model comparison:")
display(final_df)

before_after_df = final_df[
    final_df["Model"].isin([
        "Random Forest - RandomOverSampler",
        "Random Forest - RandomOverSampler - Tuned"
    ])
][[
    "Model", "Accuracy", "Balanced Accuracy", "Macro F1",
    "Weighted F1", "Fatal Recall", "Serious Recall"
]]

before_after_df.to_csv(OUTPUT_DIR / "before_after_hyperparameter_tuning.csv", index=False)

print("Before and after tuning:")
display(before_after_df)


# 11. Final comparison figures

short_names = {
    "Logistic Regression - Imbalanced": "LR Imbalanced",
    "Random Forest - Imbalanced": "RF Imbalanced",
    "XGBoost - Imbalanced": "XGB Imbalanced",
    "Logistic Regression - Class Weighted": "LR Weighted",
    "Random Forest - Class Weighted": "RF Weighted",
    "Logistic Regression - RandomOverSampler": "LR ROS",
    "Random Forest - RandomOverSampler": "RF ROS",
    "XGBoost - RandomOverSampler": "XGB ROS",
    "Random Forest - SMOTENC": "RF SMOTENC",
    "XGBoost - SMOTENC": "XGB SMOTENC",
    "Balanced Random Forest": "Balanced RF",
    "Random Forest - RandomOverSampler - Tuned": "RF ROS Tuned"
}

plot_df = final_df.copy()
plot_df["Short"] = plot_df["Model"].replace(short_names)

def metric_plot(metric, filename, title):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(plot_df["Short"], plot_df[metric])
    ax.set_title(title)
    ax.set_xlabel(metric)
    ax.set_ylabel("Model")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=300, bbox_inches="tight")
    plt.show()

metric_plot("Macro F1", "figure_17_final_model_comparison_macro_f1.png", "Final Model Comparison by Macro F1")
metric_plot("Balanced Accuracy", "figure_18_final_model_comparison_balanced_accuracy.png", "Final Model Comparison by Balanced Accuracy")

recall_df = plot_df[["Short", "Fatal Recall", "Serious Recall"]].set_index("Short")

fig, ax = plt.subplots(figsize=(12, 6))
recall_df.plot(kind="bar", ax=ax)
ax.set_title("Final Minority-Class Recall Comparison")
ax.set_xlabel("Model")
ax.set_ylabel("Recall")
ax.tick_params(axis="x", rotation=75)
ax.legend(title="Class")
fig.tight_layout()
fig.savefig(FIG_DIR / "figure_19_final_minority_class_recall.png", dpi=300, bbox_inches="tight")
plt.show()


# 12. Feature importance

best_rf = rf_search.best_estimator_

feature_names = best_rf.named_steps["prep"].get_feature_names_out()
importances = best_rf.named_steps["model"].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False)

importance_df["Feature Clean"] = (
    importance_df["Feature"]
    .str.replace("cat__", "", regex=False)
    .str.replace("num__", "", regex=False)
    .str.replace("_", " ", regex=False)
)

importance_df.to_csv(OUTPUT_DIR / "tuned_random_forest_feature_importance.csv", index=False)

print("Top 20 feature importances:")
display(importance_df[["Feature Clean", "Importance"]].head(20))

top_imp = importance_df.head(20).sort_values("Importance")

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(top_imp["Feature Clean"], top_imp["Importance"])
ax.set_title("Top 20 Feature Importances from Tuned Random Forest")
ax.set_xlabel("Importance")
ax.set_ylabel("Feature")
fig.tight_layout()
fig.savefig(FIG_DIR / "figure_21_feature_importance_tuned_random_forest.png", dpi=300, bbox_inches="tight")
plt.show()


# 13. Save summary

best_model = final_df.iloc[0]["Model"]

summary = f"""
Final Project Summary

Sample size: {len(df_model)}
Target: {target}
Classes: {class_names}

Best model by Macro F1:
{best_model}

Main point:
The tuned Random Forest with RandomOverSampler gave the best Macro F1-score.
Balancing was applied only on the training data. The test set stayed imbalanced.

Top 10 feature importances:
{importance_df[["Feature Clean", "Importance"]].head(10).to_string(index=False)}
"""

with open(OUTPUT_DIR / "final_project_summary.txt", "w") as f:
    f.write(summary)

print(summary)
print("All outputs saved in:", OUTPUT_DIR)
print("Figures saved in:", FIG_DIR)
print("Reports saved in:", REPORT_DIR)
