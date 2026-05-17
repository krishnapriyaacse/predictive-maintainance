# ==============================
# 🏭 Predictive Maintenance Model
# ==============================

# === IMPORTS ===
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Prevent plot popups
import matplotlib.pyplot as plt
import shap
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ==============================
# === LOAD DATA ===
# ==============================

df = pd.read_csv("predictive_maintenance.csv")

print("First 5 rows of dataset:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nDataset statistics:")
print(df.describe())

print("\nFailure type counts:")
print(df["Failure Type"].value_counts())

# ==============================
# === PREPROCESS DATA ===
# ==============================

# Drop unnecessary columns
df = df.drop(["UDI", "Product ID", "Failure Type"], axis=1)

# Convert 'Type' column to numerical (L/M)
df = pd.get_dummies(df, columns=["Type"], drop_first=True)

print("\nPreprocessed data (first 5 rows):")
print(df.head())

# ==============================
# === SPLIT DATA ===
# ==============================

X = df.drop("Target", axis=1)
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain size:", X_train.shape)
print("Test size:", X_test.shape)

# ==============================
# === TRAIN MODEL ===
# ==============================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("\n✅ Model training completed!")

# ==============================
# === EVALUATE MODEL ===
# ==============================

y_pred = model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==============================
# === CUSTOM THRESHOLD (0.3)
# ==============================

y_prob = model.predict_proba(X_test)[:, 1]
y_pred_custom = (y_prob > 0.3).astype(int)

print("\nConfusion Matrix (threshold=0.3):")
print(confusion_matrix(y_test, y_pred_custom))

print("\nClassification Report (threshold=0.3):")
print(classification_report(y_test, y_pred_custom))

# ==============================
# === FEATURE IMPORTANCE
# ==============================

importances = model.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(importance_df)

# Save Feature Importance Plot
plt.figure()
plt.bar(importance_df['Feature'], importance_df['Importance'])
plt.xticks(rotation=90)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("📊 Feature importance plot saved as feature_importance.png")

# ==============================
# === SHAP EXPLAINABILITY
# ==============================

# Use smaller sample for faster SHAP
X_sample = X_train.sample(500, random_state=42)

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_sample)

shap.summary_plot(shap_values.values, X_sample, plot_type="dot", show=False)
plt.savefig("shap_summary.png")
print("📈 SHAP summary plot saved as shap_summary.png")

# ==============================
# === SAVE TRAINED MODEL
# ==============================

joblib.dump(model, "machine_failure_model.pkl")
print("💾 Model saved successfully as machine_failure_model.pkl")

print("\n🎉 PROJECT COMPLETED SUCCESSFULLY!")
