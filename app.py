# --------------------------------------------
# CLV Evaluation Dashboard - Final Version
# --------------------------------------------

# 1️⃣ Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import streamlit as st

# 2️⃣ Title
st.title("🧮 Customer Lifetime Value (CLV) Evaluation Dashboard")

# 3️⃣ Load Data (with caching)
@st.cache_data
def load_data():
    df = pd.read_csv("sample_customer_data.csv")
    df['CustomerID'] = df['CustomerID'].str.replace("C", '').astype(int)
    return df

df = load_data()
st.write("✅ Data Loaded:", df.head())

# 4️⃣ Prepare Features
X = df.drop("target", axis=1)

# Use get_dummies if needed
X = pd.get_dummies(X)

y = df["target"]

# 5️⃣ Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6️⃣ Train
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 7️⃣ Evaluation
c_matrix = confusion_matrix(y_test, y_pred)
c_report = classification_report(y_test, y_pred)

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

# 8️⃣ Show Feature Importance
st.subheader("🔑 Feature Importance")
st.dataframe(feature_importance)

# 9️⃣ Confusion Matrix Plot
st.subheader("📊 Confusion Matrix")
fig, ax = plt.subplots()
ax.matshow(c_matrix, cmap="Blues")
for i in range(c_matrix.shape[0]):
    for j in range(c_matrix.shape[1]):
        ax.text(j, i, c_matrix[i, j], va='center', ha='center', color="white")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# 🔟 Show Classification Report
st.subheader("📄 Classification Report")
st.text(c_report)

# 1️⃣1️⃣ Business Insight Example
st.subheader("💡 Example Recommendation")
test_feature = "purchase_frequency"
test_value = 15

def business_insight(feature, value):
    if feature == "purchase_frequency" and value > 10:
        return "✅ Encourage loyalty upgrades"
    elif feature == "avg_order_value" and value < 100:
        return "💡 Bundle products to raise order value"
    else:
        return "📌 Maintain engagement strategy"

st.write(f"For `{test_feature}` = {test_value} → {business_insight(test_feature, test_value)}")

# Footer
st.write("---")
st.write("📌 *Run completed successfully!*")
