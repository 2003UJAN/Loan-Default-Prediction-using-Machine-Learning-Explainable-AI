import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Step 1: Create synthetic dataset
np.random.seed(42)
n_samples = 5000

data = pd.DataFrame({
    "loan_amount": np.random.randint(1000, 50000, n_samples),
    "interest_rate": np.random.uniform(5, 30, n_samples),
    "income": np.random.randint(20000, 200000, n_samples),
    "employment_years": np.random.randint(0, 40, n_samples),
    "credit_score": np.random.randint(300, 850, n_samples),
    "age": np.random.randint(18, 70, n_samples),
    "existing_debt": np.random.randint(0, 50000, n_samples),
    "loan_term": np.random.choice([12, 24, 36, 48, 60], n_samples)
})

# Define default probability
prob_default = (
    0.3 * (data["interest_rate"] / 30) +
    0.2 * (1 - (data["credit_score"] - 300) / 550) +
    0.2 * (data["loan_amount"] / 50000) +
    0.2 * (data["existing_debt"] / 50000) +
    0.1 * (1 - (data["income"] / 200000))
)

data["default"] = np.random.binomial(1, prob_default.clip(0,1))

# Step 2: Train/test split
X = data.drop("default", axis=1)
y = data["default"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Step 4: Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save dataset for reference
data.to_csv("data/synthetic_loans.csv", index=False)
print("✅ Model and synthetic dataset saved!")
