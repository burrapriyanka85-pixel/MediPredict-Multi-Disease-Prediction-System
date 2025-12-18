import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

MODEL_NAME = "Liver Disease Model"

df = pd.read_csv("data/indian_liver_patient.csv")

# Encode gender
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

# Fill missing values
df.fillna(df.mean(), inplace=True)

X = df.drop("Dataset", axis=1)
y = df["Dataset"].apply(lambda x: 1 if x == 2 else 0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"{MODEL_NAME} Accuracy: {accuracy*100:.2f}%")

pickle.dump(model, open("models/liver_model.pkl", "wb"))
print("✅ Liver model saved")
