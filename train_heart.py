import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

MODEL_NAME = "Heart Disease Model"

df = pd.read_csv("data/Heart_Disease_Prediction.csv")

X = df.drop("Heart Disease", axis=1)
y = df["Heart Disease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"{MODEL_NAME} Accuracy: {accuracy*100:.2f}%")

pickle.dump(model, open("models/heart_model.pkl", "wb"))
print("✅ Heart model saved")
