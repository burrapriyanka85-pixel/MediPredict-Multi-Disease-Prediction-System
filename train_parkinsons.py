import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

MODEL_NAME = "Parkinsons Model"

df = pd.read_csv("data/parkinsons.csv")

X = df[
    [
        "MDVP:Fo(Hz)",
        "MDVP:Jitter(%)",
        "MDVP:Shimmer",
        "NHR",
        "HNR"
    ]
]

y = df["status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SVC(kernel="linear")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"{MODEL_NAME} Accuracy: {accuracy*100:.2f}%")

pickle.dump(model, open("models/parkinsons_model.pkl", "wb"))
print("✅ Parkinson model saved")
