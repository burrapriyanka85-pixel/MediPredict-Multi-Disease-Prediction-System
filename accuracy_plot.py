import matplotlib.pyplot as plt

diseases = ["Diabetes", "Heart", "Liver", "Parkinsons"]
accuracy = [82, 88, 75, 92]  # replace with your actual values

plt.bar(diseases, accuracy)
plt.xlabel("Disease")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Comparison of Disease Prediction Models")
plt.show()
