
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Create models folder
os.makedirs("models", exist_ok=True)

# 1. Load dataset
data = pd.read_csv("data/spam.csv", encoding="latin-1")

# 2. Data cleaning
data = data[["v1", "v2"]]
data.columns = ["label", "message"]

data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})

print("Dataset loaded successfully!")
print(data.head())

# 3. Visualise dataset distribution
sns.countplot(data=data, x="label")
plt.title("Spam vs Ham Distribution")
plt.xticks([0, 1], ["Ham", "Spam"])
plt.savefig("models/class_distribution.png")
plt.show()

# 4. Separate features and target
X = data["message"]
y = data["label"]

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Define ML algorithms
algorithms = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC()
}

results = []
trained_models = {}

# 7. Train and evaluate models
for name, algorithm in algorithms.items():

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )),
        ("classifier", algorithm)
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    trained_models[name] = model

    print("\n", name)
    print(classification_report(
        y_test,
        predictions,
        target_names=["Ham", "Spam"]
    ))

# 8. Compare models
results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

# 9. Select best model based on F1 score
best_name = results_df.loc[
    results_df["F1 Score"].idxmax(), "Model"
]

best_model = trained_models[best_name]

joblib.dump(best_model, "models/best_spam_model.pkl")

print("\nBest Model:", best_name)
print("Model saved successfully!")

# 10. Visualise model comparison
results_df.set_index("Model").plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Machine Learning Model Comparison")
plt.ylabel("Performance Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("models/model_comparison.png")
plt.show()

# 11. Confusion matrix
predictions = best_model.predict(X_test)

cm = confusion_matrix(y_test, predictions)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png")
plt.show()

print("\nPROJECT TRAINING COMPLETED!")
