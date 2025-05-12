import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import joblib  # for saving the model

# Step 1: Load Dataset
df = pd.read_csv("filtered_can_data.csv")

# Step 2: Convert Hexadecimal to Integer (for Data0 to Data7)
for col in [f"Data{i}" for i in range(8)]:
    df[col] = df[col].apply(lambda x: int(str(x), 16) if pd.notnull(x) else 0)

# Step 3: Prepare Features and Labels
X = df[[f"Data{i}" for i in range(8)]]
y = df["Label"]

# Step 4: Split Data (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Train Decision Tree Model (J48-style)
clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
clf.fit(X_train, y_train)

# Step 6: Save the trained model
joblib.dump(clf, "can_model.pkl")
print("✅ Model saved to can_model.pkl")

# Step 7: Evaluate the Model
y_pred = clf.predict(X_test)
print("\n✅ Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 8: Visualize and Save the Tree
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=[f"Data{i}" for i in range(8)],
          class_names=["Normal", "Abnormal"], filled=True)
plt.title("Decision Tree (J48-style) for CAN Data")
plt.savefig("can_tree.png")
print("✅ Decision tree image saved to can_tree.png")
plt.show()
