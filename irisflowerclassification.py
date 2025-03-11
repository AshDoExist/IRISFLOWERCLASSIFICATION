import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data = pd.read_csv('iris_data.csv')  # Make sure the CSV file is in the same directory

# Step 2: Display the first few rows of the dataset to understand its structure
print("First few rows of the dataset:")
print(data.head())

# Step 3: Prepare the data
# Assuming the dataset has columns: 'SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species'
X = data[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]  # Features
y = data['Species']  # Target (flower type)

# Encode categorical target values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Feature scaling to improve model performance
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 5: Create and train the model (Optimized Random Forest Classifier)
model = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_split=2, min_samples_leaf=1,
                               max_features='sqrt', bootstrap=True, random_state=42)
model.fit(X_train, y_train)

# Step 6: Make predictions on the test data
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Improved Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Step 8: Confusion Matrix to see how well the model performs
cm = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix (Accuracy: {accuracy * 100:.2f}%)')
plt.show()