# IRISFLOWERCLASSIFICATION
# **Iris Flower Classification - Machine Learning Project**

## **Project Overview**
This project implements a machine learning model to classify Iris flower species using the Random Forest Classifier. The dataset consists of four numerical features (Sepal Length, Sepal Width, Petal Length, Petal Width) and a categorical target variable representing three species of Iris flowers.

## **Dataset**
The dataset used is the **Iris dataset**, which contains 150 samples and three classes:
- Setosa
- Versicolor
- Virginica

## **Installation and Setup**
To run this project, you need Python and the required libraries installed. Follow these steps:

1. Clone the repository or download the project files.
2. Install the required dependencies:
   ```bash
   pip install pandas scikit-learn matplotlib seaborn
   ```
3. Place the `iris_data.csv` file in the project directory.
4. Run the Python script:
   ```bash
   python iris_classification.py
   ```

## **Project Workflow**
1. **Load Data:** Read the dataset using Pandas.
2. **Preprocess Data:**
   - Encode categorical labels.
   - Standardize features using `StandardScaler`.
3. **Split Dataset:** Divide into training (80%) and testing (20%) sets.
4. **Train Model:** Use `RandomForestClassifier` with optimized hyperparameters.
5. **Make Predictions:** Test the model on unseen data.
6. **Evaluate Model:** Calculate accuracy, display classification report, and visualize the confusion matrix.

## **Results**
- Achieved an **accuracy of 98.67%**.
- High precision and recall for all three species.
- Effective feature scaling and hyperparameter tuning improved model performance.

## **Outputs**
- **Model Accuracy:** Displayed in the terminal.
- **Confusion Matrix:** Graphically represented using Seaborn.
- **Classification Report:** Shows precision, recall, and F1-score.

## **Future Enhancements**
- Experiment with other classifiers like SVM or Neural Networks.
- Deploy the model using Flask or Streamlit for a user-friendly interface.
- Implement hyperparameter tuning using GridSearchCV.

## **Author**
[Your Name]

## **License**
This project is open-source and available for modifications and improvements.

