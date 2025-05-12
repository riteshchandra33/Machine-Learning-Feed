import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model

# Load dataset
df = pd.read_csv('/Users/riteshchandra/pima-indians-diabetes.csv', index_col=0)
feature_names = df.columns[:-1]
print(df.head())

# Standardize the features
scaler = StandardScaler()
scaler.fit(df.drop('target', axis=1))
scaled_features = scaler.transform(df.drop('target', axis=1))
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
print(df_feat.head())

# Visualize the data
sns.pairplot(df, hue='target')
plt.show()

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(scaled_features, df['target'], test_size=0.3, stratify=df['target'], random_state=42)

# Define models
models = {
    "Linear Regression": linear_model.LinearRegression(),
    "Ridge Regression": linear_model.Ridge(alpha=0.5, random_state=0),
    "Lasso Regression": linear_model.Lasso(alpha=0.1, random_state=0)
}

# Store results for plotting
results = {
    "Model": [],
    "Accuracy": []
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(x_train, y_train)
    predictions_test = model.predict(x_test)
    # Replace regression results by class labels
    predictions_test[predictions_test <= 0.5] = 0
    predictions_test[predictions_test > 0.5] = 1
    predictions_test = predictions_test.astype(int)

    # Display confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_test, predictions_test, labels=[0, 1])
    confusion_matrix_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1])
    confusion_matrix_display.plot()
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

    # Report accuracy, precision, recall, F1-score
    print(f"Results for {name}:")
    print(metrics.classification_report(y_test, predictions_test, target_names=['0', '1'], zero_division=0))

    # Append accuracy for comparison later
    accuracy = metrics.accuracy_score(y_test, predictions_test)
    results["Model"].append(name)
    results["Accuracy"].append(accuracy)

# Optimize alpha for Ridge and Lasso
for reg_model, model_name in [(linear_model.Ridge, "Ridge"), (linear_model.Lasso, "Lasso")]:
    overall_accuracies = []
    alpha_values = np.arange(0.2, 2.0, 0.2)  # Alpha values from 0.2 to 1.8

    for alpha in alpha_values:
        clf = reg_model(alpha=alpha, random_state=0)
        clf.fit(x_train, y_train)
        predictions_test = clf.predict(x_test)
        predictions_test[predictions_test <= 0.5] = 0
        predictions_test[predictions_test > 0.5] = 1
        predictions_test = predictions_test.astype(int)
        overall_accuracies.append(metrics.accuracy_score(y_test, predictions_test))

    # Plot Accuracy vs. Alpha
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_values, overall_accuracies, label=f'{model_name} Accuracy', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
    plt.title(f'Accuracy vs. alpha_value ({model_name})')
    plt.xlabel('alpha-value')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Display results for Linear, Ridge, and Lasso
plt.figure(figsize=(10, 6))
plt.bar(results["Model"], results["Accuracy"], color=['blue', 'green', 'red'])
plt.title('Model Comparison - Accuracy')
plt.ylabel('Accuracy')
plt.show()
