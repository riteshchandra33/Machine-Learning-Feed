import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC

# Read dataset
df = pd.read_csv('/Users/riteshchandra/pima-indians-diabetes.csv', index_col=0)
feature_names = df.columns[:-1]

# Standardize the features
scaler = StandardScaler()
scaler.fit(df.drop('target', axis=1))
scaled_features = scaler.transform(df.drop('target', axis=1))
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])

# Split the data into train and test
x_train, x_test, y_train, y_test = train_test_split(scaled_features, df['target'], test_size=0.3, stratify=df['target'],
                                                    random_state=42)

# SVM - Hyperparameter Optimization
cross_validation_accuracies = []
cross_validation_precisions = []
cross_validation_recalls = []
cross_validation_f1scores = []
q
# Define a range of values for the C parameter (inverse of regularization strength)
c_values = (0.5,1.0,1.5,2.0)

models = {
    'Poly Degree 2': {'kernel': 'poly', 'degree': 2},quit()
    'Poly Degree 3': {'kernel': 'poly', 'degree': 3},
    'Poly Degree 4': {'kernel': 'poly', 'degree': 4},
    'Linear': {'kernel': 'linear'},
    'Sigmoid': {'kernel': 'sigmoid'},
    'RBF': {'kernel': 'rbf'}
}
# Create dictionaries to store results
accuracy_results = {model_name: [] for model_name in models}
precision_results = {model_name: [] for model_name in models}
recall_results = {model_name: [] for model_name in models}
f1_results = {model_name: [] for model_name in models}

for model_name, model_params in models.items():
    print(f"Evaluating model: {model_name}")

    for c in c_values:
        print(f'C value: {c}')

        # Initialize the SVC model with the current C and kernel/degree
        clf = SVC(C=c, kernel=model_params['kernel'], degree=model_params.get('degree', 3), random_state=42)
        clf.fit(x_train, y_train)

        # Cross-validation for accuracy
        accuracy = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='accuracy').mean()
        accuracy_results[model_name].append(accuracy)

        # Cross-validation for precision
        precision = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='precision').mean()
        precision_results[model_name].append(precision)

        # Cross-validation for recall
        recall = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='recall').mean()
        recall_results[model_name].append(recall)

        # Cross-validation for F1-score
        f1score = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='f1').mean()
        f1_results[model_name].append(f1score)

        print(f'Cross-validation accuracy for {model_name} with C {c}: {accuracy}')
        print(f'Cross-validation precision for {model_name} with C {c}: {precision}')
        print(f'Cross-validation recall for {model_name} with C {c}: {recall}')
        print(f'Cross-validation F1-score for {model_name} with C {c}: {f1score}')

plt.figure(figsize=(12, 8))
for model_name, accuracies in accuracy_results.items():
    plt.plot(c_values, accuracies, label=model_name, marker='o')  # Different markers for each model

# Customize the plot
plt.title('Accuracy vs. C Value for Different SVM Models')
plt.xlabel('C Value')
plt.ylabel('Accuracy')
plt.xticks(c_values)  # Set x-ticks to the defined C values
plt.legend(title='SVM Models')
plt.grid(True)
plt.tight_layout()
plt.show()
