import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB

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

# Apply Naive Bayes without specific priors (default)
clf = GaussianNB(priors=None)
clf.fit(x_train, y_train)

# Predictions
predictions_test = clf.predict(x_test)

# Display confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test, predictions_test)
confusion_matrix_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
confusion_matrix_display.plot()
plt.show()

# Report Overall Accuracy, precision, recall, F1-score
print(metrics.classification_report(y_test, predictions_test, zero_division=0))

# Hyperparameter Optimization
cross_validation_accuracies = []
cross_validation_precisions = []
cross_validation_recalls = []
cross_validation_f1scores = []

# Define the priors
priors = [
    [0.0, 1.0],  # 0-100
    [0.1, 0.9],  # 10-90
    [0.2, 0.8],  # 20-80
    [0.3, 0.7],  # 30-70
    [0.4, 0.6],  # 40-60
    [0.5, 0.5],  # 50-50
    [0.6, 0.4],  # 60-40
    [0.7, 0.3],  # 70-30
    [0.8, 0.2],  # 80-20
    [0.9, 0.1],  # 90-10
    [1.0, 0.0]  # 100-0
]

# Loop through the priors and evaluate performance
for p in priors:
    print('Priors are: ', p)

    # Initialize and fit the Naive Bayes model with the current prior
    clf = GaussianNB(priors=p)
    clf.fit(x_train, y_train)

    # Cross-validation for accuracy
    accuracy = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='accuracy').mean()
    cross_validation_accuracies.append(accuracy)
    print(f'Cross-validation accuracy for prior {p}: {accuracy}')

    # Cross-validation for precision
    precision = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='precision').mean()
    cross_validation_precisions.append(precision)
    print(f'Cross-validation precision for prior {p}: {precision}')

    # Cross-validation for recall
    recall = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='recall').mean()
    cross_validation_recalls.append(recall)
    print(f'Cross-validation recall for prior {p}: {recall}')

    # Cross-validation for F1-score
    f1score = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='f1').mean()
    cross_validation_f1scores.append(f1score)
    print(f'Cross-validation f1-score for prior {p}: {f1score}')

# Plot accuracy vs. prior
plt.figure(figsize=(10, 6))
plt.plot(
    ['0-100', '10-90', '20-80', '30-70', '40-60',
     '50-50', '60-40', '70-30', '80-20', '90-10', '100-0'],
    cross_validation_accuracies,
    color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10
)
plt.title('Accuracy vs. Prior')
plt.xlabel('Prior')
plt.ylabel('Accuracy')
plt.show()
