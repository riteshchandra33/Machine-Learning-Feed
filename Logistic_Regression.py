import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression

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

# Logistic Regression - Hyperparameter Optimization
cross_validation_accuracies = []
cross_validation_precisions = []
cross_validation_recalls = []
cross_validation_f1scores = []

# Define a range of values for the C parameter (inverse of regularization strength)
c_values = (0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1)

# Loop through the values of C and evaluate performance
for c in c_values:
    print(f'C value: {c}')

    # Initialize and fit the Logistic Regression model with the current C
    clf = LogisticRegression(C=c, random_state=42, max_iter=1000)
    clf.fit(x_train, y_train)

    # Cross-validation for accuracy
    accuracy = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='accuracy').mean()
    cross_validation_accuracies.append(accuracy)
    print(f'Cross-validation accuracy for C {c}: {accuracy}')

    # Cross-validation for precision
    precision = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='precision').mean()
    cross_validation_precisions.append(precision)
    print(f'Cross-validation precision for C {c}: {precision}')

    # Cross-validation for recall
    recall = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='recall').mean()
    cross_validation_recalls.append(recall)
    print(f'Cross-validation recall for C {c}: {recall}')

    # Cross-validation for F1-score
    f1score = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='f1').mean()
    cross_validation_f1scores.append(f1score)
    print(f'Cross-validation f1-score for C {c}: {f1score}')

# Plot accuracy vs. C value
plt.figure(figsize=(10, 6))
plt.plot(
    c_values,
    cross_validation_accuracies,
    color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10
)
plt.title('Accuracy vs. C Value for Logistic Regression')
plt.xlabel('C Value')
plt.ylabel('Accuracy')
plt.show()
