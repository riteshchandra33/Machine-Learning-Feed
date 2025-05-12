import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

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

# Hyperparameter configurations
min_samples_split_values = [5, 10, 15, 20]
min_samples_leaf_values = [3, 7, 11, 15]

# Loop through Decision Tree configurations and print cross-validation results
print(f'Decision Tree Model')
for min_split in min_samples_split_values:
    for min_leaf in min_samples_leaf_values:
        clf = DecisionTreeClassifier(min_samples_split=min_split, min_samples_leaf=min_leaf)

        # Cross-validation for accuracy
        accuracy = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='accuracy').mean()
        print(f'Cross-validation accuracy for min_samples_split {min_split}, min_samples_leaf {min_leaf}: {accuracy}')

        # Cross-validation for precision
        precision = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='precision').mean()
        print(f'Cross-validation precision for min_samples_split {min_split}, min_samples_leaf {min_leaf}: {precision}')

        # Cross-validation for recall
        recall = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='recall').mean()
        print(f'Cross-validation recall for min_samples_split {min_split}, min_samples_leaf {min_leaf}: {recall}')

        # Cross-validation for F1-score
        f1score = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='f1').mean()
        print(f'Cross-validation F1-score for min_samples_split {min_split}, min_samples_leaf {min_leaf}: {f1score}')
        print()  # Blank line for better readability between configurations

# Loop through Random Forest configurations and print cross-validation results
print(f'Random Forest Model')
for min_split in min_samples_split_values:
    for min_leaf in min_samples_leaf_values:
        clf = RandomForestClassifier(min_samples_split=min_split, min_samples_leaf=min_leaf, n_estimators=100)

        # Cross-validation for accuracy
        accuracy = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='accuracy').mean()
        print(f'Cross-validation accuracy for min_samples_split {min_split}, min_samples_leaf {min_leaf}: {accuracy}')

        # Cross-validation for precision
        precision = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='precision').mean()
        print(f'Cross-validation precision for min_samples_split {min_split}, min_samples_leaf {min_leaf}: {precision}')

        # Cross-validation for recall
        recall = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='recall').mean()
        print(f'Cross-validation recall for min_samples_split {min_split}, min_samples_leaf {min_leaf}: {recall}')

        # Cross-validation for F1-score
        f1score = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='f1').mean()
        print(f'Cross-validation F1-score for min_samples_split {min_split}, min_samples_leaf {min_leaf}: {f1score}')
        print()

# Produce visualizations of the tree graph for the last trained model
print(f'Hyperparameters used for the plotted Decision Tree:')
print(f'min_samples_split: {min_split}')
print(f'min_samples_leaf: {min_leaf}')

# Visualize the Decision Tree with sample parameters
clf = DecisionTreeClassifier(min_samples_split=7, min_samples_leaf=2)
clf.fit(x_train, y_train)
plt.figure(figsize=(16, 8))
plot_tree(clf, max_depth=3, feature_names=feature_names, class_names=list(map(str, clf.classes_)), filled=True)
plt.show()

import matplotlib.pyplot as plt

# Initialize dictionary to store accuracy values for plotting
accuracy_results = {split: [] for split in min_samples_split_values}

# Loop through each configuration to calculate accuracy and store it
for min_split in min_samples_split_values:
    accuracies_for_split = []
    for min_leaf in min_samples_leaf_values:
        clf = DecisionTreeClassifier(min_samples_split=min_split, min_samples_leaf=min_leaf)
        accuracy = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='accuracy').mean()
        accuracies_for_split.append(accuracy)
    accuracy_results[min_split] = accuracies_for_split

# Plotting
plt.figure(figsize=(10, 6))
for min_split, accuracies in accuracy_results.items():
    plt.plot(min_samples_leaf_values, accuracies, linestyle='dashed', marker='o', markerfacecolor='blue', markersize=6,
             label=f'min_samples_split={min_split}')

plt.title('Accuracy vs. min_samples_leaf for Decision Tree')
plt.xlabel('min_samples_leaf')
plt.ylabel('Accuracy')
plt.legend(title="min_samples_split")
plt.show()

# Initialize dictionary to store accuracy values for plotting
rf_accuracy_results = {split: [] for split in min_samples_split_values}

# Loop through each configuration to calculate accuracy and store it
for min_split in min_samples_split_values:
    rf_accuracies_for_split = []
    for min_leaf in min_samples_leaf_values:
        clf = RandomForestClassifier(min_samples_split=min_split, min_samples_leaf=min_leaf, n_estimators=100)
        accuracy = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='accuracy').mean()
        rf_accuracies_for_split.append(accuracy)
    rf_accuracy_results[min_split] = rf_accuracies_for_split

# Plotting
plt.figure(figsize=(10, 6))
for min_split, accuracies in rf_accuracy_results.items():
    plt.plot(min_samples_leaf_values, accuracies, linestyle='dashed', marker='o', markerfacecolor='green', markersize=6,
             label=f'min_samples_split={min_split}')

plt.title('Accuracy vs. min_samples_leaf for Random Forest')
plt.xlabel('min_samples_leaf')
plt.ylabel('Accuracy')
plt.legend(title="min_samples_split")
plt.show()

