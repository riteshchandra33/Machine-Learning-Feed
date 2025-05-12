from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score

# Read dataset
df = pd.read_csv('/Users/riteshchandra/pima-indians-diabetes.csv', index_col=0)

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop('target', axis=1))
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])

# Split the data into train and test
x_train, x_test, y_train, y_test = train_test_split(scaled_features, df['target'], test_size=0.3, stratify=df['target'], random_state=42)

# Define the activation functions and hidden layer sizes to loop through
activation_functions = ['relu', 'identity', 'logistic', 'tanh']
hidden_layer_sizes = [(10), (20), (10, 10), (20, 20)]

# List to store results
results = []

# Loop through each activation function and hidden layer size combination
for activation in activation_functions:
    for hidden_layers in hidden_layer_sizes:
        # Initialize the MLPClassifier with the current settings
        clf = MLPClassifier(
            random_state=1,
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            solver='adam',
            alpha=0.00001,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=1000,
            shuffle=True,
            tol=0.0001,
            early_stopping=False,
            n_iter_no_change=10
        )

        # Fit the model
        clf.fit(x_train, y_train)

        # Calculate cross-validated performance metrics
        accuracy = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='accuracy').mean()
        precision = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='precision').mean()
        recall = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='recall').mean()
        f1 = cross_val_score(clf, df_feat, df['target'], cv=10, scoring='f1').mean()

        # Store the results
        results.append({
            'Activation': activation,
            'Hidden Layers': hidden_layers,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Iterations': clf.n_iter_,
            'Loss Curve': clf.loss_curve_
        })

        # Print results for this configuration
        print(f"Activation: {activation}, Hidden Layers: {hidden_layers}")
        print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

        # Plot loss curve for this configuration
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(clf.loss_curve_)), clf.loss_curve_, color='blue', linestyle='solid', marker='.',
                 markerfacecolor='red', markersize=1)
        plt.title(f'Loss vs. Iteration (Activation: {activation}, Layers: {hidden_layers})')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.show()

# Overall plot of accuracy vs. hidden layer sizes for each activation function
plt.figure(figsize=(10, 6))

for activation in activation_functions:
    accuracies = [res['Accuracy'] for res in results if res['Activation'] == activation]
    plt.plot([str(hidden_layer) for hidden_layer in hidden_layer_sizes], accuracies, label=activation)

plt.title('Overall Accuracy vs. Hidden Layer Size')
plt.xlabel('Hidden Layer Sizes')
plt.ylabel('Accuracy')
plt.legend(title='Activation Function')
plt.grid(True)
plt.show()
