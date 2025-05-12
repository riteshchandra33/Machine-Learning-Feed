import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

df = pd.read_csv('/Users/riteshchandra/pima-indians-diabetes.csv', index_col=0)
feature_names = df.columns[:-1]
print(df.head())

# Standardize the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('target', axis=1))
StandardScaler(copy=True, with_mean=True, with_std=True)
scaled_features = scaler.transform(df.drop('target', axis=1))
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
print(df_feat.head())

import seaborn as sns
sns.pairplot(df, hue='target')
plt.show()

# Apply Least Squares
from sklearn import linear_model
# Ordinary least squares
clf = linear_model.LinearRegression()

clf = clf.fit(x_train, y_train)

# Predictions
predictions_test = clf.predict(x_test)
class_names = [0, 1]
predictions_test[predictions_test <= 0.5] = 0
predictions_test[predictions_test > 0.5] = 1
predictions_test = predictions_test.astype(int)

# Display confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test, predictions_test, labels=class_names)
confusion_matrix_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=class_names)
confusion_matrix_display.plot()
plt.show()

# Report Overall Accuracy, precision, recall, F1-score
print(metrics.classification_report(
    y_true=y_test,
    y_pred=predictions_test,
    target_names=list(map(str, class_names)),
    zero_division=0 # Whenever number is divided by zero, instead of nan, return 0
))

