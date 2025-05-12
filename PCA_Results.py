import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, Lasso, Ridge

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

# Reduce the number of features to 2, so you can plot them
pca = PCA(n_components=2) # Create a PCA Object that will generate two features from the existing features
pca = pca.fit(x_train) # Fit PCA to training data
x_train_2 = pca.transform(x_train)
x_test_2 = pca.transform(x_test)

# Apply SVM
clf_pca = SVC(C=1.0, kernel='linear', degree=3)
clf_pca = clf_pca.fit(x_train_2, y_train)
predictions_test = clf_pca.predict(x_test_2)

# Scatter plot
y_train = y_train.tolist()
plt.scatter(x_train_2[:, 0], x_train_2[:, 1], c=y_train, s=10, cmap='viridis')

# Create a mesh to plot in
first_dimension_min, first_dimension_max = x_train_2[:, 0].min() - 1, x_train_2[:, 0].max() + 1
second_dimension_min, second_dimension_max = x_train_2[:, 1].min() - 1, x_train_2[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(first_dimension_min, first_dimension_max, .01), np.arange(second_dimension_min, second_dimension_max, .01))
Z = clf_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
# Draw contour line
plt.contour(xx, yy, Z)
plt.title('SVM Linear Decision Surface')
plt.axis('off')
plt.show()

# Apply SVM with Polynomial Kernel
clf_poly = SVC(C=1.0, kernel='poly')
clf_poly = clf_poly.fit(x_train_2, y_train)
predictions_test_poly = clf_poly.predict(x_test_2)

# Scatter plot
plt.scatter(x_train_2[:, 0], x_train_2[:, 1], c=y_train, s=10, cmap='viridis')

# Create a mesh to plot in
first_dimension_min, first_dimension_max = x_train_2[:, 0].min() - 1, x_train_2[:, 0].max() + 1
second_dimension_min, second_dimension_max = x_train_2[:, 1].min() - 1, x_train_2[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(first_dimension_min, first_dimension_max, .01), np.arange(second_dimension_min, second_dimension_max, .01))
Z_poly = clf_poly.predict(np.c_[xx.ravel(), yy.ravel()])
Z_poly = Z_poly.reshape(xx.shape)

# Draw contour line
plt.contour(xx, yy, Z_poly)
plt.title('SVM with Polynomial Kernel Decision Surface')
plt.axis('off')
plt.show()

# Apply SVM with Sigmoid Kernel
clf_sig = SVC(C=1.0, kernel='sigmoid')
clf_sig = clf_sig.fit(x_train_2, y_train)
predictions_test_sig = clf_sig.predict(x_test_2)

# Scatter plot
plt.scatter(x_train_2[:, 0], x_train_2[:, 1], c=y_train, s=10, cmap='viridis')

# Create a mesh to plot in
first_dimension_min, first_dimension_max = x_train_2[:, 0].min() - 1, x_train_2[:, 0].max() + 1
second_dimension_min, second_dimension_max = x_train_2[:, 1].min() - 1, x_train_2[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(first_dimension_min, first_dimension_max, .01), np.arange(second_dimension_min, second_dimension_max, .01))
Z_sig = clf_sig.predict(np.c_[xx.ravel(), yy.ravel()])
Z_sig = Z_sig.reshape(xx.shape)

# Draw contour line
plt.contour(xx, yy, Z_sig)
plt.title('SVM with Sigmoid Kernel Decision Surface')
plt.axis('off')
plt.show()



# Apply SVM with RBF Kernel
clf_rbf = SVC(C=1.0, kernel='rbf', gamma='scale')
clf_rbf = clf_rbf.fit(x_train_2, y_train)
predictions_test_rbf = clf_rbf.predict(x_test_2)

# Scatter plot
plt.scatter(x_train_2[:, 0], x_train_2[:, 1], c=y_train, s=10, cmap='viridis')

# Create a mesh to plot in
first_dimension_min, first_dimension_max = x_train_2[:, 0].min() - 1, x_train_2[:, 0].max() + 1
second_dimension_min, second_dimension_max = x_train_2[:, 1].min() - 1, x_train_2[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(first_dimension_min, first_dimension_max, .01), np.arange(second_dimension_min, second_dimension_max, .01))
Z_rbf = clf_rbf.predict(np.c_[xx.ravel(), yy.ravel()])
Z_rbf = Z_rbf.reshape(xx.shape)

# Draw contour line
plt.contour(xx, yy, Z_rbf)
plt.title('SVM with RBF Kernel Decision Surface')
plt.axis('off')
plt.show()
# Apply Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
clf_tree = DecisionTreeClassifier(random_state=42)
clf_tree.fit(x_train_2, y_train)
predictions_test_tree = clf_tree.predict(x_test_2)

# Scatter plot for Decision Tree
plt.scatter(x_train_2[:, 0], x_train_2[:, 1], c=y_train, s=10, cmap='viridis')

# Decision boundary for Decision Tree
Z_tree = clf_tree.predict(np.c_[xx.ravel(), yy.ravel()])
Z_tree = Z_tree.reshape(xx.shape)
plt.contour(xx, yy, Z_tree)
plt.title('Decision Tree Decision Surface')
plt.axis('off')
plt.show()

# Apply Random Forest CLassifier
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(x_train_2, y_train)
predictions_test_rf = clf_rf.predict(x_test_2)

# Scatter plot for Random Forest
plt.scatter(x_train_2[:, 0], x_train_2[:, 1], c=y_train, s=10, cmap='viridis')

# Decision boundary for Random Forest
Z_rf = clf_rf.predict(np.c_[xx.ravel(), yy.ravel()])
Z_rf = Z_rf.reshape(xx.shape)
plt.contour(xx, yy, Z_rf)
plt.title('Random Forest Decision Surface')
plt.axis('off')
plt.show()

# Apply Logistic Regression
from sklearn.linear_model import LogisticRegression
clf_lr = LogisticRegression(random_state=42)
clf_lr.fit(x_train_2, y_train)
predictions_test_lr = clf_lr.predict(x_test_2)

# Scatter plot for Logistic Regression
plt.scatter(x_train_2[:, 0], x_train_2[:, 1], c=y_train, s=10, cmap='viridis')

# Decision boundary for Logistic Regression
Z_lr = clf_lr.predict(np.c_[xx.ravel(), yy.ravel()])
Z_lr = Z_lr.reshape(xx.shape)
plt.contour(xx, yy, Z_lr)
plt.title('Logistic Regression Decision Surface')
plt.axis('off')
plt.show()

# Apply Naïve Bayes Classifier
from sklearn.naive_bayes import GaussianNB
clf_nb = GaussianNB()
clf_nb.fit(x_train_2, y_train)
predictions_test_nb = clf_nb.predict(x_test_2)

# Scatter plot for Naïve Bayes
plt.scatter(x_train_2[:, 0], x_train_2[:, 1], c=y_train, s=10, cmap='viridis')

# Decision boundary for Naïve Bayes
Z_nb = clf_nb.predict(np.c_[xx.ravel(), yy.ravel()])
Z_nb = Z_nb.reshape(xx.shape)
plt.contour(xx, yy, Z_nb)
plt.title('Naïve Bayes Decision Surface')
plt.axis('off')
plt.show()

# Apply K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
clf_knn = KNeighborsClassifier(n_neighbors=5)
clf_knn.fit(x_train_2, y_train)
predictions_test_knn = clf_knn.predict(x_test_2)

# Scatter plot for K-Nearest Neighbors
plt.scatter(x_train_2[:, 0], x_train_2[:, 1], c=y_train, s=10, cmap='viridis')

# Decision boundary for K-Nearest Neighbors
Z_knn = clf_knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z_knn = Z_knn.reshape(xx.shape)
plt.contour(xx, yy, Z_knn)
plt.title('K-Nearest Neighbors Decision Surface')
plt.axis('off')
plt.show()

## Apply Ordinary Least Squares
clf_ols = LinearRegression()
clf_ols.fit(x_train_2, y_train)
predictions_test_ols = clf_ols.predict(x_test_2)
class_names = [0, 1]
predictions_test_ols[predictions_test_ols <= 0.5] = 0
predictions_test_ols[predictions_test_ols > 0.5] = 1
predictions_test_ols = predictions_test_ols.astype(int)


#y_train = y_train.tolist()
plt.scatter(x_train_2[:, 0], x_train_2[:, 1], c=y_train, s=10, cmap='viridis')

first_dimension_min, first_dimension_max = x_train_2[:, 0].min() - 1, x_train_2[:, 0].max() + 1
second_dimension_min, second_dimension_max = x_train_2[:, 1].min() - 1, x_train_2[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(first_dimension_min, first_dimension_max, .01),
                     np.arange(second_dimension_min, second_dimension_max, .01))

# Predict values for mesh grid
Z_clf_ols = clf_ols.predict(np.c_[xx.ravel(), yy.ravel()])
Z_clf_ols[Z_clf_ols <= 0.5] = 0
Z_clf_ols[Z_clf_ols > 0.5] = 1
Z_clf_ols = Z_clf_ols.astype(int)
Z_clf_ols= Z_clf_ols.reshape(xx.shape)

# Draw contour line
plt.contour(xx, yy, Z_clf_ols)
plt.title('Ordinary Least Squares Classifier')
plt.axis('off')
plt.show()



# Apply Ridge Regression
clf_ridge = Ridge(alpha=.5, # Regularization strength: Larger values specify stronger regularization.
                 random_state=42, # # The seed of the pseudo random number generator to use when shuffling the data.
                 )
clf_ridge.fit(x_train_2, y_train)
predictions_test_ridge = clf_ridge.predict(x_test_2)
class_names = [0, 1]
predictions_test_ridge[predictions_test_ridge <= 0.5] = 0
predictions_test_ridge[predictions_test_ridge > 0.5] = 1
predictions_test_ridge = predictions_test_ridge.astype(int)

#y_train = y_train.tolist()
plt.scatter(x_train_2[:, 0], x_train_2[:, 1], c=y_train, s=10, cmap='viridis')

first_dimension_min, first_dimension_max = x_train_2[:, 0].min() - 1, x_train_2[:, 0].max() + 1
second_dimension_min, second_dimension_max = x_train_2[:, 1].min() - 1, x_train_2[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(first_dimension_min, first_dimension_max, .01),
                     np.arange(second_dimension_min, second_dimension_max, .01))

# Predict values for mesh grid
Z_clf_ridge = clf_ridge.predict(np.c_[xx.ravel(), yy.ravel()])
Z_clf_ridge [Z_clf_ridge  <= 0.5] = 0
Z_clf_ridge [Z_clf_ridge  > 0.5] = 1
Z_clf_ridge  = Z_clf_ridge .astype(int)
Z_clf_ridge  = Z_clf_ridge .reshape(xx.shape)

# Draw contour line
plt.contour(xx, yy, Z_clf_ridge )
plt.title('Ridge Regression Classifier')
plt.axis('off')
plt.show()



# Apply Lasso Regression
clf_lasso =Lasso(alpha=0.1, # Regularization strength: Larger values specify stronger regularization. Default is 1.
                     random_state=42, # The seed of the pseudo random number generator that selects a random feature to update.
                     )
clf_lasso.fit(x_train_2, y_train)
predictions_test_lasso = clf_lasso.predict(x_test_2)
class_names = [0, 1]
predictions_test_lasso[predictions_test_lasso <= 0.5] = 0
predictions_test_lasso[predictions_test_lasso > 0.5] = 1
predictions_test_lasso = predictions_test_lasso.astype(int)

#y_train = y_train.tolist()
plt.scatter(x_train_2[:, 0], x_train_2[:, 1], c=y_train, s=10, cmap='viridis')

first_dimension_min, first_dimension_max = x_train_2[:, 0].min() - 1, x_train_2[:, 0].max() + 1
second_dimension_min, second_dimension_max = x_train_2[:, 1].min() - 1, x_train_2[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(first_dimension_min, first_dimension_max, .01),
                     np.arange(second_dimension_min, second_dimension_max, .01))

# Predict values for mesh grid
Z_clf_lasso = clf_lasso.predict(np.c_[xx.ravel(), yy.ravel()])
Z_clf_lasso [Z_clf_lasso  <= 0.5] = 0
Z_clf_lasso [Z_clf_lasso  > 0.5] = 1
Z_clf_lasso = Z_clf_lasso .astype(int)
Z_clf_lasso = Z_clf_lasso.reshape(xx.shape)

# Draw contour line
plt.contour(xx, yy, Z_clf_lasso)
plt.title('Lasso Regression Classifier')
plt.axis('off')
plt.show()

