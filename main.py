import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# load dataset
df = pd.read_csv("path/to/iris.csv")
X = df[['sepal_length','sepal_width','petal_length','petal_width']]
y = df['species']

# split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

df.isnull().sum()

# the count of each species
y.value_counts()
sns.countplot(x=df['species'], hue=df['species'])

# visualize dataset
plt.figure(figsize=(8, 6))
sns.pairplot(df, hue='species', height=2)
plt.show()

# support vector machine (SVM)
svm_classifier = SVC(kernel = 'linear', C=1.0, random_state=42)
svm_classifier.fit(X_train, y_train)
y_pred_svm = svm_classifier.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

# k-nearest neighbors (KNN)
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

# decision tree
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)

# print accuracies of each model
print(f"SVM Accuracy: {accuracy_svm *100:.2f}%")
print(f"KNN Accuracy: {accuracy_knn *100:.2f}%")
print(f"Decision Tree Accuracy: {accuracy_dt *100:.2f}%")

results_df = pd.DataFrame({
    'Model': ['SVM', 'KNN', 'Decision Tree'],
    'Accuracy': [accuracy_svm, accuracy_knn, accuracy_dt]
})

# plot bar graph
plt.figure(figsize=(6, 6))
sns.barplot(x='Model', y='Accuracy', data=results_df, palette='viridis')
plt.title('Accuracy of Different Models')
plt.ylim(0, 1)
plt.show()




