import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


X = np.load('x_features.npy')
y = np.load('y_labels.npy')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


k_values = [3, 5, 7]

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, weights='uniform', metric='euclidean')
    knn.fit(X_train, y_train)

    train_accuracy = knn.score(X_train, y_train)
    test_accuracy = knn.score(X_test, y_test)

    print(f"K={k}: Train: {train_accuracy*100:.2f}%, Test: {test_accuracy*100:.2f}%")
