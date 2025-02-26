import numpy as np
from collections import Counter
import pandas as pd
from sklearn.metrics import accuracy_score


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        return self

    def predict(self, X_test):
        X_test = np.array(X_test)
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        # Öklid Hesabı
        distances = [np.sqrt(np.sum((x - x_train) ** 2))
                     for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


if __name__ == "__main__":
    data = pd.read_csv('ortopedik_hastaların_biyomekanik_özellikleri.csv')
    x_data = data.drop(['class'], axis=1)
    y_data = data['class'].values

    x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y_data, test_size=0.2, random_state=42
    )

    knn = KNN(k=3)
    knn.fit(x_train, y_train)
    predictions = knn.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.2f}")
    print(f"Predictions for test set: {predictions[:5]}")
    print(f"Actual values: {y_test[:5]}")
