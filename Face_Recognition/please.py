import numpy as np

class GaussianNaiveBayes:
    def __init__(self):
        self.class_probs = None
        self.class_means = None
        self.class_variances = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Initialize dictionaries to store class probabilities, means, and variances
        self.class_probs = {}
        self.class_means = {}
        self.class_variances = {}

        # Calculate class probabilities
        for c in self.classes:
            X_c = X[y == c]
            self.class_probs[c] = len(X_c) / n_samples

            # Calculate mean and variance for each feature in each class
            self.class_means[c] = np.mean(X_c, axis=0)
            self.class_variances[c] = np.var(X_c, axis=0)

    def predict(self, X):
        y_pred = [self._predict_sample(x) for x in X]
        return np.array(y_pred)

    def _predict_sample(self, x):
        posteriors = []

        # Calculate posterior probability for each class
        for c in self.classes:
            prior = np.log(self.class_probs[c])
            posterior = prior + np.sum(np.log(self._pdf(c, x)))
            posteriors.append(posterior)

        # Return class with highest posterior probability
        return self.classes[np.argmax(posteriors)]

    def _pdf(self, class_label, x):
        mean = self.class_means[class_label]
        variance = self.class_variances[class_label]
        variance[variance == 0 ] += 1e-20
        numerator = np.exp(-((x - mean) ** 2) / (2 * variance))
        denominator = np.sqrt(2 * np.pi * variance)
        prob = numerator / denominator
        prob[prob == 0] += 1e-20
        return prob