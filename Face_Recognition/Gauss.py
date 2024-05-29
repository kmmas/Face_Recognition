import math
import numpy as np

class GaussianNaiveBayes:
    def fit(self, X_train, y_train):
        # Store training data and labels
        self.X_train = X_train
        self.y_train = y_train
        # Get unique classes in the training labels
        self.classes = np.unique(y_train)
        # Initialize dictionaries to store parameters and class priors
        self.parameters = {}
        self.class_prior = {}
        # Iterate over each class
        for c in self.classes:
            # Extract data for the current class
            X_c = X_train[y_train == c]
            # Calculate mean and standard deviation for each feature of the current class
            self.parameters[c] = {
                'mean': np.mean(X_c, axis=0),
                'std': np.std(X_c, axis=0)
            }
            # Calculate class prior probability
            self.class_prior[c] = len(X_c) / len(X_train)
    
    def calculate_probability(self, x, mean, std):
        # Small epsilon to avoid division by zero
        eps = 1e-20
        # Check if standard deviation is zero
        if std == 0:
            std += eps  # Add epsilon to avoid division by zero
        # Calculate the Gaussian probability density function
        exponent = np.exp(-(x - mean)**2 / (2 * (std**2 )))
        denominator = (1 / (np.sqrt(2 * np.pi) * (std )))
        prob = denominator * exponent
        if(prob == 0):
            prob += eps
        return  prob
    
    def calculate_class_probabilities(self, x):
        probabilities = {}
        # Iterate over each class
        for c in self.classes:
            # Initialize probability with class prior probability
            probabilities[c] = self.class_prior[c]
            # Iterate over each feature
            for i in range(len(self.parameters[c]['mean'])):
                # Extract mean and standard deviation for the current class and feature
                mean = self.parameters[c]['mean'][i]
                std = self.parameters[c]['std'][i]
                # Calculate the probability for the current feature
                probabilities[c] *= self.calculate_probability(x[i], mean, std)
                
        return probabilities
    
    def predict_single(self, x):
        # Calculate probabilities for each class
        probabilities = self.calculate_class_probabilities(x)
        best_label, best_prob = None, -1
        # Find the class with the highest probability
        for class_value, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_value
        return best_label
    
    def predict(self, X_test):
        # Predict the class for each sample in the test data
        y_pred = [self.predict_single(x) for x in X_test]
        return y_pred
