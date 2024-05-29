import numpy as np
from dataset import dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.naive_bayes import GaussianNB
# from sklearn.decomposition import PCA
# from Gauss import GaussianNaiveBayes
from please import GaussianNaiveBayes
from PCA import PCA

class FaceRecognitionNB(dataset):
    def __init__(self, dataset_matrix = None, labels = None) -> None:
        super().__init__(dataset_matrix, labels)
    def form_labels(self,classes=40) :
        super(dataset,self).__setattr__('labels', [i for i in range(1, classes+1) for _ in range(10)] ) 
    def split_data(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.dataset_matrix, self.labels, test_size=test_size, random_state=random_state)
    
    def train_naive_bayes(self):
        self.model = GaussianNaiveBayes()
        self.model.fit(self.X_train, self.y_train)
    
    # def predict(self, X):
    #     return self.model.predict(X)
    
    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print("Classification Accuracy:", accuracy)
        
        cm = confusion_matrix(self.y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)
        
        error_indices = np.where(y_pred != self.y_test)[0]
        print("\nError Cases:")
        for idx in error_indices:
            print(f"Predicted: {y_pred[idx]}, Actual: {self.y_test[idx]}")
    
    def apply_pca(self, n_components=40):
        pca = PCA()
        faces = pca.read_faces("Dataset")
        pca.concatenate_images(faces)
        pca.form_labels()
        self.X_train_pca, self.X_test_pca, self.y_train, self.y_test = pca.split_matrix_bonus(5,5)
        # self.pca = PCA(n_components=n_components)
        # self.X_train_pca = self.pca.fit_transform(self.X_train)
        # self.X_test_pca = self.pca.transform(self.X_test)
    
    def train_naive_bayes_with_pca(self):
        self.model_pca = GaussianNaiveBayes()
        self.model_pca.fit(self.X_train_pca, self.y_train)
    
    def evaluate_with_pca(self):
        y_pred_pca = self.model_pca.predict(self.X_test_pca)
        accuracy_pca = accuracy_score(self.y_test, y_pred_pca)
        print("\nClassification Accuracy with PCA:", accuracy_pca)
        
        cm_pca = confusion_matrix(self.y_test, y_pred_pca)
        print("Confusion Matrix with PCA:")
        print(cm_pca)
        
        error_indices_pca = np.where(y_pred_pca != self.y_test)[0]
        print("\nError Cases with PCA:")
        for idx in error_indices_pca:
            print(f"Predicted: {y_pred_pca[idx]}, Actual: {self.y_test[idx]}")

# Instantiate the class with your data
face_recognition = FaceRecognitionNB()
faces = face_recognition.read_faces("Dataset")
face_recognition.concatenate_images(faces)
face_recognition.form_labels()
# Split the data into train and test sets
face_recognition.split_data()

# Train the Naive Bayes classifier
face_recognition.train_naive_bayes()

# Evaluate the classifier
face_recognition.evaluate()

# Apply PCA
face_recognition.apply_pca()

# Train the Naive Bayes classifier with PCA
face_recognition.train_naive_bayes_with_pca()

# Evaluate the classifier with PCA
face_recognition.evaluate_with_pca()
