from dataset import dataset
import numpy as np
from helper import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import scipy.linalg as la

class LDA(dataset) :
    def __init__(self, dataset_matrix = None, labels = None) -> None:
        super().__init__(dataset_matrix, labels)
    
    def lda(self,points, label,n_components=39) :
        Sb = self.calculate_between_class_scatter(points, label)
        S  = self.calculate_within_class_scatter(points,label)
        S_inv = np.linalg.pinv(S)
        eigen_values, eigen_vectors = np.linalg.eig(S_inv.dot(Sb))
        eigen_values, eigen_vectors = order_eigens(eigen_values, eigen_vectors)
        U = eigen_vectors[:, :n_components]
        return np.real(U)

    def calculate_between_class_scatter(self,X, y):
        unique_classes = np.unique(y)
        num_features = X.shape[1]

        overall_mean = np.mean(X, axis=0)
        Sb = np.zeros((num_features, num_features))

        for c in unique_classes:
            # Select data points for the current class
            X_class = X[y == c]

            # Calculate the mean vector for the current class
            mean_vector = np.mean(X_class, axis=0)

            # Update Sb using the mean vector, overall mean, and the number of samples in the class
            Sb += len(X_class) * np.outer((mean_vector - overall_mean), (mean_vector - overall_mean))

        return Sb

    def calculate_within_class_scatter(self,X, y):
        unique_classes = np.unique(y)
        num_features = X.shape[1]

        Sw = np.zeros((num_features, num_features))

        for c in unique_classes:
            # Select data points for the current class
            X_class = X[y == c]
            X_class = centralize_data(X_class)
            # Calculate the covariance matrix for the current class
            cov_matrix = np.cov(X_class, rowvar=False)

            # Update Sw using the covariance matrix and the number of samples in the class
            Sw += (len(X_class)-1) * cov_matrix

        return Sw
    def LDA(self,train,n_components=39):
        data_reshaped = train.reshape((40, 5, 10304))
        means = np.mean(data_reshaped, axis=1)
        print (means.shape)
        print(means)
        overall_mean = np.mean(train, axis=0)
        Sb=np.zeros((10304, 10304))
        for i in range(40):
            Sb += 5* np.outer(means[i] - overall_mean, means[i] - overall_mean)
        S=np.zeros((10304, 10304)) 
        current_mean_index=0
        for i in range(0, len(train), 5):
            group = train[i:i+5] 
            centered_group = group - means[current_mean_index]
            current_mean_index=current_mean_index+1
            Z = np.array(centered_group)
            S += np.dot(Z.T,Z)
        S_inverse = la.pinv(S)
        Sw = np.dot(S_inverse,Sb)
        T, U = la.eig(Sw)
        U = np.real(U)
        sorted_indices = np.argsort(T)[::-1]
        sorted_eigenvectors = U[:, sorted_indices]
        eigen_faces =  sorted_eigenvectors[:, :39]
        print(eigen_faces)
        return eigen_faces

    def form_labels_with_faces(self, classes=40) :
        super(dataset,self).__setattr__('labels', [i for i in range(1, classes+1) for _ in range(10)])

if __name__ == "__main__":

    facesOrNot = input("faces(f)/nonfaces(n) : ")
    
    lda = LDA()
    print("processing datamatrix...")
    faces = lda.read_faces("Dataset")
    if(facesOrNot.lower().startswith("f")) :
        lda.concatenate_images(faces)
        lda.form_labels_with_faces()
        training, testing, train_label, test_label = lda.split_matrix()
    else :
        split_ratio = input("Enter split ratio train/test(e.g. 7 3) : ").split()
        non_faces = lda.read_non_faces("cars_vs_flowers/training_set/car", n = 1000)
        lda.concatenate_images(faces,non_faces)
        lda.form_labels_with_non_faces(faces.shape[0],non_faces.shape[0])
        training, testing, train_label, test_label = lda.split_matrix_bonus(int(split_ratio[0]),int(split_ratio[1]))
    print(f"training shape : {training.shape} and training label shape : {len(train_label)}")
    print(f"testing shape : {testing.shape} and testing label shape : {len(test_label)}")
    print("processing datamatrix done!")
    #U = lda.lda(training, train_label)
    U = lda.LDA(training,train_label)
    projected_data_train = projected_data_calculation(centralize_data(training),U)
    print("training is done!")
    print("testing...")
    projected_data_test = projected_data_calculation(centralize_data(testing),U)
    print("testing is done!")

    knn_classifier = KNeighborsClassifier(n_neighbors=1)
    knn_classifier.fit(projected_data_train, train_label)
    y_pred = knn_classifier.predict(projected_data_test)
    correct, faces_count, non_faces_count = lda.predict_check(test_label,y_pred)
    if facesOrNot == "n" :
        print(f"number of correct checks: {correct}/{len(test_label)}")
        print(f"faces predicted correctly : {faces_count}/{np.sum(np.array(test_label) == 0)}")
        print(f"non-faces predicted correctly : {non_faces_count}/{np.sum(np.array(test_label) == 1)}")
    accuracy = accuracy_score(test_label, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")