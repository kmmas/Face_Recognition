import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from PCA import PCA
from LDA import LDA



def get_projected_test(test_data, eigen_faces):
    centered_test_images = test_data - np.mean(test_data,axis=0)
    return np.dot(centered_test_images,eigen_faces)

def get_projected_train(train_data, eigen_faces):
    centered_train_images = train_data - np.mean(train_data,axis=0)
    return np.dot(centered_train_images,eigen_faces)

def knn_classifier(X_train, labels_train, X_test, labels_test, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, labels_train)
    labels_pred = knn.predict(X_test)
    accuracy = accuracy_score(labels_test, labels_pred)
    return accuracy
#***************************** Main Part*****************************************
k_values = [1, 3, 5, 7]
pca_accuracies = []
lda_accuracies = []

pca = PCA()
lda = LDA()

face_images = pca.read_faces("Dataset") # WARNING!!!! CHANGE PATH
pca.concatenate_images(face_images)
pca.form_labels()
training, testing, train_label, test_label = pca.split_matrix()

for k in k_values :
    eigen_faces_pca= pca.pca(training,0.8)
    projected_train_pca=get_projected_train(training,eigen_faces_pca)
    projected_test_pca = get_projected_test(testing,eigen_faces_pca)
    pca_accuracy = knn_classifier(projected_train_pca, train_label, projected_test_pca, test_label, k)
    pca_accuracies.append(pca_accuracy)
    
    U = lda.LDA(training)
    projected_train_lda = np.dot(training, U)
    projected_test_lda = np.dot(testing, U)
    lda_accuracy = knn_classifier(projected_train_lda, train_label, projected_test_lda, test_label, k)
    lda_accuracies.append(lda_accuracy)
plt.plot(k_values, pca_accuracies, label='PCA accuracy')
plt.plot(k_values, lda_accuracies, label='LDA accuracy')
plt.title('Performance Measure (Accuracy) vs. K Value')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.legend()
plt.grid(True)
plt.show()