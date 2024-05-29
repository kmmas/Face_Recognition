from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import cv2
import os
nPerson = 40
nImages = 10
label = []

def pca(matrix) : 
    Z = centralize_data(matrix)
    C = np.cov(Z,rowvar=False)

    eigen_values, eigen_vectors = np.linalg.eigh(C)
    eigen_values, eigen_vectors = order_eigens(eigen_values, eigen_vectors)

    eigenValuesSum = eigen_values.sum()
    
    alpha = 0.70
    ratio = 0
    count = 0
    sum = 0
    for item in eigen_values :
            sum += item
            count += 1
            ratio = sum / eigenValuesSum
            if(ratio > alpha) :
                break          

    U = eigen_vectors[:, :count]
    return U

def lda(points, label):
    Z = centralize_data(points)
    Sb = calculate_between_class_scatter(points, label)
    S  = calculate_within_class_scatter(points,label)
    eigen_values, eigen_vectors = np.linalg.eigh(np.linalg.inv(S).dot(Sb))
    eigen_values, eigen_vectors = order_eigens(eigen_values, eigen_vectors)
    U = eigen_vectors[:, :39]
    return U

def calculate_between_class_scatter(X, y):
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


def calculate_within_class_scatter(X, y):
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


def centralize_data(data_matrix) :
    row_means = np.mean(data_matrix, axis=0)
    Z = data_matrix - row_means
    return Z

def projected_data_calculation(Z,U) :
    projected_data = np.dot(Z, U)
    return projected_data

def order_eigens(eigen_values,eigen_vectors) :
    sorted_indices = np.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[sorted_indices]
    eigen_vectors = eigen_vectors[:, sorted_indices]
    return eigen_values, eigen_vectors

def readDataset(dataPath):
    result_2d_array = np.empty((0, 112*92), dtype=int)
    label = []
    for i in range(1, nPerson+1):
        for j in range(1, nImages+1):
            label.append(i)
            img  = Image.open(f"{dataPath}/s{i}/{j}.pgm")
            # img = Image.open(f"Dataset/s{i}/{j}.pgm")
            imgMatrix = np.array(img)
            imgMatrix1d = imgMatrix.flatten()
            result_2d_array = np.vstack([result_2d_array, imgMatrix1d])
    return result_2d_array,label


def splitMatrix(dataMatrix,label) :
    trainingMatrix = np.empty((0, 112*92), dtype=int)
    testingMatrix = np.empty((0, 112*92), dtype=int)
    trainingLabel = []
    testingLabel = []
    for i in range(0, 30, 2):
        print(i)
        trainingMatrix = np.vstack([trainingMatrix, dataMatrix[i+1]])
        trainingLabel.append(label[i+1])
        testingMatrix = np.vstack([testingMatrix, dataMatrix[i]])
        testingLabel.append(label[i])
    return trainingMatrix, testingMatrix, trainingLabel, testingLabel


def resize_images(input_folder, output_folder, target_size=(92, 112)):
    # Create the output folder if it doesn't exist
    
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images = np.empty((0, 112*92), dtype=int)
    for image_file in image_files :
        image_path = os.path.join(input_folder, image_file)
        img = img = cv2.imread(image_path)  
        resize_image = cv2.resize(img,(112,92))
        grayscale_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2GRAY)
        grayscale_image = np.array(grayscale_image)
        grayscale_image = grayscale_image.flatten()
        # print(grayscale_image.shape)
        images = np.vstack([images,grayscale_image])
    return images







# for PCA ################################################
# print("processing datamatrix...")
# dataMatrix,label = readDataset("Dataset")
# training, testing, train_label, test_label = splitMatrix(dataMatrix,label)
# print("processing datamatrix done!")
# print("training...")
# eigen_faces = pca(training)
# projected_data_train = projected_data_calculation(centralize_data(training),eigen_faces)
# print("training is done!")
# print("testing...")
# projected_data_test = projected_data_calculation(centralize_data(testing),eigen_faces)
# print("testing is done!")
# knn_classifier = KNeighborsClassifier(n_neighbors=1)
# knn_classifier.fit(projected_data_train, train_label)

# # Predict labels for the test set
# y_pred = knn_classifier.predict(projected_data_test)
# accuracy = accuracy_score(test_label, y_pred)
# print(f"Accuracy: {accuracy * 100:.2f}%")


non_face_images = resize_images("cars_vs_flowers/training_set/car","output")[0:21]
non_face_labels = [5]*20
# plt.imshow(images[0], cmap='gray')
# plt.show()

# for LDA ################################################
print("processing datamatrix...")
dataMatrix,label = readDataset("Dataset")
faces_labels = [0]*20
print(f"size of nonfaces: {non_face_images.shape} and size of faces: {dataMatrix[0:21].shape}")
images = np.empty((0,112*92),dtype=int)
images = np.vstack([images,non_face_images,dataMatrix])
labels = non_face_labels + faces_labels
training, testing, train_label, test_label = splitMatrix(images,labels)
# training, testing, train_label, test_label = splitMatrix(dataMatrix,label)
print("processing datamatrix done!")
print("training...")
eigen_faces = lda(training,train_label)
print(f"Size of U : {eigen_faces.shape}")
projected_data_train = projected_data_calculation(centralize_data(training),eigen_faces)
print("training is done!")
print("testing...")
# projected_data_test =  projected_data_calculation(centralize_data(image),eigen_faces)
projected_data_test = projected_data_calculation(centralize_data(testing),eigen_faces)
print("testing is done!")

knn_classifier = KNeighborsClassifier(n_neighbors=1)
knn_classifier.fit(projected_data_train, train_label)


y_pred = knn_classifier.predict(projected_data_test)
print(f"y-pred: {y_pred}")
accuracy = accuracy_score(test_label, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")



# n_components = 10
# plt.figure(figsize=(12, 6))
# for i in range(n_components):
#     plt.subplot(2, n_components // 2, i + 1)
#     plt.imshow(eigen_faces[:, i].reshape(112,92), cmap='gray')
#     plt.title(f'Eigenface {i + 1}')
#     plt.axis('off')

# plt.tight_layout()
# plt.show()

