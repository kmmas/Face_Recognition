from dataset import dataset
import numpy as np
from helper import centralize_data,order_eigens,projected_data_calculation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class PCA(dataset) :
    def __init__(self, dataset_matrix = None, labels = None) -> None:
        super().__init__(dataset_matrix, labels)

    def pca(self,training,alpha) :
        Z = centralize_data(training)
        C = np.cov(Z,rowvar=False)

        eigen_values, eigen_vectors = np.linalg.eigh(C)
        eigen_values, eigen_vectors = order_eigens(eigen_values, eigen_vectors)

        eigenValuesSum = eigen_values.sum()
        
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

    def form_labels(self,classes=40) :
        super(dataset,self).__setattr__('labels', [i for i in range(1, classes+1) for _ in range(10)] ) 

if __name__ == "__main__":
    facesOrNot = input("faces(f)/nonfaces(n) : ")
    split_ratio = input("Enter split ratio train/test(e.g. 7 3) : ").split()
    pca = PCA()
    print("processing datamatrix...")
    faces = pca.read_faces("Dataset")
    if(facesOrNot.lower().startswith("f")) :
        pca.concatenate_images(faces)
        pca.form_labels()
    else :
        non_faces = pca.read_non_faces("cars_vs_flowers/training_set/car", n = 1000)
        pca.concatenate_images(faces,non_faces)
        pca.form_labels_with_non_faces(faces.shape[0],non_faces.shape[0])

    training, testing, train_label, test_label = pca.split_matrix_bonus(int(split_ratio[0]),int(split_ratio[1]))
    # training, testing, train_label, test_label = pca.split_matrix()
    print(f"training shape : {training.shape} and training label shape : {len(train_label)}")
    print(f"testing shape : {testing.shape} and testing label shape : {len(test_label)}")
    print("processing datamatrix done!")
    U = pca.pca(training,0.85)
    projected_data_train = projected_data_calculation(centralize_data(training),U)
    print("training is done!")
    print("testing...")
    projected_data_test = projected_data_calculation(centralize_data(testing),U)
    print("testing is done!")

    knn_classifier = KNeighborsClassifier(n_neighbors=1)
    knn_classifier.fit(projected_data_train, train_label)
    y_pred = knn_classifier.predict(projected_data_test)
    correct, faces_count, non_faces_count = pca.predict_check(test_label,y_pred)
    if facesOrNot == "n" :
        print(f"number of correct checks: {correct}/{len(test_label)}")
        print(f"faces predicted correctly : {faces_count}/{np.sum(np.array(test_label) == 0)}")
        print(f"non-faces predicted correctly : {non_faces_count}/{np.sum(np.array(test_label) == 1)}")
    accuracy = accuracy_score(test_label, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    