import numpy as np
from PIL import Image
import cv2,os
class dataset :
    def __init__(self,dataset_matrix = None, labels = None) -> None:
        self.dataset_matrix = dataset_matrix
        self.labels = labels

    def read_faces(self, data_path, classes = 40, imgpclass = 10):
        face_images = np.empty((0, 112*92), dtype=int)
        for i in range(1, (classes+1)):
            for j in range(1, (imgpclass+1)):
                img  = Image.open(f"{data_path}/s{i}/{j}.pgm")
                # img = Image.open(f"Dataset/s{i}/{j}.pgm")
                imgMatrix = np.array(img)
                imgMatrix1d = imgMatrix.flatten()
                face_images = np.vstack([face_images, imgMatrix1d])
        return face_images

    def read_non_faces(self,data_path, n) :
        image_files = [f for f in os.listdir(data_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        non_face_images = np.empty((0, 112*92), dtype=int)
        for image_file in image_files :
            image_path = os.path.join(data_path, image_file)
            img = img = cv2.imread(image_path)  
            resize_image = cv2.resize(img,(112,92))
            grayscale_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2GRAY)
            grayscale_image = np.array(grayscale_image)
            grayscale_image = grayscale_image.flatten()
            # print(grayscale_image.shape)
            non_face_images = np.vstack([non_face_images,grayscale_image])
        return non_face_images[0:n]

    def concatenate_images(self,faces,non_faces = None) :
        if non_faces is None :
            self.dataset_matrix = faces
        else :
            images = np.empty((0,112*92),dtype=int)
            images = np.vstack([images,faces,non_faces])
            self.dataset_matrix = images

    def split_matrix(self) :
        trainingMatrix = np.empty((0, 112*92), dtype=int)
        testingMatrix = np.empty((0, 112*92), dtype=int)
        trainingLabel = []
        testingLabel = []
        for i in range(0, len(self.labels)-1, 2):
            trainingMatrix = np.vstack([trainingMatrix, self.dataset_matrix[i+1]])
            trainingLabel.append(self.labels[i+1])
            testingMatrix = np.vstack([testingMatrix, self.dataset_matrix[i]])
            testingLabel.append(self.labels[i])
        return trainingMatrix, testingMatrix, trainingLabel, testingLabel
    
    def split_matrix_bonus(self, n_train = 5, n_test = 5) :
        trainingMatrix = np.empty((0, 112*92), dtype=int)
        testingMatrix = np.empty((0, 112*92), dtype=int)
        trainingLabel = []
        testingLabel = []
        for class_index in range(0,self.dataset_matrix.shape[0],10) :
            if(self.dataset_matrix.shape[0]-class_index >= 10) :
                index = 0
                while index < n_train :
                    trainingMatrix = np.vstack([trainingMatrix, self.dataset_matrix[class_index+index]])
                    trainingLabel.append(self.labels[class_index+index])
                    index +=1
                while index < n_train + n_test :
                    testingMatrix = np.vstack([testingMatrix, self.dataset_matrix[class_index+index]])
                    testingLabel.append(self.labels[class_index+index])
                    index +=1
            else : break
        return trainingMatrix, testingMatrix, trainingLabel, testingLabel
    
    def predict_check(self, actual, predicted) :
        correct = 0
        faces = 0
        non_faces = 0
        for i in range(len(actual)) :
            if actual[i] == predicted[i] : 
                correct += 1
                if predicted[i] == 1 :
                    non_faces += 1
                else :
                    faces += 1
        return correct,faces,non_faces
    def form_labels_with_non_faces(self,faces = 400,none_faces = 400) :
        super(dataset,self).__setattr__('labels', [0]*faces + [1]*none_faces)
    
if __name__ == "__main__" :
    s = dataset()
    print(f"correct: {s.predict_check([1,1,1,0,0,0,0], [1,1,0,1,0,0,1])}")