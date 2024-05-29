import matplotlib.pyplot as plt

# Fixed number of face images
num_face_images = 400

# Varying number of non-face images from 0 to 1000
num_non_face_images = [0,50,100,300,400,600,850,1000]

# Placeholder for accuracy values (you should replace this with your actual accuracy calculation)
accuracy_values = [1,0.9689,0.972,0.86,0.8675,0.846,0.8384,0.8329]

# Plotting
plt.plot(num_non_face_images, accuracy_values, marker='o', linestyle='-')

# Add labels and title
plt.xlabel('Number of Non-Face Images')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Non-Face Images')

# Show the plot
plt.show()