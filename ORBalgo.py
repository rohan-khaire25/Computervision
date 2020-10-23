import cv2
from matplotlib import pyplot as plt
import copy

#Reading the image
img = cv2.imread("face.jpeg")
#Convertng to RGB format
train_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#Convert to grayscale
train_gray = cv2.cvtColor(train_img, cv2.COLOR_RGB2GRAY)
#query image
query = cv2.imread("Team.jpeg")
query = cv2.cvtColor(query, cv2.COLOR_BGR2RGB)
#convert to grayscale
query_gray = cv2.cvtColor(query, cv2.COLOR_RGB2GRAY)

#show the image

#f, (ax1, ax2) = plt.subplots(1, 2)
#ax2.imshow(train_gray, cmap='gray')
#ax1.imshow(train_img)
#plt.show()

#Initializing ORB algorithm
#Set the number of features to be detected and the pyramid decimation ratio
orb = cv2.ORB_create(5000, 2.0)

#Find keypoints in the training image and query image. We are not using a mask here.
keypoints, descriptor = orb.detectAndCompute(train_gray, None)
q_keypoints, q_descriptor = orb.detectAndCompute(query_gray, None)
#creating copies of training image
copy_with_size = copy.copy(train_img)
copy_without_size = copy.copy(train_img)

#draw keypts without size and orientation
cv2.drawKeypoints(train_img, keypoints, copy_without_size, color = (0, 255, 0))
#draw keypts with size and orientation
cv2.drawKeypoints(train_img, keypoints, copy_with_size, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#show the images
#f, (ax3, ax4) = plt.subplots(1,2)
#ax3.imshow(copy_without_size)
#ax4.imshow(copy_with_size)
#plt.show()

#Feature matching
#creating a brute force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

# Perform the matching between the ORB descriptors of the training image and the query image
matches = bf.match(descriptor, q_descriptor)

# The matches with shorter distance are the ones we want. So, we sort the matches according to distance
matches = sorted(matches, key = lambda x : x.distance)

# Connect the keypoints in the training image with their best matching keypoints in the query image.
# We draw the first 85 mathces and use flags = 2 to plot the matching keypoints without size or orientation.
result = cv2.drawMatches(train_gray, keypoints, query_gray, q_keypoints, matches[:85], query_gray, flags = 2)
#show the result
print("Number of Keypoints Detected In The Training Image: ", len(keypoints))
print("Number of Keypoints Detected In The Query Image: ", len(q_keypoints))
plt.title("Best matching Points")
plt.imshow(result)
plt.show()