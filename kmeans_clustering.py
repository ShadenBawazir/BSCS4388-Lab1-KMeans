"""
BSCS 4388 - Computer Vision
Lab 1: Image Clustering - K-Means
Dataset 1: Generated data (make_blobs, 1000 samples, 4 centers)
Dataset 2: Real image (skimage.data.coffee)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from skimage import data

# ----------------------------------------------------------------------
# Dataset 1: Generated Data (1000 samples)
# ----------------------------------------------------------------------

# Step 1 - Generate the dataset
X, y_true = make_blobs(n_samples=1000, centers=4, random_state=42)

# Step 2 - Plot the dataset before applying K-Means
plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], c="gray", s=20)
plt.title("Generated Data - Before K-Means")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.tight_layout()
plt.savefig("kmeans_data_before.png", dpi=150)
plt.show()

# Step 3 - Apply K-Means (4 clusters)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

# Step 4 - Plot the dataset after applying K-Means
plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=20)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    c="red",
    marker="X",
    s=200,
    edgecolors="black",
    label="Cluster Centers",
)
plt.title("Generated Data - After K-Means (4 clusters)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.tight_layout()
plt.savefig("kmeans_data_after.png", dpi=150)
plt.show()

# ----------------------------------------------------------------------
# Dataset 2: Real Image (coffee)
# ----------------------------------------------------------------------

# Step 1 - Load the image
coffee = data.coffee()

# Step 2 - Show the image before applying K-Means
plt.figure(figsize=(8, 6))
plt.imshow(coffee)
plt.title("Coffee Image - Before K-Means")
plt.axis("off")
plt.tight_layout()
plt.savefig("kmeans_image_before.png", dpi=150)
plt.show()

# Step 3 - Apply K-Means on the pixel colours
h, w, c = coffee.shape
pixels = coffee.reshape(-1, 3)

kmeans_img = KMeans(n_clusters=4, random_state=42, n_init=10)
img_labels = kmeans_img.fit_predict(pixels)

# Step 4 - Replace every pixel with the colour of its cluster center
centers = kmeans_img.cluster_centers_.astype(np.uint8)
segmented = centers[img_labels].reshape(h, w, c)

plt.figure(figsize=(8, 6))
plt.imshow(segmented)
plt.title("Coffee Image - After K-Means (4 clusters)")
plt.axis("off")
plt.tight_layout()
plt.savefig("kmeans_image_after.png", dpi=150)
plt.show()
