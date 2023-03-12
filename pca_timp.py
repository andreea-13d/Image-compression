import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA
import matplotlib.image as mpimg
from datetime import datetime


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

image_raw = imread("LOGO_3000.png")
print(image_raw.shape)

# Displaying the image


image_sum = image_raw.sum(axis=2)


image_bw = image_sum/image_sum.max()



start = datetime.now()
pca = PCA()
pca.fit(image_bw)

# Getting the cumulative variance

var_cumu = np.cumsum(pca.explained_variance_ratio_)*100





# How many PCs explain 95% of the variance?
k = np.argmax(var_cumu>95)
#print("\n")




# plt.figure(figsize=[10,5])
# plt.title('Cumulative Explained Variance explained by the components')
# plt.ylabel('Cumulative Explained variance')
# plt.xlabel('Principal components')
# plt.axvline(x=k, color="k", linestyle="--")
# plt.axhline(y=95, color="r", linestyle="--")
# ax = plt.plot(var_cumu)

ipca = IncrementalPCA(n_components=k)
image_recon = ipca.inverse_transform(ipca.fit_transform(image_bw))

stop = datetime.now()

print("Number of components explaining 95% variance: "+ str(k))

print(f"Durata: {(stop-start)}")


#Nr pixeli            Timp
# 64x64              0.016 sec
# 100x100            0.033 sec
# 500x500            0.50 sec
# 1000x1000          1.85 sec
# 1500x1500          4.51 sec
# 2000x2000          8.55 sec
# 3000x3000          8.55 sec
# 4000x4000          8.55 sec
# 5000x5000          8.55 sec
