import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
	# Loading and displaying image
	path=input('Enter the full name of image:')
	try:
		img=cv2.imread(path)
	except:
		print('Image not found')
		return

	img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	plt.title('Original image')
	plt.imshow(img)
	plt.axis('off')
	plt.show()

	# Flatening the image
	all_pixels=img.reshape((-1, 3))
	all_pixels=np.array(all_pixels)
	dominant_colors=int(input('Enter the number of dominant colors:'))

	from sklearn.cluster import KMeans
	km=KMeans(n_clusters=dominant_colors)
	km.fit(all_pixels)

	# Cluster centers
	centers=km.cluster_centers_
	centers=np.array(centers,dtype='uint8')

	plt.figure(0, figsize=(8, 2))
	for i, color in enumerate(centers):
	    plt.subplot(1, dominant_colors, i+1)
	    plt.axis('off')
	    # Color swatch
	    a=np.zeros((100, 100, 3), dtype='uint8')
	    a[:, :, :]=color
	    plt.imshow(a)
	plt.show()

	# Segmentation of original image
	new_img=np.zeros((all_pixels.shape), dtype='uint8')
	labels=km.labels_
	for i in range(new_img.shape[0]):
	    new_img[i]=centers[labels[i]]
	new_img=new_img.reshape(img.shape)
	plt.title('Segmented image')
	plt.imshow(new_img)
	plt.axis('off')
	plt.show()

	# Saving the image
	img_name='segmented(k='+str(dominant_colors)+')'+path.split(sep='/')[-1]
	new_img=cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
	cv2.imwrite(img_name, new_img)
	print('Saved segmented image:', img_name)

if __name__ == '__main__':
	main()