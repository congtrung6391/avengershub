# face detection for the 5 Celebrity Faces Dataset
import os
from os import listdir
from os.path import isdir
from PIL import Image
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from termcolor import colored

os.environ['KMP_DUPLICATE_LIB_OK']='True'
# create the detector, using default weights
detector = MTCNN()

# extract a single face from a given photograph
def extract_face(directory, filename, count, required_size=(160, 160)):
	# image path
	path = src + directory + filename
	# load image from file
	image = Image.open(path)
	# convert to RGB, if needed
	image = image.convert('RGB')
	# convert to array
	pixels = asarray(image)
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	try:
		for result in results:
			x1, y1, width, height = result['box']
			# bug fix
			x1, y1 = abs(x1), abs(y1)
			x2, y2 = x1 + width, y1 + height
			# extract the face
			face = pixels[y1:y2, x1:x2]
			# resize pixels to the model size
			imageFace = Image.fromarray(face)
			imageFace = imageFace.resize(required_size)
			count = count + 1
			imageFace.save(dest + directory + '/image{}.jpg'.format(count))
			print(colored('loaded image {}'.format(count), 'yellow'))
			if count > 300:
				return count
	except:
		print(colored('{} doesn\'t contains any face'.format(filename), 'red'))
	return count

# load images and extract faces for all images in a directory
def load_faces(directory):
	# enumerate files
	count = 0
	idimage = 0
	print(colored('loading directory: {}'.format(directory), 'green'))
	for filename in listdir(src+directory):
		# path
		path = src + directory + filename
		# store
		if(os.path.exists(dest + directory) == 0):
			os.mkdir(dest + directory)
		# get face
		count = face = extract_face(directory, filename, count)
		idimage = idimage + 1
		print(colored('loaded main image {}'.format(idimage), 'red'))
		if idimage == 150: 
			return
		# store
	print(colored('loaded directory: {}'.format(directory), 'green'))
	return 
 
# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
	X, y = list(), list()
	# enumerate folders, on per class
	for subdir in listdir(src+directory):
		# path
		path = directory + subdir + '/'
		# skip any files that might be in the dir
		if not isdir(src+path):
			continue
		# load all faces in the subdirectory
		load_faces(path)
	return
 
src = '../DatasetNotExtract/'
dest = '../Dataset/'
# load train dataset
load_dataset('train/')
# load test dataset
load_dataset('val/')