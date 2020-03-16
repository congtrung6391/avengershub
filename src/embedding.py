# calculate a face embedding for each face in the dataset using facenet
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model
import os
from PIL import Image
from termcolor import colored

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# load the facenet model
model = load_model('models/facenet_keras.h5')
print(colored('Loaded Model', 'green'))

# get the face embedding for one face
def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]

def load_embeddings(directory):
	embeddings = list()
	for filename in os.listdir(directory):
		# path
		path = directory + filename
		if not os.path.isfile(path): 
			continue
		# get face
		face = Image.open(path)
		# convert face to RGB color 
		face = face.convert('RGB')
		# conver face to pixels
		pixels = asarray(face)
		# get embedding of image
		embeddings.append(get_embedding(model, pixels))
	return embeddings

def load_dataset(directory):
	X, y = list(), list()
	# enumerate folders, on per class
	for subdir in os.listdir(directory):
		# path
		path = directory + subdir + '/'
		# skip any files that might be in the dir
		if not os.path.isdir(path):
			continue
		# load all faces in the subdirectory
		faces = load_embeddings(path)
		# create labels
		labels = [subdir for _ in range(len(faces))]
		# summarize progress
		print(colored('>loaded %d examples for class: %s' % (len(faces), subdir), 'green'))
		# store
		X.extend(faces)
		y.extend(labels)
	return asarray(X), asarray(y) 

# load train set
trainX, trainY = load_dataset('../Dataset/train/')
print(trainX.shape)
# load test set
testX, testY = load_dataset('../Dataset/val/')
print(testX.shape)
# save arrays to one file in compressed format
savez_compressed('10avengers.npz', trainX, trainY, testX, testY)