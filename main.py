from flask import request, redirect, render_template
import flask
import numpy as np
from numpy import load
from numpy import expand_dims
from numpy import asarray
from keras.models import load_model
import os
from os import listdir
from os.path import isdir
from PIL import Image
from mtcnn.mtcnn import MTCNN
import pickle
import tensorflow as tf
import json
import random

os.environ['KMP_DUPLICATE_LIB_OK']='True'

sess = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()

app = flask.Flask(__name__)
app.config["IMAGE_UPLOADS"] = "static/image/uploads/"

data = []
with open('dataJson.txt', 'r') as json_data:
	data = json.load(json_data)
print('loaded json file')

model_path = 'src/models/'

with sess.as_default():
	with graph.as_default():
		# load the facenet model
		facenetModel = load_model(model_path+'facenet_keras.h5')
		print('Loaded Facenet')
		# load the mtcnn model
		mtcnnModel = MTCNN()
		print('Loaded MTCNN')
		# load svm classifier
		svmModel = pickle.load(open(model_path+'svm.sav', 'rb'))
		print('Loaded svm')
		# load output encoder
		out_encoder = pickle.load(open(model_path+'encoder.pkl', 'rb'))
		print('Loaded encoder')

# get the face embedding for one face
def get_embedding(face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	with sess.as_default():
		with graph.as_default():
			yhat = facenetModel.predict(samples)
	#yhat = np.zeros((128, 128))
	return yhat[0]

# extract a single face from image
def extract_face(filename, required_size=(160, 160)):
	# load image from file
	image = Image.open(filename)
	# convert to RGB, if needed
	image = image.convert('RGB')
	# convert to array
	pixels = asarray(image)
	# detect faces in the image
	with sess.as_default():
		with graph.as_default():
			results = mtcnnModel.detect_faces(pixels)
	# extract the bounding box from the first face
	try:
		x1, y1, width, height = results[0]['box']
		# bug fix
		x1, y1 = abs(x1), abs(y1)
		x2, y2 = x1 + width, y1 + height
		# extract the face
		face = pixels[y1:y2, x1:x2]
		# resize pixels to the model size
		image = Image.fromarray(face)
	except:
		print('{} doesn\'t contains any face'.format(filename))
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

def classify(filename):
	# get face
	face = extract_face(filename)
	pixels = asarray(face)
	#get embed
	face_embed = get_embedding(pixels)
	sample = expand_dims(face_embed, axis=0)
	#sample = np.zeros((1, 128))
	# predict
	with sess.as_default():
		with graph.as_default():
			pred_class = svmModel.predict(sample)
			pred_prob = svmModel.predict_proba(sample)
			# get name
			class_id = pred_class[0]
			class_prob = pred_prob[0, class_id] * 100
			pred_name = out_encoder.inverse_transform([class_id])
			print('Predict: %s (%0.3f)' % (pred_name[0], class_prob))
	return pred_name[0]

@app.route('/', methods = ['POST', 'GET'])
def home():
	backgroundUrl = random.choice(os.listdir('static/image/background/Overall/'))
	return render_template("home.html", backgroundUrl = 'static/image/background/Overall/'+backgroundUrl)

@app.route('/upload-image', methods = ['POST', 'GET'])
def upload_image():
	if request.method == 'POST':
		actor = classify(request.files['image'])
		backgroundUrl = random.choice(os.listdir('static/image/background/{}/'.format(actor)))
		backgroundUrl = 'static/image/background/{}/'.format(actor)+backgroundUrl
	return render_template("info.html", data = data[actor][0], backgroundUrl = backgroundUrl)

#detectthread = detectThread(1, "facenet", 1)

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8000, debug=True, threaded=False)