from flask import Flask, flash, request, redirect, render_template,url_for
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import io
from PIL import Image
import base64
from ImgResizer import *

app = Flask(__name__)   

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg','jfif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/upload')
def upload_form():
	return render_template('upload.html')

   

@app.route('/predict', methods=['POST'])
def upload_image():
	images = []
	# For printing the value at cmd for tracking error...
	for file in request.files.getlist("file[]"):
		if file.filename == '':
			flash('No image selected for uploading')
			return redirect(request.url)
		
		# If input given	
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			filestr = file.read()
			npimg = np.frombuffer(filestr, np.uint8)

			#decoding the file
			image = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
			orig = image.copy()

			# Resizing the image
			image = ImgResizer.resize(image, height = 500)

			# Main work for the app
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			fm = cv2.Laplacian(gray, cv2.CV_64F).var()

			# Thresholding the value
			result = "Not Blurry"
			if fm < 100:
				result = "Blurry"

			# Formatting the result
			sharpness_value = "{:.0f}".format(fm)
			message = [result,sharpness_value]

			# changing the image from BGR2RGB
			img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			file_object = io.BytesIO()

			# Image resize for display
			img= Image.fromarray(ImgResizer.resize(img,width=500))
			img.save(file_object, 'PNG')

			# Image encoding to png
			base64img = "data:image/png;base64," + base64.b64encode(file_object.getvalue()).decode('ascii')
			images.append([message,base64img])

	print("images:", len(images))
	return render_template('prediction.html', images=images )
	

if __name__ == "__main__":
	app.run(debug=True) 
