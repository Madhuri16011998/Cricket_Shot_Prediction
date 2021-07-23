#Import necessary libraries
from flask import Flask, render_template, request

import numpy as np
import os

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

filepath = 'C:/Users/Madhuri/AppData/Local/Programs/Python/Python38/Cricket_Shot_Prediction/model.h5'
model = load_model(filepath)
print(model)

print("Model Loaded Successfully")

def pred_cricket(cricket_shot):
  test_image = load_img(cricket_shot, target_size = (128, 128)) # load image 
  print("@@ Got Image for prediction")
  
  test_image = img_to_array(test_image)/255 # convert image to np array and normalize
  test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
  
  result = model.predict(test_image) # predict diseased palnt or not
  print('@@ Raw result = ', result)
  
  pred = np.argmax(result, axis=1)
  print(pred)
  if pred==0:
    return "Cricket - CUT SHOT",'Cricket-Cut_Shot.html'
       
  elif pred==1:
    return "Cricket - DRIVE SHOT",'Cricket-Drive.html'
        
  elif pred==2:
    return "Cricket - SWEEP SHOT",'Cricket-Sweep_Shot.html'
        

# Create flask instance
app = Flask(__name__)

# render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
        return render_template('index.html')
    
 
# get input image from client then predict class and render respective .html page for solution
@app.route("/predict", methods = ['GET','POST'])
def predict():
     if request.method == 'POST':
        file = request.files['image'] # fet input
        filename = file.filename        
        print("@@ Input posted = ", filename)
        
        file_path = os.path.join('C:/Users/Madhuri/AppData/Local/Programs/Python/Python38/Cricket_Shot_Prediction/static/upload/', filename)
        file.save(file_path)

        print("@@ Predicting class......")
        pred, output_page = pred_cricket(cricket_shot=file_path)
              
        return render_template(output_page, pred_output = pred, user_image = file_path)
    
# For local system & cloud
if __name__ == "__main__":
    app.run(threaded=False,port=8080) 
    
    
