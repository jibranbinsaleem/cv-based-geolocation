
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.model_selection import train_test_split
import os
import webbrowser
import cv2

#init app and loading model
app = Flask(__name__)
my_model = load_model('C:/Users/jibra/pieas/icv/Model94%.h5')

ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

labels = ['Chapman', 'Cockcroft', 'Library', 'Maxwell', 'Media_City_Campus', 'New_Adelphi', 'New_Science', 'Newton', 'Sports_Center', 'University_House']
coords = [[53.48765075617312, -2.2748258018463816], [53.486144,-2.2732329],[53.487376, -2.272849], [53.485193939065006, -2.2706789766950846], 
          [53.473298621033294, -2.296998484648068], [53.48675453128888, -2.2745663476307167], [53.48837637970398, -2.274583715369131],
          [53.486126688337755, -2.2740041558451054],[53.48992703125414, -2.2734526278430414], [53.48917607578771, -2.2735923288503397]]


img_size = 224
def get_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)
print('here')
train = get_data("C:/Users/jibra/pieas/icv/uos/test_data")

train, val = train_test_split(train, test_size=0.2, random_state=25)


x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in train:
  x_train.append(feature)
  y_train.append(label)

for feature, label in val:
  x_val.append(feature)
  y_val.append(label)

# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)


 
print('here')
img_size = 224
def get_data_n(data_dir):
    data = [] 
    path = os.path.join(data_dir)
    for img in os.listdir(path):
        try:
            img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
            resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
            data.append([resized_arr])
        except Exception as e:
            print(e)
    return np.array(data)
print('here')


@app.route('/')
@app.route('/home', methods = ['GET', 'POST'])
def index():
    if request.method == 'GET':
        print("")
        data = ''
        return render_template('index.html', data=data)
    else:  #AFTER FILE SUBMISSION
        coords = [[53.48765075617312, -2.2748258018463816], [53.486144,-2.2732329],[53.487376, -2.272849], [53.485193939065006, -2.2706789766950846], 
          [53.473298621033294, -2.296998484648068], [53.48675453128888, -2.2745663476307167], [53.48837637970398, -2.274583715369131],
          [53.486126688337755, -2.2740041558451054],[53.48992703125414, -2.2734526278430414], [53.48917607578771, -2.2735923288503397]]

        print("")
        if 'file' not in request.files:
            return 'No file part'
        

        file = request.files['file']

        # Check if the file is empty
        if file.filename == '':
            return 'No selected file'

        if file and allowed_file(file.filename):
            filename = secure_filename("test" +'.' + file.filename.rsplit('.', 1)[1])
            filename = os.path.join("inputs", filename)
            file.save(filename)
        
        predictions = my_model.predict(x_val)
        Y_pred = predictions
        Y_pred_classes = np.argmax(predictions,axis = 1) 
        print(Y_pred_classes)
        print(y_val)
        print('as')


        img_size = 224
        test = get_data_n("C:/Users/jibra/pieas/interface_geoloc/inputs")
        print('done')
        x_test = []
        for feature in test:
            x_test.append(feature)
        x_test = np.array(x_test) / 255

        x_test.reshape(-1, img_size, img_size, 1)
        #print(x_test.shape)
        predictions = my_model.predict(x_test)
        Y_pred_classes = np.argmax(predictions,axis = 1) 
        print(Y_pred_classes)
        coordinates = coords[Y_pred_classes[0]]
        data = f"https://www.google.com/maps/search/?api=1&query={coordinates[0]},{coordinates[1]}"
        webbrowser.open(data, new=2)

        print(data)
        
        return redirect(url_for('index'))

@app.route('/test')
def test():

    predictions = my_model.predict(x_val)
    Y_pred = predictions
    Y_pred_classes = np.argmax(predictions,axis = 1) 
    print(Y_pred_classes)
    print(y_val)


    img_size = 224
    test = get_data_n("C:/Users/jibra/pieas/interface_geoloc/inputs")
    print('done')
    x_test = []
    for feature in test:
        x_test.append(feature)
    x_test = np.array(x_test) / 255

    x_test.reshape(-1, img_size, img_size, 1)
    #print(x_test.shape)
    predictions = my_model.predict(x_test)
    Y_pred_classes = np.argmax(predictions,axis = 1) 
    print(Y_pred_classes)
    return redirect(url_for('index'))




if __name__ == "__main__":
    app.run()

