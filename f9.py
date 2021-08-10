import os
import time
import cv2
from flask import Flask, flash, render_template, request
# please note the import from `flask_uploads` - not `flask_reuploaded`!!
# this is done on purpose to stay compatible with `Flask-Uploads`
from flask_uploads import IMAGES, UploadSet, configure_uploads
from keras.models import load_model
import numpy as np
from keras.preprocessing import image

app = Flask(__name__)
photos = UploadSet("photos", IMAGES)
app.config["UPLOADED_PHOTOS_DEST"] = "uploads/images"
app.config["SECRET_KEY"] = os.urandom(24)
configure_uploads(app, photos)


@app.route("/", methods=['GET', 'POST'])
def upload():

    return render_template('upload.html')

# form will submit here
@app.route("/detect", methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        f = request.files['photo']
        photos.save(f)
        print(f.filename)

        cf=read_image("./uploads/images/"+f.filename)
        cf = cv2.resize(cf, (150, 150))
        cv2.imwrite("./templates/a.jpg", cf)
        p=test_mask('facemask_cnn_model-1.h5','./templates/a.jpg')
        flash(p)
        # storing the log
        l=[f.filename,p]
        readcsv(l)
    return render_template('upload.html')

def read_image(img):
    import cv2
    import glob
    import os
    import matplotlib.pyplot as plt
    import sys
    image = cv2.imread(img)
    return image

def face_cut(img,outputFolder) -> object:
    """

    :rtype: object
    """
    import cv2
    import glob
    import os
    import matplotlib.pyplot as plt
    import sys
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt2.xml')
    image=cv2.imread(img)
    gray_img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray_img, scaleFactor = 1.20 , minNeighbors=4)
    print("[INFO] Found {0} Faces!".format(len(faces)))
    # mark and write the image
    for x,y,w,h in faces:
        img4=cv2.rectangle(image, (x,y),(x+w,y+h),(0,255,0),3)
        cf = image[y:y + h, x:x + w]
        #imgResized=cv2.resize(image,(150,150))
        #cv2.imwrite(outputFolder+img,cf)
        #plt.imshow(cv2.cvtColor(cf, cv2.COLOR_BGR2RGB))
        #plt.axis('off')
        #plt.show()
        print(cf.shape)
        return cf


def test_mask(model, testimage):
    classifier = load_model(model)
    #training_set.class_indices
    # importing images
    test_img = image.load_img(testimage, target_size=(150, 150))
    # converting image to array
    test_img = image.img_to_array(test_img)
    test_img = np.expand_dims(test_img, axis=0)
    result = classifier.predict(test_img)
    if result[0][0] >= 0.5:
        prediction = ' Picture without mask'
    else:
        prediction = 'Picture with mask'
    return prediction

def readcsv(List):
    from csv import writer

# List


# Open our existing CSV file in append mode
# Create a file object for this file
    with open('record.csv', 'a') as f_object:
    # Pass this file object to csv.writer()
    # and get a writer object
        writer_object = writer(f_object)

    # Pass the list as an argument into
    # the writerow()
        writer_object.writerow(List)

    # Close the file object
    f_object.close()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=80)
