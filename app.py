#pip install reportlab
#pip install pillow

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from PIL import Image
import telepot
import shutil
import os
import sys
from flask import Flask, render_template, request
import cv2  # working with, mainly resizing, images
import numpy as np  # dealing with arrays
import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import \
    tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import matplotlib.pyplot as plt

from utile import daugman, DiscreteCosineTransform, msg_encrypt, find_iris

from rubikencryptor.rubikencryptor import RubikCubeCrypto
from PIL import Image
from AUDIO import Encrypt, Decrypt
import fitz
a=0
b=0
a1=0
b1=0
key = 'key.txt'

def WaterMark(img, text):
    text = str(text)
    path = "static/original/"+img
    image = cv2.imread(path)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
    print(text_size[0])
    text_x = int(image.shape[1]/2) - int(text_size[0]/2)
    text_y = int(image.shape[0]/2) - int(text_size[1]/2)
    overlay = np.zeros_like(image, dtype=np.uint8)
    alpha = 0.1 
    cv2.putText(overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (185, 192, 201), 3)
    output = cv2.addWeighted(image, 1, overlay, alpha, 0)
    cv2.imwrite("static/watermarked/"+img, output)

def WaterMark1(img, text):
    text = str(text)
    path = "static/pdforiginal/"+img
    print('filepath', path)
    image = cv2.imread(path)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
    print(text_size[0])
    text_x = int(image.shape[1]/2) - int(text_size[0]/2)
    text_y = int(image.shape[0]/2) - int(text_size[1]/2)
    overlay = np.zeros_like(image, dtype=np.uint8)
    alpha = 0.1 
    cv2.putText(overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (185, 192, 201), 3)
    output = cv2.addWeighted(image, 1, overlay, alpha, 0)
    cv2.imwrite("static/watermarked/"+img, output)

# global b
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/audiopage')
def audiopage():
    return render_template('audio.html')

@app.route('/pdfpage')
def pdfpage():
    return render_template('PDF.html')

@app.route('/encrypt', methods=['GET', 'POST'])
def encrypt():
    if request.method == 'POST':
        fileName=request.form['filename1']
        fileName2=request.form['filename2']
        fileName3=request.form['filename3']
        
        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'iris-{}-{}.model'.format(LR, '2conv-basic')
    ##    MODEL_NAME='keras_model.h5'
        def process_verify_data():
            verifying_data = []
            path = "static/iris/"+fileName2
            img_num = fileName2.split('.')[0]
            print(img_num)
            global ie
            ie=img_num
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            verifying_data.append([np.array(img), img_num])
            np.save('verify_data1.npy', verifying_data)
            return verifying_data

        verify_data = process_verify_data()
        #verify_data = np.load('verify_data.npy')

        
        tf.compat.v1.reset_default_graph()
        #tf.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 2, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log1')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('model loaded!')


        fig = plt.figure()
        str_label1=" "
        for num, data in enumerate(verify_data):

            img_num = data[1]
            img_data = data[0]

            y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            # model_out = model.predict([data])[0]
            model_out = model.predict([data])[0]
            print(model_out)
            print('model {}'.format(np.argmax(model_out)))

            if np.argmax(model_out) == 1:
                str_label1 = 'Fake'
            elif np.argmax(model_out) == 0:
                str_label1 = 'Live'

        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'fingerprint-{}-{}.model'.format(LR, '2conv-basic')
    ##    MODEL_NAME='keras_model.h5'
        def process_verify_data():
            verifying_data = []
            path = "static/fingerprint/"+fileName3
            img_num = fileName3.split('.')[0]
            print(img_num)
            global fe
            fe=img_num
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            verifying_data.append([np.array(img), img_num])
            np.save('verify_data.npy', verifying_data)
            return verifying_data

        verify_data = process_verify_data()
        #verify_data = np.load('verify_data.npy')

        
        tf.compat.v1.reset_default_graph()
        #tf.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 2, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('model loaded!')


        fig = plt.figure()
        str_label=" "
        for num, data in enumerate(verify_data):

            img_num = data[1]
            img_data = data[0]

            y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            # model_out = model.predict([data])[0]
            model_out = model.predict([data])[0]
            print(model_out)
            print('model {}'.format(np.argmax(model_out)))

            if np.argmax(model_out) == 0:
                str_label = 'Fake'
            elif np.argmax(model_out) == 1:
                str_label = 'Live'

        if str_label == 'Live' and str_label1 == 'Live':
            input_image = "static/original/"+fileName
            output_image = "static/encrypted/"+fileName

            input_image = Image.open(input_image)
            rubixCrypto = RubikCubeCrypto(input_image)
            
            encrypted_image = rubixCrypto.encrypt(alpha=8, iter_max=10, key_filename=key)
            encrypted_image.save(output_image)

            import random
            otp2 = random.randint(1000,9999)
            otp2 = str(otp2)
            print(otp2)

            import telepot
            bot = telepot.Bot("7312910280:AAGi3SrW71DPse8F_iS-qtMh1otyQLv9kEs")
            bot.sendMessage("860233347", str(otp2))

            return render_template('home.html', result1 = fileName, otp2=otp2)
        else:
            return render_template('home.html', msg="fingerprint or iris does not match")
    return render_template('home.html')

@app.route('/verify1', methods=['GET', 'POST'])
def verify1():
    if request.method == 'POST':
        fileName=request.form['filename1']
        otp1=int(request.form['otp1'])
        otp2=int(request.form['otp2'])

        if otp1 == otp2:
            WaterMark(fileName, otp1)
            return render_template('home.html',  ImageDisplay1="static/watermarked/"+fileName, ImageDisplay2="static/encrypted/"+fileName)
        else:
            return render_template('home.html', msg="Entered wrong otp")

        return render_template('home.html')
    return render_template('home.html')

@app.route('/decrypt', methods=['GET', 'POST'])
def decrypt():
    if request.method == 'POST':
        fileName1=request.form['filename1']
        fileName2=request.form['filename2']
        fileName3=request.form['filename3']

        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'iris-{}-{}.model'.format(LR, '2conv-basic')
    ##    MODEL_NAME='keras_model.h5'
        def process_verify_data():
            verifying_data = []
            path = "static/iris/"+fileName2
            img_num = fileName2.split('.')[0]
            global a, b
            if img_num==ie:
                a=1
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            verifying_data.append([np.array(img), img_num])
            np.save('verify_data1.npy', verifying_data)
            return verifying_data

        verify_data = process_verify_data()
        #verify_data = np.load('verify_data.npy')

        
        tf.compat.v1.reset_default_graph()
        #tf.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 2, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log1')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('model loaded!')


        fig = plt.figure()
        str_label1=" "
        for num, data in enumerate(verify_data):

            img_num = data[1]
            img_data = data[0]

            y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            # model_out = model.predict([data])[0]
            model_out = model.predict([data])[0]
            print(model_out)
            print('model {}'.format(np.argmax(model_out)))

            if np.argmax(model_out) == 1:
                str_label1 = 'Fake'
            elif np.argmax(model_out) == 0:
                str_label1 = 'Live'

        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'fingerprint-{}-{}.model'.format(LR, '2conv-basic')
    ##    MODEL_NAME='keras_model.h5'
        def process_verify_data():
            verifying_data = []
            path = "static/fingerprint/"+fileName3
            img_num = fileName3.split('.')[0]
            global a, b
            if img_num==fe:
                b=1
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            verifying_data.append([np.array(img), img_num])
            np.save('verify_data.npy', verifying_data)
            return verifying_data

        verify_data = process_verify_data()
        #verify_data = np.load('verify_data.npy')

        
        tf.compat.v1.reset_default_graph()
        #tf.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 2, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('model loaded!')


        fig = plt.figure()
        str_label=" "
        for num, data in enumerate(verify_data):

            img_num = data[1]
            img_data = data[0]

            y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            # model_out = model.predict([data])[0]
            model_out = model.predict([data])[0]
            print(model_out)
            print('model {}'.format(np.argmax(model_out)))

            if np.argmax(model_out) == 0:
                str_label = 'Fake'
            elif np.argmax(model_out) == 1:
                str_label = 'Live'
        str_label = 'Live'
        str_label1 = 'Live'
        global a, b
        if str_label == 'Live' and str_label1 == 'Live' and a==1 and b==1:
            a=0
            b=0
            input_image = "static/encrypted/"+fileName1
            output_image = "static/decrypted/"+fileName1
            
            input_image = Image.open(input_image)
            rubixCrypto = RubikCubeCrypto(input_image)
            
            decrypted_image = rubixCrypto.encrypt(key_filename=key)
            decrypted_image.save(output_image)

            import random
            otp2 = random.randint(1000,9999)
            otp2 = str(otp2)
            print(otp2)

            
            
            bot = telepot.Bot("7312910280:AAGi3SrW71DPse8F_iS-qtMh1otyQLv9kEs")
            bot.sendMessage("860233347", str(otp2))

            return render_template('home.html', result= fileName1, otp2=otp2)
        else:
            
            bot = telepot.Bot("7312910280:AAGi3SrW71DPse8F_iS-qtMh1otyQLv9kEs")
            bot.sendMessage("860233347", str("UNAUTHORISED PERSON TRYING TO ACCESS THE DOCUMENTS"))
            return render_template('home.html', msg="fingerprint or iris does not match")

    return render_template('home.html')

@app.route('/verify', methods=['GET', 'POST'])
def verify():
    if request.method == 'POST':
        fileName1=request.form['filename1']
        otp1=int(request.form['otp1'])
        otp2=int(request.form['otp2'])

        if otp1 == otp2:
            return render_template('home.html', Image1="http://127.0.0.1:5000/static/encrypted/"+fileName1, Image2="http://127.0.0.1:5000/static/original/"+fileName1)
        else:
            return render_template('home.html', msg="Entered wrong otp")

        return render_template('home.html')
    return render_template('home.html')

@app.route('/audio_encrypt', methods=['GET', 'POST'])
def audio_encrypt():
    if request.method == 'POST':
        af=request.form['Input']
        string=request.form['Text']
        output=request.form['Output']
        msg = Encrypt('static/audio/'+af, string, 'static/result/'+output+'.wav')
        return render_template('audio.html', msg=msg)
    return render_template('audio.html')

@app.route('/audio_decrypt', methods=['GET', 'POST'])
def audio_decrypt():
    if request.method == 'POST':
        af=request.form['Input']
        msg1 = Decrypt('static/result/'+af)
        return render_template('audio.html', msg1=msg1, ad = 'static/result/'+af)
    return render_template('audio.html')





@app.route('/pdfencrypt', methods=['GET', 'POST'])
def pdfencrypt():
    if request.method == 'POST':
        pdffile=request.form['filename1']
        fileName2=request.form['filename2']
        fileName3=request.form['filename3']
        print(pdffile)
        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'iris-{}-{}.model'.format(LR, '2conv-basic')
    ##    MODEL_NAME='keras_model.h5'
        def process_verify_data():
            verifying_data = []
            path = "static/iris/"+fileName2
            img_num = fileName2.split('.')[0]
            global ie1
            ie1=img_num
            print("pdf name 1 : {} ".format(img_num))
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            verifying_data.append([np.array(img), img_num])
            np.save('verify_data1.npy', verifying_data)
            return verifying_data

        verify_data = process_verify_data()
        #verify_data = np.load('verify_data.npy')

        
        tf.compat.v1.reset_default_graph()
        #tf.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 2, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log1')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('model loaded!')


        fig = plt.figure()
        str_label1=" "
        for num, data in enumerate(verify_data):

            img_num = data[1]
            img_data = data[0]

            y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            # model_out = model.predict([data])[0]
            model_out = model.predict([data])[0]
            print(model_out)
            print('model {}'.format(np.argmax(model_out)))

            if np.argmax(model_out) == 1:
                str_label1 = 'Fake'
            elif np.argmax(model_out) == 0:
                str_label1 = 'Live'

        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'fingerprint-{}-{}.model'.format(LR, '2conv-basic')
    ##    MODEL_NAME='keras_model.h5'
        def process_verify_data():
            verifying_data = []
            path = "static/fingerprint/"+fileName3
            img_num = fileName3.split('.')[0]
            global fe1
            fe1=img_num
            print("pdf name 1 : {} ".format(img_num))
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            verifying_data.append([np.array(img), img_num])
            np.save('verify_data.npy', verifying_data)
            return verifying_data

        verify_data = process_verify_data()
        #verify_data = np.load('verify_data.npy')

        
        tf.compat.v1.reset_default_graph()
        #tf.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 2, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('model loaded!')


        fig = plt.figure()
        str_label=" "
        for num, data in enumerate(verify_data):

            img_num = data[1]
            img_data = data[0]

            y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            # model_out = model.predict([data])[0]
            model_out = model.predict([data])[0]
            print(model_out)
            print('model {}'.format(np.argmax(model_out)))

            if np.argmax(model_out) == 0:
                str_label = 'Fake'
            elif np.argmax(model_out) == 1:
                str_label = 'Live'

        if str_label == 'Live' and str_label1 == 'Live':
            pdf_file_path = "static/pdf/"+pdffile
            filename11 = pdffile.replace('.pdf', '.png')
            print(filename11)
            output_image_path = "static/pdforiginal/"+filename11
            doc = fitz.open(pdf_file_path)
            page = doc[0]  # Assuming you want to convert the first page
            pix = page.get_pixmap()
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            target_width = 512  # Example target width
            target_height = 512  # Example target height
            if image.width != target_width or image.height != target_height:
                image = image.resize((target_width, target_height))
            image.save(output_image_path)
            input_image = Image.open(output_image_path)
            rubixCrypto = RubikCubeCrypto(input_image)
            decrypted_image = rubixCrypto.encrypt(alpha=8, iter_max=10, key_filename=key)
            decrypted_image.save("static/pdfencrypted/"+filename11)

            import random
            otp2 = random.randint(1000,9999)
            otp2 = str(otp2)
            print(otp2)

            import telepot
            bot = telepot.Bot("7312910280:AAGi3SrW71DPse8F_iS-qtMh1otyQLv9kEs")
            bot.sendMessage("860233347", str(otp2))

            return render_template('PDF.html', result1 = filename11, otp2=otp2)
        else:
            return render_template('PDF.html', msg="fingerprint or iris does not match")
    return render_template('PDF.html')

@app.route('/pdfverify1', methods=['GET', 'POST'])
def pdfverify1():
    if request.method == 'POST':
        fileName=request.form['filename1']
        otp1=int(request.form['otp1'])
        otp2=int(request.form['otp2'])
        print(fileName)
        if otp1 == otp2:
            WaterMark1(fileName, otp1)
            return render_template('PDF.html',  ImageDisplay1="static/watermarked/"+fileName, ImageDisplay2="static/pdfencrypted/"+fileName)
        else:
            return render_template('PDF.html', msg="Entered wrong otp")

        return render_template('PDF.html')
    return render_template('PDF.html')

@app.route('/pdfdecrypt', methods=['GET', 'POST'])
def pdfdecrypt():
    if request.method == 'POST':
        fileName1=request.form['filename1']
        fileName2=request.form['filename2']
        fileName3=request.form['filename3']

        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'iris-{}-{}.model'.format(LR, '2conv-basic')
    ##    MODEL_NAME='keras_model.h5'
        def process_verify_data():
            verifying_data = []
            path = "static/iris/"+fileName2
            img_num = fileName2.split('.')[0]
            global a1, b1
            if img_num==ie1:
                a1=1
                print('iris macthed...')
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            verifying_data.append([np.array(img), img_num])
            np.save('verify_data1.npy', verifying_data)
            return verifying_data

        verify_data = process_verify_data()
        #verify_data = np.load('verify_data.npy')

        
        tf.compat.v1.reset_default_graph()
        #tf.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 2, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log1')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('model loaded!')


        fig = plt.figure()
        str_label1=" "
        for num, data in enumerate(verify_data):

            img_num = data[1]
            img_data = data[0]

            y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            # model_out = model.predict([data])[0]
            model_out = model.predict([data])[0]
            print(model_out)
            print('model {}'.format(np.argmax(model_out)))

            if np.argmax(model_out) == 1:
                str_label1 = 'Fake'
            elif np.argmax(model_out) == 0:
                str_label1 = 'Live'

        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'fingerprint-{}-{}.model'.format(LR, '2conv-basic')
    ##    MODEL_NAME='keras_model.h5'
        def process_verify_data():
            verifying_data = []
            path = "static/fingerprint/"+fileName3
            img_num = fileName3.split('.')[0]
            global a1, b1
            if img_num==fe1:
                b1=1
                print('fingerprint macthed...')
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            verifying_data.append([np.array(img), img_num])
            np.save('verify_data.npy', verifying_data)
            return verifying_data

        verify_data = process_verify_data()
        #verify_data = np.load('verify_data.npy')

        
        tf.compat.v1.reset_default_graph()
        #tf.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 2, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('model loaded!')


        fig = plt.figure()
        str_label=" "
        for num, data in enumerate(verify_data):

            img_num = data[1]
            img_data = data[0]

            y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            # model_out = model.predict([data])[0]
            model_out = model.predict([data])[0]
            print(model_out)
            print('model {}'.format(np.argmax(model_out)))

            if np.argmax(model_out) == 0:
                str_label = 'Fake'
            elif np.argmax(model_out) == 1:
                str_label = 'Live'
        global a1,b1
        if str_label == 'Live' and str_label1 == 'Live' and a1==1 and b1==1:
            a1=0
            b1=0
            input_image = "static/pdfencrypted/"+fileName1
            output_image = "static/pdfdecrypted/"+fileName1
            
            input_image = Image.open(input_image)
            rubixCrypto = RubikCubeCrypto(input_image)
            
            decrypted_image = rubixCrypto.decrypt(key_filename=key)
            decrypted_image.save(output_image)

            import random
            otp2 = random.randint(1000,9999)
            otp2 = str(otp2)
            print(otp2)

            import telepot
            bot = telepot.Bot("7312910280:AAGi3SrW71DPse8F_iS-qtMh1otyQLv9kEs")
            bot.sendMessage("860233347", str(otp2))

            return render_template('PDF.html', result= fileName1, otp2=otp2)
        else:
            return render_template('PDF.html', msg="fingerprint or iris does not match")

    return render_template('PDF.html')

@app.route('/pdfverify', methods=['GET', 'POST'])
def pdfverify():
    if request.method == 'POST':
        fileName1=request.form['filename1']
        otp1=int(request.form['otp1'])
        otp2=int(request.form['otp2'])
        fileName2 = fileName1
        fileName2 = fileName2.replace('.png', '.pdf')
        if otp1 == otp2:
            image_path = "static/pdfdecrypted/"+fileName1
            pdf_path = "static/pdfdecrypted/"+fileName2
            image = Image.open(image_path)
            c = canvas.Canvas(pdf_path, pagesize=letter)
            aspect_ratio = image.width / image.height
            width = 7.5 * inch
            height = width / aspect_ratio
            c.drawImage(image_path, 0, 0, width=width, height=height)
            c.save()
            return render_template('PDF.html', Image1="http://127.0.0.1:5000/static/pdfencrypted/"+fileName1, Image2="http://127.0.0.1:5000/"+pdf_path)
        else:
            return render_template('PDF.html', msg="Entered wrong otp")

        return render_template('PDF.html')
    return render_template('PDF.html')
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
