# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 03:26:57 2020

@author: jozic
"""
import streamlit as st 
import pandas as pd
import numpy as np
from PIL import Image
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

st.title('Pneumonia classification task')

st.markdown('Honours Project  **Demonstration**.')


uploaded_file = st.file_uploader("Choose an image...", type="jpeg")
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    #st.write("")
    st.write("Classifying...")
    #st.write(image)
   # image_array = np.array(image)
    dsize = (250,250)
    image = image.resize(dsize)
    image_array = np.array(image)
    image_array = image_array.astype('float32')
    image_array /=255.0

    #st.write(image_array.shape)
#    st.info("Model prediction:Pneumonia")
#    st.success("Pneumonia:89%")
#    st.warning("Normal:11%")
    
    
    json_file = open('saved_model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("saved_model/model.h5")
    
    X_test = image_array.reshape(1,250,250,1)
    print(X_test.shape)
    
    prediction = loaded_model.predict(X_test)
    
    class_prediction = loaded_model.predict_classes(X_test)
    
    if class_prediction[0]==0:
        st.info("Model Prediction:Healthy")
    else:
        st.info("Model Prediction:Pneumonia")
        
        


    pn = "Pnuemonia:"+str("{:.0%}".format(float(prediction[0][1])))
    normal = "Normal:"+str("{:.0%}".format(float(prediction[0][0])))
    st.success(pn)
    st.warning(normal)
    
    

    


