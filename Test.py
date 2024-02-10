# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from PIL import Image
from random import sample
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess
from tensorflow.keras.models import Model
from pathlib import Path
Image.LOAD_TRUNCATED_IMAGES = True
import requests
from io import BytesIO
import os
import streamlit as st
import pickle
import tensorflow as tf



model = pickle.load(open(r"C:\Users\SATYO\Music\DCD_21\resnet_model.pkl","rb"))
#input_file =  st.file_uploader("input file",type=["png","jpg"],accept_multiple_files= True)
#for image in input_file:
 #       st.image(image)

import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

# Load ResNet50 model
st.cache(allow_output_mutation=True)
def load_model():
    model = ResNet50(weights='imagenet')
    return model

# Function to preprocess the image
def preprocess_image(image):
    img = tf.image.decode_image(image.read(), channels=3)
    img = tf.image.resize(img, (224, 224))
    img = preprocess_input(img)
    img = tf.expand_dims(img, axis=0)
    return img

# Main function
def main():
    st.title("ResNet50 Image Classifier")
    st.write("Upload an image for classification")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = tf.image.decode_image(uploaded_file.read(), channels=3)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Load the model
        model = load_model()

        # Preprocess the image
        processed_image = preprocess_image(uploaded_file)

        # Make predictions
        predictions = model.predict(processed_image)
        decoded_predictions = decode_predictions(predictions, top=3)[0]

        # Display the predictions
        st.write("Predictions:")
        for i, (image, label, score) in enumerate(decoded_predictions):
            st.write(f"{i+1}: {label} ({score:.2f})")
            

if __name__ == '__main__':
    main()



#def display_image(image_path):
#    img = io.imread(image_path)
#    plt.imshow(img)
 #   plt.axis('off')
 #3   plt.show()
  #  return img

#image_path = (r"C:\Users\SATYO\Documents\GitHub\dcd21\model-img\model")
#image = display_image(image_path)

#st.code(model)

#st.image(model)
#st.pyplot(model)   

#st.image(display_resnet_results)



st.title("Image Similarity For E-Commerce")
st.subheader("Image Search")
st.markdown("---")

opt = st.sidebar.radio("",options=["Background","About Model","Input File", "URL Input","Biography"],)
#input_file =  st.file_uploader("input file",type=["png","jpg"],accept_multiple_files= True)




if opt == "Background":
    
    st.write("""### Background ###""")
    
    st.markdown(
    """
    <style>
        .justified-text {
            text-align: justify;
            text-justify: inter-word;
        }
    </style>
    """,
    unsafe_allow_html=True
)

    st.write(
    """
    E-Commercee menjadi sebuah wadah baru dalam melakukan transaksi jual beli,
    dengan perkembangan teknologi yang bertumbuh secara pesat,
    tidak menutup kemungkinan para pelaku usaha berlomba-lomba dalam mengembangkan inovasi produk mereka,
    dengan hal tersebut otomatis semakin banyak juga pelabelan atau penamaan produk, dimana para user
    akan kesulitan dalam mencari keyword produk yang ingin dicari, maka dari itu ide dalam pembuatan Image Similarity for E-Commerce muncul dimana
    prinsip kerjanya seperti layaknya search engine, dimana akan meningkatkan efektitas dan efisiensi user dalam mencari produk yang mereka cari.
    """,
    unsafe_allow_html=True
    )
    
elif opt == "About Model":
    st.write("""### About Model ###""")
    
    st.markdown(
     """
    <style>
        .justified-text {
            text-align: justify;
            text-justify: inter-word;
        }
    </style>
    """,
    unsafe_allow_html=True
    )
    st.write(
    """
    Project ini dirancang dengan model berupa Image Similarity dimana inputan image akan di proses menggunakan Neural Network.
    Kemudian image diproses dengan menyesuaikan pixel pada image yang berada pada dataset, machine akan mendeteksi pixel dan
    warna yang memiliki nilai yang sama dengan image inputan, dengan begitu hasil output akan menyesuaikan dengan hasil dari nilai inputan.
    Pada project ini inputan image berupa file directory/ file dari internal data dan berupa link URL,
    sehingga user dapat dengan mudah untuk menjalankan model ini.
    """,
    unsafe_allow_html=True
    )



elif opt == "Input File":
    input_file =  st.file_uploader("input file",type=["png","jpg"],accept_multiple_files= True)
        
    for image in input_file:
        st.image(image)

    def predict(image):
                prediction = model.predict(image)
                prediction = predict(image=True)
                st.write(prediction)
                st.image(prediction)
                st.write(predict)
                st.image(predict)
                st.pyplot(predict)
                st.pyplot(prediction)
                st.code(predict)
    
        #with st.echo():
           # st.write(model)
        #prediction = model.predict(input_file)
        #st.write(testModel(input_file),"rb")
        #st.code(testModel)
        #st.code(model)
        #st.code(model.predict([input_file]))
        #st.write(fig)


elif opt == "URL Input":
    url = st.text_input('The URL link')
    st.image(url)
    

elif opt == "Biography":
    st.write("""### Biography ###        
                
                    """)
    st.markdown(
     """
    <style>
        .justified-text {
            text-align: justify;
            text-justify: inter-word;
        }
    </style>
    """,
    unsafe_allow_html=True
    )
    st.write(
    """
    I'am Satyo Pradana 
    who is passionate about Data Analytics, Business Intelligence, Machine Learning, 
    Tech and AI. I am a self managed, goal driven with strong process-acumen and fast learner.
    My area of expertise comprises: Data Analysis, Data Scientist,
    Database (MySQL), presentation skills such as Tableau, Power BI and
    code programming (Python and R)


    """,
    unsafe_allow_html=True
    )
    
    st.image(r"C:\Users\SATYO\Documents\GitHub\dcd21\model-img\model\10057862_316449_300.jpg")
    st.markdown("#### My Link ####")
    st.markdown("www.linkedin.com/in/satyo-pradana")
    st.markdown("https://github.com/satyopradana/")
    st.markdown("https://rpubs.com/Satyo1")


#tab1,tab2 = st.tabs(['Input File','URL Input'])

#with tab1: input_file =  st.file_uploader("input file",type=["png","jpg"],accept_multiple_files= True)
#if input_file is not None:
    #for image in input_file:
       # st.image(image)

#with tab2: url = st.text_input('The URL link')
#if url is not None:
  #  st.image(url)

#connected = http.client.HTTPConnection(url)

#st.image(testModel(input_file))
#if input_file == True:
#    st.image(input_file)
#elif:url
#    st.image(url)
  


#input_file = st.file_uploader("Input File")
#camera_input = st.camera_input("Camera Input")



#st.success("Done")


 




