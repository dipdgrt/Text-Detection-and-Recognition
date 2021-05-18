

import streamlit as st
import numpy as np
import cv2 
import cfg
import Final



st.title('Text Detection And Recognising ')

st.header('Input Image')

images=st.file_uploader('upload a image')

if images:
    st.image(images)
    image_array = np.asarray(bytearray(images.read()), dtype=np.uint8)
    final_image = cv2.imdecode(image_array, 3)
    
    im,blog,pred=Final.load_image(final_image)

    im=cv2.resize(im,dsize=(1200,720))
    st.image(im)
    for i in range (len(blog)):
        st.image(blog[i])
        st.text(pred[i])


    






