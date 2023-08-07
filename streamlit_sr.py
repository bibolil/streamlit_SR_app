import streamlit as st
import pandas as pd
from PIL import Image
import time
from SwinIR import main_test_swinir
import os
import shutil
def UI_generation():
    st.markdown("""
        <style>
        .stRadio [role=radiogroup]{
            align-items: center;
            justify-content: center;
        }

        </style>
    """,unsafe_allow_html=True)

    st.title('Super Resolution using SwinIR')
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='test')

    radio_option = st.radio('Select the model', ('SwinIR', 'SwinIR-L'),horizontal=True)
    if radio_option == 'SwinIR':
        large_model = False
    else:
        large_model = True  
        
    upscale = st.button("Upscale", use_container_width=True,disabled=uploaded_file is None)
    if upscale:
        upload_folder = 'Images_to_process'
        result_folder = 'results'
        if os.path.isdir(upload_folder):
            shutil.rmtree(upload_folder)
        if os.path.isdir(result_folder):
            shutil.rmtree(result_folder)
        os.mkdir(upload_folder)
        os.mkdir(result_folder)
        dst_path = os.path.join(upload_folder,'image.jpg')
        print(image.format)
        image.save(dst_path)

        with st.spinner('Wait for it...'):
            if large_model:
                model_path='experiments/pretrained_models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth'
                st.write("Upscaling using SwinIR-L")
                main_test_swinir.main('real_sr',4,15,40,128,large_model,model_path,upload_folder,None,None,32)

            else:
                model_path='experiments/pretrained_models/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth'
                st.write("Upscaling using SwinIR")
                main_test_swinir.main('real_sr',4,15,40,128,large_model,model_path,upload_folder,None,None,32)
                
    
        st.success('Done!')
        if(large_model):
            st.image('results/swinir_real_sr_x4_large/image_SwinIR.png', caption='test')
            result_path = 'results/swinir_real_sr_x4_large/image_SwinIR.png'
        else:
            st.image('results/swinir_real_sr_x4/image_SwinIR.png', caption='test')
            result_path = 'results/swinir_real_sr_x4/image_SwinIR.png'     

                
        with open(result_path, "rb") as file:
                btn = st.download_button(
                        label="Download image",
                        data=file,
                        file_name="upscaled_image.png",
                        mime="image/png"
                    )    

if __name__ == "__main__":
    try:
        UI_generation()
    except Exception as e:
        st.error(e)