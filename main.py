import streamlit as st
from PIL import Image
from gfpgan_inference import enhance_with_gfpgan
import io

st.title("Portrait Enhancer")
st.write("Enhance face image resolution using GFPGAN!")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
rescaling_factor = st.selectbox("Choose rescaling factor (upscale)", [1, 2, 3, 4], index=1)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Enhance Image"):
        with st.spinner("Enhancing image..."):
            input_image_path = "/tmp/input_image.jpg"
            image.save(input_image_path)

            enhanced_image = enhance_with_gfpgan(input_image_path, upscale=rescaling_factor)

        st.image(enhanced_image, caption="Uploaded Image", use_container_width=True)

        img_byte_arr = io.BytesIO()
        enhanced_image.save(img_byte_arr, format="JPEG", quality=95)
        img_byte_arr = img_byte_arr.getvalue()

        st.download_button(
            label="Download Enhanced Image",
            data=img_byte_arr,
            file_name="enhanced_image.jpg",
            mime="image/jpeg"
        )



