import streamlit as st  # type: ignore
from PIL import Image
import requests # type: ignore
from io import BytesIO
import os

st.set_page_config(
    page_title="Bank Marketing",
    page_icon="💬"
)

# CSS - - - - - 
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
local_css("styles.css")
# - - - - - - - 
st.markdown('<h1 class="custom-title">Contacts</h1>', unsafe_allow_html=True)



if st.button("◀️\u2003 🎲 Avez vous souscrit ?"):
    st.switch_page("pages/7_🎲_Avez_vous_souscrit?.py")
st.markdown('<hr class="my_custom_hr">', unsafe_allow_html=True)


st.markdown("""
            Contactez nous si vous voulez plus d'informations 
            """)


# Function to resize image to a fixed height
def resize_image_to_height(image_path, height):
    # Check if the image path is a URL or a local path
    if image_path.startswith('http'):
        response = requests.get(image_path)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(image_path)
    
    # Resize image
    wpercent = (height / float(img.size[1]))
    width = int((float(img.size[0]) * float(wpercent)))
    img = img.resize((width, height), Image.ANTIALIAS)
    return img

# Define the desired height
fixed_height = 300

col1, col2, col3 = st.columns(3)

with col1:
    st.header("Estelle MIR-EMARATI")
    img = resize_image_to_height("photos/Estelle.jpeg", fixed_height)
    st.image(img)
    st.markdown("emiremarati@gmail.com ")


with col2:
    st.header("Laurent COCHARD")
    img = resize_image_to_height("photos/Laurent.jpeg", fixed_height)
    st.image(img)
    st.markdown("lololaclaye35@gmail.com ")

with col3:
    st.header("Guillaume BRETHES")
    img = resize_image_to_height("photos/Guillaume.png", fixed_height)
    st.image(img)
    st.markdown("LinkedIn: [www.linkedin.com/in/guillaume-brethes](https://www.linkedin.com/in/guillaume-brethes)")
    st.markdown("GitHub: [https://github.com/guillaumebrethes/Oct_cda_bankmarketing](https://github.com/guillaumebrethes/Oct_cda_bankmarketing)")


st.markdown(" ")

st.image("https://datascientest.com/wp-content/uploads/2022/03/logo-2021.png")