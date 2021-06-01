import streamlit as st
import matplotlib.pyplot as plt
from style_transfer import style_transfer, deprocess

# Parameters
im_size = 512

st.title("Neural Style Transfer")
st.header("Transferring styles to images")

@st.cache(suppress_st_warning=True)
def do_style_transfer(content_im, style_im):
    img = style_transfer(content_im, style_im, im_size, print_progress, 100)
    img = deprocess(img)
    return img

def print_progress(iter_num, num_steps, style_loss, content_loss):
    st.text(f"Iteration {iter_num} / {num_steps}")
    st.text(f"Style loss: {style_loss} Content loss: {content_loss}")

col1, col2 = st.beta_columns(2)
content_uploaded = col1.file_uploader(label="Upload the content image", type=['png', 'jpeg', 'jpg'])
style_uploaded = col2.file_uploader(label="Upload the style image", type=['png', 'jpeg', 'jpg'])

if content_uploaded:
    content_img = content_uploaded.read()
    col1.image(content_img, use_column_width=True)

if style_uploaded:
    style_img = style_uploaded.read()
    col2.image(style_img, use_column_width=True)

if not content_uploaded or not style_uploaded:
    st.warning("Please upload both content and style images")
    st.stop()

if content_uploaded and style_uploaded: 
    start_transfer_btn = st.button("Transfer style")

if start_transfer_btn:
    # user clicked on the button, start style transfer
    res = do_style_transfer(content_img, style_img)
    st.image(res)
