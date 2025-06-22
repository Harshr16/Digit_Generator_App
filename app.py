import streamlit as st
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from generator_model import Generator

@st.cache_resource
def load_generator_model():
    model = Generator().to("cpu")
    model.load_state_dict(torch.load("mnist_generator.pth", map_location="cpu"))
    model.eval()
    return model

def generate_images(generator, digit, num_images=5, latent_dim=100):
    with torch.no_grad():
        z = torch.randn(num_images, latent_dim)
        samples = generator(z)
        return samples

st.title("?? Handwritten Digit Generator (GAN)")
st.write("Select a digit (0-9) and generate synthetic handwritten images using a trained GAN.")

digit = st.selectbox("Select a digit", list(range(10)))

if st.button("Generate Images"):
    generator = load_generator_model()
    images = generate_images(generator, digit)

    st.subheader(f"Generated Images for Digit {digit}:")
    cols = st.columns(5)
    for i in range(5):
        img = images[i].squeeze().numpy()
        cols[i].image(img, use_column_width=True, clamp=True, channels="L", caption=f"Image {i+1}")
