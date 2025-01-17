import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Netクラスのインポートまたは定義が必要
from model import Net  # もしNetクラスが別のファイルに定義されている場合

# モデルのロード
# 'mnist_cnn.pt'が同じディレクトリにあるかどうかを確認
try:
    model = Net()
    model.load_state_dict(torch.load('mnist_cnn.pt'))
    model.eval()
except FileNotFoundError:
    st.error("Model file 'mnist_cnn.pt' not found. Please ensure it is in the same directory as this script.")
    st.stop()

# 画像をテンソルに変換する関数
def image_loader(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(image).unsqueeze(0)

st.title('MNIST Digit Recognition')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    try:
        # 推論
        tensor = image_loader(image)
        with torch.no_grad():
            output = model(tensor)
            pred = output.argmax(dim=1, keepdim=True)
        
        st.write(f'Predicted Digit: {pred.item()}')
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")