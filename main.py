import streamlit as st
import cv2

# Выводим версию OpenCV
st.title("Проверка версии OpenCV")
try:
    st.write("Версия OpenCV: ", cv2.__version__)
except ModuleNotFoundError:
    st.error("OpenCV не установлен в текущем окружении!")



