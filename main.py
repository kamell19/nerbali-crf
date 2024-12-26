import pandas as pd
import streamlit as st
from lib import *
from model import *
from model import *


def main():
    st.title(
        "Final Projek NLP-:blue[NER dengan Bahasa ]:green[ Bali]")
    st.header('Conditional Random Field (CRF)')

    teks = st.text_input('Input text')

    # Tidak perlu preprocessing kompleks untuk NER
    if st.button('Prediksi Kalimat'):
        if teks:
            prediction = predict_text(teks)

            if prediction is not None:  # Check if prediction is valid
                st.write("Hasil Prediksi NER:")
                for word, tag in prediction:
                    st.write(f"{word}: {tag}")
            else:
                st.write("Prediksi NER gagal.")
        else:
            st.write("Masukkan teks terlebih dahulu.")


if __name__ == '__main__':
    main()
