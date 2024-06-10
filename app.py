import streamlit as st
import pandas as pd
import joblib
import json
from preprocessing import cleanse_text, case_folding, tokenize_text, norm, remove_stopwords, stem_text


model = joblib.load('nb.pkl')
vectorizer = joblib.load('tf-idf.pkl')

st.title('Sentiment Analysis Danau Kelimutu with NB + TFIDF')

# Pilihan input: Teks atau File
option = st.sidebar.selectbox('Choose Input Option:', ['Text', 'File'])

with open('normalisasi.txt') as f:
    word = f.read()
dict_slang = json.loads(word)

if option == 'Text':
    # Input teks
    user_input = st.text_area('Enter Text:', '')
    if st.button('Analyze'):
        # Preprocessing teks
        cleaned_text = cleanse_text(user_input)
        folded_text = case_folding(cleaned_text)
        tokenized_text = tokenize_text(folded_text)
        norm_text = norm(tokenized_text, dict_slang)
        wstopword_text = remove_stopwords(norm_text)
        stemmed_text = ' '.join(stem_text(wstopword_text))

        # Tampilkan tahapan preprocessing
        st.subheader('Preprocessing Steps:')
        st.write(pd.DataFrame({'Step': ['Cleaning', 'Case Folding', 'Tokenization', 'Normalization', 'Stopword Removal',
                                        'Stemming'],
                               'Result': [cleaned_text, folded_text, tokenized_text,  norm_text, wstopword_text,
                                          stemmed_text]}))

        # Proses teks dan lakukan prediksi
        new_data_tfidf = vectorizer.transform([stemmed_text])
        prediction = model.predict(new_data_tfidf)
        st.write('Sentiment:', prediction)

elif option == 'File':
    # Input file Excel/CSV
    uploaded_file = st.file_uploader('Upload Excel/CSV File:', type=['csv', 'xlsx'])
    if uploaded_file is not None:
        # Baca file dan lakukan prediksi
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)

        # Preprocessing teks dalam kolom 'Text'
        df['text'] = df['text'].astype(str)
        df['clean'] = df['text'].apply(cleanse_text)
        df['case_folding'] = df['clean'].apply(case_folding)
        df['tokenization'] = df['case_folding'].apply(tokenize_text)
        df['normalize'] = df['tokenization'].apply(lambda x: norm(x, dict_slang))
        df['stopword_removal'] = df['normalize'].apply(remove_stopwords)
        df['Stemmed'] = df['stopword_removal'].apply(stem_text)
        df['Stemmed'] = df['Stemmed'].apply(lambda x: ' '.join(x))

        # Proses teks dan lakukan prediksi
        # Jumlah fitur yang diinginkan sesuai dengan model SVM
        new_data_tfidf = vectorizer.transform(df['Stemmed'])

        predictions = model.predict(new_data_tfidf)
        df['Sentiment'] = predictions

        # Tampilkan jumlah label
        st.subheader('Sentiment Distribution:')
        st.write(df['Sentiment'].value_counts())

        # Tampilkan barchart
        st.bar_chart(df['Sentiment'].value_counts())

        # Tampilkan tahapan preprocessing
        st.subheader('Preprocessing Steps:')
        st.write(df)
