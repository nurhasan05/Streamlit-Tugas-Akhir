import nltk
import preprocessor as p
import re
import string
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')


pattern = r'[0-9]'
def cleanse_text(text):
    for punctuation in string.punctuation:
        text = p.clean(text) #menghapus tag, hashtag
        text = re.sub(r'http[s]?://\S+','',text) #menghapus URL
        text = text.replace(punctuation, '') #menghapus tanda baca
        text = re.sub(pattern, '', text)#menghapus angka
        text = re.sub(r'\r?\n|\r','',text)#menghapus baris baru
        text = text.encode('ascii', 'ignore').decode('ascii') #menghapus emoji
    return text

#case folding
def case_folding(text):
    return text.lower()

# tokenisasi
def tokenize_text(text):
    return text.split()

# Fungsi untuk mengganti kata-kata dalam kalimat dengan value dari dictionary
def norm(text, dictionary):
    text = [dictionary.get(word, word) for word in text]
    return text

list_stopwords = stopwords.words('indonesian')

list_stopwords.extend(["masih","yg", "dg", "rt", "dgn", "ny", 'nya','klo',
                       'kalo', 'amp', 'biar', 'bikin', 'bilang',
                       'gak', 'ga', 'krn', 'nya', 'nih', 'sih',
                       'si', 'tau', 'tdk', 'tuh', 'utk', 'ya',
                       'jd', 'jgn', 'sdh', 'aja', 'udah',
                       'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                       '&amp', 'yah', 'sdgkan', 'sdg', 'emg', 'sm', 'pls', 'mlu', 'ken',
                       'allah', 'brb', 'btw', 'b/c', 'cod', 'cmiiw', 'fyi',
                       'gg', 'ggwp', 'idk', 'ikr', 'lol', 'ootd', 'lmao', 'oot',
                       'pap', 'otw', 'tfl', 'vc', 'ygy','mh','mah', 'tis', 'tisss'])

txt_stopword = pd.read_csv("stopwords.txt", names= ["stopwords"], header = None)

list_stopwords.extend(txt_stopword["stopwords"][0].split(' '))

list_stopwords = set(list_stopwords)

# stopword
def remove_stopwords(words):
    return [word for word in words if word not in list_stopwords]

factory = StemmerFactory()
stemmer = factory.create_stemmer()

# stemming
def stem_text(text):
    text = " ".join(text)
    text = stemmer.stem(text)
    return text.split()

