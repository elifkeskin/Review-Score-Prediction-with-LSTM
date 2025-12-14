"""
Problem: yorumlardan -> puan tahmini (1 -5), regresyon problemi

- çok iyiydi, çok memnun kaldım -> 4.5
- berbatti, bir daha gelmem -> 1.2

Dataset:https://huggingface.co/datasets/Yelp/yelp_review_full

"""

# import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle # tokenizer'ı diske kaydetmek için

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler #normalization için

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError


# load yelp dataset

import pandas as pd

splits = {'train': 'yelp_review_full/train-00000-of-00001.parquet', 'test': 'yelp_review_full/test-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/Yelp/yelp_review_full/" + splits["train"])

print(df.head())


# etiketleri 0-4 aralığından 1-5 aralığına dönüştürme

df['label'] = df['label'] + 1


# data preprocessing

texts = df['text'].values  # yorum metinleri
labels = df['label'].values  # puanlar 1-5 arasında

# tokenizer: metni sayıya çevirme
# oov_token : bilinmeyen kelimeleri bu etiketle gösterir.
# num_words : en sık geçen ilk 10000 kelimeyi alır.

tokenizer = Tokenizer(num_words = 10000, oov_token = "<OOV>")

# metni sayılara dönüştürür.
tokenizer.fit_on_texts(texts)

# tokenizer'ı diske kaydet
with open("tokenizer.pickle", "wb") as f:
    pickle.dump(tokenizer, f)

# yorumları dizi haline getirme
sequences = tokenizer.texts_to_sequences(texts)


# padding: tüm dizileri sabit uzunluğa getirme (kısa olanları sonuna 0 ile doldurur.)
# maxlen : 100 token
padded_sequences = pad_sequences(sequences, maxlen=100,  padding = "post", truncating = "post")


# etiketler 1 ile 5 arasındadır, normalization ile 0 ile 1 arasına getirelim.
# Regresyon problemlerinde normalization daha stabil bir öğrenme sağlar.
scaler = MinMaxScaler()
labels_scaled  = scaler.fit_transform(labels.reshape(-1, 1))

# eğitim ve test veri setlerini ayırma
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels_scaled, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}")
print(f"X_train shape: {X_train[:2]}")
print(f"y_train shape: {y_train.shape}")
print(f"y_train shape: {y_train[:2]}")


# LSTM tabanlı regression model
model = Sequential()

# Embedding: Kelime indekslerini vektör uzayına dönüştürür.
# input_dim : kelime sayımız
# output_dim : her bir kelime 128 boyutlu vektörle temsil edilir.
# input_length : sabit dizi uzunluğu yani her metin 100 token uzunluğuna sahip olur.
model.add(Embedding(input_dim=10000, output_dim=128, input_length=100))

# LSTM katmanı: sıralı veride bağlamı öğrenecek katman
model.add(LSTM(128)) # 128: lstm de bulunan hücre sayısı yani daha fazla öğrenme kapasitesi

# tam bağlı (dense) layer 
# Hidden layer: relu, tanh genellikle kullanılır.
model.add(Dense(64, activation='relu'))

# output layer
# Regression problemi için linear activation kullanılır.
# Sigmoid ve softmax ise sınıflandırma problemlerinde kullanılır.
# Sigmoid: iki sınıflı sınıflandırma problemlerinde kullanılır.
# Softmax: çok sınıflı sınıflandırma problemlerinde kullanılır.
model.add(Dense(1, activation='linear'))

# model compile & training
model.compile(optimizer='adam', loss=MeanSquaredError(),metrics=[MeanAbsoluteError()])

# modeli eğitme
# validation_split : eğitim veri setinden %20'ini validation için kullanır.
# batch_size : her adımda işlenecek örnek sayısı
# epochs : toplam eğitim sayısı
history = model.fit(X_train, y_train, epochs=3, batch_size=64, validation_split=0.2)


# eğitim kayıp grafiğini görselleştirme ve modeli kaydetme
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Eğitim süreci MSE")
plt.xlabel("Epoch")
plt.ylabel("Loss: Mean Squared Error")    
plt.legend()
plt.show()

model.save('regression_lstm_yelp.h5')