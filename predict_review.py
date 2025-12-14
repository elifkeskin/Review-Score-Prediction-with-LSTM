import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# modeli yükle
model = load_model('regression_lstm_yelp.h5', compile=False)

# tokenizer yükle
with open('tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

#  ornek input
#  1 star - 5 star
texts = [
"Terrible. Preordered my tires and when I arrived they couldn't find the order anywhere. Once we got through that process I waited over 2 hours for them to be put on... I was originally told it would take 30 mins. Slow, over priced, I'll go elsewhere next time.",
 "BEST DINER IN THE COUNRTY!!! We've been to many famous diners across the country and we still give Gab and Eat the best rating !! I was a little intimidated when I first walked in and there was like 2 pounds of butter just sitting on top of the homefires on the grill. If you are looking for a healthy breakfast they probably can accomodate you, but Everything I ate was clearly the opposite of healthy. After trying like every meal they have I would recommend the mix grill(half unless you are sharing) adding cajun seasoning with the texas toast. Burgers are great too.\n\nIt's hard to find a place that makes a better breakfast than you could make by yourself at home, but this place does it. The atmosphere is classy, old school Americana.",
 "Best experience ever!",
 "Worst experience ever!"
]

# Preprocessing
# texti sayılara çevirme
sequences = tokenizer.texts_to_sequences(texts)

# padding
padded = pad_sequences(sequences, maxlen=100, padding='post')

# LSTM prediction
predictions = model.predict(padded)

# POST PREPROCESSING
# tahminleri 1-5 aralığında normalize etme
predictions_scaled = predictions * 5

# tahminleri yazdırma
for i, comment in enumerate(texts):
    print(f"Yorum:\n {comment}")
    print(f"Tahmini skor değeri: {predictions_scaled[i][0]:.2f}")




