import numpy as np
from tensorflow.keras.models import load_model

# Load the previously trained model
model = load_model('textgennietsche.h5')

# Load text data and mappings as before
with open('nietzsche.txt', 'r', encoding='utf-8') as file:
    text = file.read().lower()

chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 40

# Input your own text
input_text = "i live therefore i am  , i am therefore i resist , i resist therefore i am alive"
assert len(input_text) >= 40, "Input text must be at least 40 characters long."

# Generate new text
generated = ''
sentence = input_text[:maxlen]
generated += sentence
print('----- Generating with seed: "' + sentence + '"')

for i in range(400):
    x_pred = np.zeros((1, maxlen, len(chars)))
    for t, char in enumerate(sentence):
        x_pred[0, t, char_indices[char]] = 1.

    preds = model.predict(x_pred, verbose=0)[0]
    next_index = np.argmax(preds)
    next_char = indices_char[next_index]

    sentence = sentence[1:] + next_char

    generated += next_char
print(generated)