from flask import Flask, render_template, request, url_for
import pickle 
from keras.utils import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, GRU, LSTM
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer


model = load_model("Model.h5")
with open('tokenize.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

app = Flask(__name__)
@app.route('/')
def main():
    return render_template("Home.html")

@app.route('/predict', methods = ['POST'])
def classify():
    text = request.form["text"]
    if text.strip() == "":  
        return render_template("home.html",res = "", answer = "")

    text_list = [text]
    text_token = tokenizer.texts_to_sequences(text_list)
    text_pad = pad_sequences(text_token, maxlen = 241, padding = 'pre')
    pred = model.predict(text_pad)
    if pred[0][0] > 0.5:
        result = "Positive Review!"
        per = round((pred[0][0])*100 , 2)
    else:
        result = 'Negative Review! '
        per = round((pred[0][0])*100,2) 
    return render_template("home.html",res = per, answer = result)

if __name__ == "__main__":
    app.run()