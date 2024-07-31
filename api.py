from flask import Flask, request, render_template 
from newspaper import article
from newspaper import ArticleException
import re
from anyascii import anyascii
import joblib
import pandas as pd
from markupsafe import Markup
import numpy

def scrape_text(url):
    try: 
        a = article(url)
    except(ArticleException):
        return None
    text = re.sub('\n', '', a.text)
    return text

def clean(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text) # anything in brackets
    text = re.sub("\\W"," ",text) # non-alphanumerics
    text = re.sub('https?://\S+|www\.\S+', '', text) # urls
    text = re.sub('<.*?>+', '', text) # html tags
    text = re.sub('[%s]' % re.escape('!\"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'), '', text) # punctuation
    text = re.sub('\n', '', text) # new lines
    text = re.sub('\w*\d\w*', '', text) # numbers within letters (not seperated by spaces)
    text = re.sub(' +', ' ', text) # multiple spaces
    text = re.sub('\%\r', '', text)
    text = anyascii(text)
    return text

def detect(url):
    text = scrape_text(url)
    if text is None: 
        return "Article content could not be found, may be behind a paywall"
    text_dict = {"text":[text]}
    text_df = pd.DataFrame(text_dict)
    text_df["text"] = text_df["text"].apply(clean) 
    clean_text_df = text_df["text"]
    vectorizer = joblib.load("vectroizer.pkl")
    vector_text = vectorizer.transform(clean_text_df)
    lr = joblib.load("log_reg_model.pkl")
    pred = str(lr.predict_proba(vector_text))
    return pred

def format_prediction(prob, fake):
    message = "The article is <strong>FAKE NEWS</strong>"
    if fake is False:
        message = "The article is <strong>NOT FAKE NEWS</strong>"
    if fake is None:
        message = "The article is <strong>EQUALLY LIKELY</strong> fake or not fake news"
    return message, f'with {round(prob*100, 2)}% certainty'


# Flask constructor
app = Flask(__name__)   
 
# A decorator used to tell the application
# which URL is associated function
@app.route('/', methods =["GET", "POST"])
def submitURL():
    if request.method == "POST":
        url = request.form.get("url")
        pred = detect(url)
        not_fake_prob = float(pred[2:12])
        fake_prob = float(pred[13:23])
        if not_fake_prob == fake_prob:
            fake = None
            prob = .5
        elif not_fake_prob > fake_prob:
            fake = False
            prob = not_fake_prob
        else: 
            fake = True
            prob = fake_prob
        format_pred = format_prediction(prob, fake)
        print(format_pred)
        return render_template("home.html", message=Markup(format_pred[0]), prob=format_pred[1])
    return render_template("home.html")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
