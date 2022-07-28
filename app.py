
from cProfile import label
from flask import Flask, render_template, request
import pickle

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
filename = 'model.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('vectorize.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict():
   if request.method == 'POST':
    label={0:'this is not spam',1:'this is spam'}
    message = request.form['message']
    data = [message]
    vect = cv.transform(data).toarray()
    my_prediction = classifier.predict(vect)[0]
    return render_template('index.html', prediction_text=label[my_prediction])


if __name__ == '__main__':
	app.run(debug=True)


