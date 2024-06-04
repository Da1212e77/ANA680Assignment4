from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

filename = 'file_iris.pkl'
model = pickle.load(open(filename, 'rb')) 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    Sepal_Length = request.form['sepal_length']
    Sepal_Width = request.form['sepal_width']
    Petal_Length = request.form['petal_length']
    Petal_Width = request.form['petal_width']
    
    pred = model.predict(np.array([[Sepal_Length, Sepal_Width, Petal_Length, Petal_Width]]))
    # print(pred)
    return render_template('index.html', predict=str(pred))

if __name__ == '__main__':
    app.run(debug=True)
