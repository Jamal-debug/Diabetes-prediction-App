
import numpy as np
import pickle
from flask import Flask , render_template,request

app=Flask(__name__,template_folder='template')
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output=round(prediction[0],2)
    if(output > 0.5):
        return render_template('home.html', pred='you should be diabetic .')
    else:
        return render_template('home.html', pred='you should not diabetic .')

if __name__ == '__main__':
    app.run(debug=True)
