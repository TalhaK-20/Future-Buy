from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

app = Flask(__name__)

# data = pd.read_csv("User_Data.csv")
# label_encoder = LabelEncoder()
# data['Gender'] = label_encoder.fit_transform(data['Gender'])
#
# X = data[['Gender', 'Age', 'EstimatedSalary']]
# y = data['Purchased']
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# svm_model = SVC(kernel='linear')
# svm_model.fit(X_train, y_train)
#
# with open('svm_model.pkl', 'wb') as model_file:
#     pickle.dump(svm_model, model_file)
#
# print("Model training complete and saved as svm_model.pkl")

with open('svm_model.pkl', 'rb') as model_file:
    svm_model = pickle.load(model_file)


def predict_purchase(gender, age, salary):
    gender_map = {'male': 0, 'female': 1}
    input_data = np.array([[gender_map[gender.lower()], age, salary]])
    prediction = svm_model.predict(input_data)
    return prediction[0]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':

        gender = request.form['gender']
        age = int(request.form['age'])
        salary = float(request.form['salary'])
        prediction = predict_purchase(gender, age, salary)

        return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
