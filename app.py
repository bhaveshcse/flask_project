from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Loading the CSV data into a Pandas DataFrame
heart_data = pd.read_csv('heart.csv')

# Displaying the first few rows of the dataset
print(heart_data.head())

# Getting some info about the data
heart_data.info()

# Checking for missing values
print(heart_data.isnull().sum())

# Statistical measures about the data
print(heart_data.describe())

# Checking the distribution of the target variable
print(heart_data['target'].value_counts())

# Splitting the data into features and target
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the Logistic Regression model
log_model = LogisticRegression(solver='lbfgs', max_iter=1000)
log_model.fit(X_train, Y_train)

# Create and train the KNN model
knn_model = KNeighborsClassifier()

# Hyperparameter Tuning using GridSearchCV
param_grid = {'n_neighbors': np.arange(1, 31), 'weights': ['uniform', 'distance']}
grid_search = GridSearchCV(knn_model, param_grid, cv=5)
grid_search.fit(X_train, Y_train)
best_knn_model = grid_search.best_estimator_

# Training accuracy of Logistic Regression
train_logistic_accuracy = accuracy_score(log_model.predict(X_train), Y_train)
print('Logistic Regression Training Accuracy:', train_logistic_accuracy)

# Test accuracy of Logistic Regression
test_logistic_accuracy = accuracy_score(log_model.predict(X_test), Y_test)
print('Logistic Regression Test Accuracy:', test_logistic_accuracy)

# Training accuracy of the best KNN model
train_knn_accuracy = accuracy_score(best_knn_model.predict(X_train), Y_train)
print('KNN Training Accuracy:', train_knn_accuracy)

# Test accuracy of the best KNN model
test_knn_accuracy = accuracy_score(best_knn_model.predict(X_test), Y_test)
print('KNN Test Accuracy:', test_knn_accuracy)

prediction = 0
# Initialize Flask app
app = (Flask(__name__))

@app.route('/')
def login1():
    return render_template('login.html')

@app.route('/heart2')
def heart2():
    return render_template('heart.html')

@app.route('/about')
def about1():
    return render_template('about.html')

# Serve the registration page
@app.route('/register', methods=['GET', 'POST'])
def register1():
    if request.method == 'GET':
        return render_template('register.html')
    elif request.method == 'POST':
        # Here you can handle the registration form submission
        username = request.form['username']
        password = request.form['password']
        # Example: saving the user to a database
        # You would usually do more validation and error handling here
        return f"Registered user: {username}"

@app.route('/heart1', methods=['get', 'POST'])
def heart1():
    age = 0
    sex = 0
    cp = 0
    trestbps = 0
    chol = 0
    fbs = 0
    restecg = 0
    thalach = 0
    exang = 0
    oldpeak = 0
    slope = 0
    ca = 0
    thal = 0
    if request.method == 'POST':
        # Using KNN model for prediction
        age = request.form["age"]
        sex = request.form["sex"]
        cp = request.form["cp"]
        trestbps = request.form["trestbps"]
        chol = request.form["chol"]
        fbs = request.form["fbs"]
        restecg = request.form["restecg"]
        thalach = request.form["thalach"]
        exang = request.form["exang"]
        oldpeak = request.form["oldpeak"]
        slope = request.form["slope"]
        ca = request.form["ca"]
        thal = request.form["thal"]
         
    input_data_as_numpy_array = np.asarray([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Scaling the input data
    input_data_scaled = scaler.transform(input_data_reshaped)

    # Use KNN model for prediction
    prediction = best_knn_model.predict(input_data_scaled)
    print(prediction)
    if prediction[0] == 0:
        print('The Person does not have Heart Disease')
    else:
        print('The Person has Heart Disease')
    return render_template('result1.html', result1=prediction)


if __name__ == '__main__':
    app.run(debug=True)