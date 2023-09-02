from flask import Flask, render_template, request
import numpy as np
import pickle
from flask_wtf import FlaskForm
from wtforms import FloatField,IntegerField, SubmitField
from wtforms.validators import DataRequired
from flask_bootstrap import Bootstrap

app = Flask(__name__)
bootstrap = Bootstrap(app)
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pickle', 'rb'))
app.config['SECRET_KEY'] = 'hard to guess string'

class PredictForm(FlaskForm):
    sepal_length = FloatField('Sepal Length', validators=[DataRequired()])
    sepal_width = FloatField('Sepal Width', validators=[DataRequired()])
    petal_length = FloatField('Petal Length', validators=[DataRequired()])
    petal_width = FloatField('Petal Width', validators=[DataRequired()])
    submit = SubmitField('Predict')

# @app.route('/')
# def Home():
    
#     return render_template('index.html')



@app.route('/', methods=['POST', 'GET'])
def predict():
    #sepal_length, sepal_width, petal_length, petal_width = None, None, None, None  # Initialize variables
    form = PredictForm()
    if form.validate_on_submit():
        sepal_length = form.sepal_length.data
        sepal_width = form.sepal_width.data
        petal_length = form.petal_length.data
        petal_width = form.petal_width.data

        float_features = [sepal_length, sepal_width, petal_length, petal_width]
        features = [np.array(float_features)]
        final_features = scaler.transform(features)
        prediction = model.predict(final_features)
        print(prediction)

        return render_template('index.html', form=form, prediction=f'The iris flower species is {prediction[0]}')

    return render_template('index.html', form=form, prediction=None)

if __name__ == '__main__':
    app.run(debug=True)





# from flask import Flask, render_template, request
# import numpy as np
# import pickle
# from flask_wtf import FlaskForm
# from wtforms import StringField, SubmitField, IntegerField
# from wtforms.validators import DataRequired
# from flask_bootstrap import Bootstrap


# app = Flask(__name__)
# bootstrap = Bootstrap(app)
# model = pickle.load(open('model.pkl', 'rb'))
# app.config['SECRET_KEY'] = 'hard to guess string'


# @app.route('/')
# def Home():
#     #form = PredictForm()
#     return render_template('index.html')


# @app.route('/predict', methods=['POST', 'GET'])
# def predict():
#     sepal_length, sepal_width, petal_length, petal_width = None, None, None, None  # Initialize variables
#     form = PredictForm()
#     if form.validate_on_submit():
#         # Extract data from the form
#         sepal_length = form.sepal_length.data
#         sepal_width = form.sepal_width.data
#         petal_length = form.petal_length.data
#         petal_width = form.petal_width.data

#         # Convert input values to float
#         float_features = [float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)]

#         # Prepare features for prediction
#         features = [np.array(float_features)]  # Reshape to 2D array

#         # Make the prediction using your model (assuming 'model' is already defined)
#         prediction = model.predict(features)  # Assuming you want the first prediction if there are multiple classes

#         # Map the numeric prediction to the corresponding class (e.g., 'setosa', 'versicolor', 'virginica')
#         #iris_classes = ['setosa', 'versicolor', 'virginica']
#         #predicted_class = iris_classes[prediction]

#         return render_template('index.html', form=form, prediction=f'The iris flower species is {prediction}')

#     return render_template('index.html',form=form, prediction=None)  # Return None if form is not validated


# class PredictForm(FlaskForm):
#     sepal_length = IntegerField('Sepal Length', validators=[DataRequired()])
#     sepal_width = IntegerField('Sepal Width', validators=[DataRequired()])
#     petal_length = IntegerField('Petal Length', validators=[DataRequired()])
#     petal_width = IntegerField('Petal Width', validators=[DataRequired()])
#     submit = SubmitField('Predict')

# @app.route('/predict', methods = ['POST'])
# def predict():
#     sepal_length, sepal_width, petal_length, petal_width = None
#     form = PredictForm()
#     if form.validate_on_submit():
#         float_features = [float(X) for X in form.[sepal_length, sepal_width, petal_length, petal_width].data]
#         features = [np.array(float_features)]
#         prediction = model.predict(features)
#     return render_template('index.html', prediction=f'The iris flower species is {prediction}')

# @app.route('/', methods=['GET', 'POST'])
# def index():
# name = None
# form = NameForm()
# if form.validate_on_submit():
#     name = form.name.data
# form.name.data = ''
# return render_template('index.html', form=form, name=name)

if __name__=='__main__':
    app.run(debug=True)