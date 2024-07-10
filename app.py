from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


app = Flask(__name__)

def create_model(inputShape):
    model = Sequential() # Capa de entrada
    model.add(Dense(64, input_dim=10, activation='relu'))
    model.add(Dense(32, activation='linear'))  # Capa oculta # Capa oculta
    model.add(Dense(1))  # Capa de salida
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

try:
    model = joblib.load('modeloRf93.pkl')
    scaler = joblib.load('dataFrameScalado.pkl')
    app.logger.debug('Modelo cargado correctamente.')

except Exception as e: 
    app.logger.error(f'Error al cargar el modelo: {str(e)}')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        Year = float(request.form['Year'])
        Fuel_Type = float(request.form['Fuel_Type'])
        Transmission = float(request.form['Transmission'])
        Mileage = float(request.form['Mileage'])
        Engine = float(request.form['Engine'])
        Seats = float(request.form['Seats'])

        # Verificar los datos recibidos
        app.logger.debug(f'Year: {Year}, Fuel_Type: {Fuel_Type}, Transmission: {Transmission}, Mileage: {Mileage}, Engine: {Engine}, Seats: {Seats}')

        input_data = pd.DataFrame({
            'Name': [0],
            'Location': [0],
            'Year': [Year],
            'Kilometers_Driven': [0],
            'Fuel_Type': [Fuel_Type],
            'Transmission': [Transmission],
            'Owner_Type': [0],
            'Mileage': [Mileage],
            'Engine': [Engine],
            'Power': [0],
            'Seats': [Seats],
            'New_Price': [0]
        })

        app.logger.debug(f'DataFrame de entrada creado:{input_data}')

        # Escalar los datos de entrada
        scaled_data = scaler.transform(input_data)

        # Seleccionar solo las características usadas para el modelo
        scaled_data_for_prediction = scaled_data[:, [2, 4, 5, -5, -4, -2]]  # Asegúrate de que estos índices son correctos

        # Realizar la predicción con los datos escalados
        prediccion = model.predict(scaled_data_for_prediction)

        # Devolver la predicción como JSON
        prediction_value = round(float(prediccion[0]), 2)

        return jsonify({'prediction': prediction_value})

    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
