from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

model = None

# Cargar el modelo entrenado
try:
    model = joblib.load('modeloNeuronaR2.pkl')
    app.logger.debug('Modelo cargado correctamente.')

except Exception as e: 
    app.logger.error(f'Error al cargar el modelo: {str(e)}')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        return jsonify({'error': 'Modelo no está cargado'}), 500
    
    try:
        # Obtener los datos enviados en el request
        year = float(request.form['year'])
        transmission = request.form['transmission']
        engine = float(request.form['engine'])
        seats = float(request.form['seats'])

        # Mapear valores de transmisión a numéricos
        transmission_mapping = {'Automatic': 0, 'Manual': 1}
        transmission_num = transmission_mapping[transmission]

        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[year,transmission_num,engine,seats]], columns=['year', 'transmission','engine','seats'])
        app.logger.debug(f'DataFrame creado: {data_df}')
        
        # Realizar predicciones
        prediction = model.predict(data_df)
        app.logger.debug(f'Predicción: {prediction[0]}')

        # Devolver las predicciones como respuesta JSON
        return jsonify({'categoria': prediction[0]})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
