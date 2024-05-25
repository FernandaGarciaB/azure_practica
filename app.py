from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Cargar el modelo entrenado y los escaladores
with open('modelo_entrenado.pkl', 'rb') as f:
    model = pickle.load(f)
with open('feature_scaler.pkl', 'rb') as f:
    feature_scaler = pickle.load(f)
with open('price_scaler.pkl', 'rb') as f:
    price_scaler = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = None
    if request.method == 'POST':
        try:
            # Recupera los datos del formulario
            sembrada = float(request.form.get('Sembrada'))
            cosechada = float(request.form.get('Cosechada'))
            volumenproduccion = float(request.form.get('Volumenproduccion'))
            rendimiento = float(request.form.get('Rendimiento'))
            valorproduccion = float(request.form.get('Valorproduccion'))
            tipo_cultivo = request.form.get('Nomcultivo')

            # Codificación del tipo de cultivo
            cultivo_frijol = 1 if tipo_cultivo == 'Frijol' else 0
            cultivo_maiz = 1 if tipo_cultivo == 'Maíz grano' else 0

            # Crear el arreglo de características
            features = np.array([[sembrada, cosechada, volumenproduccion, rendimiento, valorproduccion, cultivo_frijol, cultivo_maiz]])

            # Escalar las características
            features_scaled = feature_scaler.transform(features)

            # Realizar la predicción
            prediction_scaled = model.predict(features_scaled)

            # Desescalar la predicción
            prediction = price_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))

            # Redondear el resultado a dos decimales
            prediction_final = round(prediction[0][0], 2)

            # Devuelve el resultado de la predicción
            prediction_text = f'Resultado de la predicción (Precio promedio por kilo): {prediction_final}'
        except Exception as e:
            prediction_text = f"Error: {str(e)}"
    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
