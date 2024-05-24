from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Cargar el modelo entrenado
model = pickle.load(open('modelo_entrenado.pkl', 'rb'))

# Definir las opciones del menú desplegable
semillas = ['Maíz grano', 'Frijol']

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = None
    if request.method == 'POST':
        try:
            # Recupera los datos del formulario
            sembrada = request.form['Sembrada']
            cosechada = float(request.form['Cosechada'])
            siniestrada = float(request.form['Siniestrada'])
            cantidad = float(request.form['Cantidad'])
            rendimiento = float(request.form['Rendimiento'])
            volumenproduccion = float(request.form['Volumenproduccion'])

            # Codificar las semillas para el modelo
            if sembrada == 'Maíz grano':
                sembrada_codificada = 0
            elif sembrada == 'Frijol':
                sembrada_codificada = 1
            else:
                raise ValueError("Semilla no válida")

            # Prepara los datos para el modelo
            features = np.array([[sembrada_codificada, cosechada, siniestrada, volumenproduccion, rendimiento, cantidad]])
            prediction = model.predict(features)

            # Devuelve el resultado de la predicción
            prediction_text = f'Resultado de la predicción (Precio promedio por tonelada): {prediction[0]}'
        except Exception as e:
            prediction_text = f"Error: {str(e)}"
    return render_template('index.html', semillas=semillas, prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
