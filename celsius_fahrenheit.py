"""
Este script entrena un modelo de red neuronal usando TensorFlow para convertir grados Celsius a Fahrenheit.

📌 Descripción:
- Genera datos de entrenamiento aleatorios (Celsius) y sus equivalentes exactos en Fahrenheit.
- Utiliza un modelo de red neuronal secuencial con una sola capa densa y una unidad de salida.
- Optimiza los pesos usando el optimizador Adam para minimizar el error cuadrático medio.
- Después del entrenamiento, predice la conversión de un valor Celsius dado.

"""

import tensorflow as tf
import numpy as np

# ------------------------------
# Datos
# ------------------------------
def create_data(n, min_val=-273.15, max_val=1000):
    celsius = np.random.uniform(min_val, max_val, size=(n,))
    fahrenheit = celsius * 1.8 + 32
    return celsius, fahrenheit

celsius, fahrenheit = create_data(100)

# ------------------------------
# Parámetros y modelo
# ------------------------------
learning_rate = 0.1

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss="mean_squared_error"
)

# ------------------------------
# Entrenamiento
# ------------------------------
print("El entrenamiento ha comenzado...")
history = model.fit(celsius, fahrenheit, epochs=1000, verbose=0)  # verbose=0 para menos spam
print("El entrenamiento ha terminado...")

# ------------------------------
# Predicción
# ------------------------------
example = 39.0
result = model.predict(np.array([[example]]), verbose=0)
print(f"Predicción para {example}: {result[0][0]:.2f}°F")

# ------------------------------
#Variables del modelo
print(input.get_weights())
# ------------------------------
