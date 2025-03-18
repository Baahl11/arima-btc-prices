#!/usr/bin/env python
"""
Proyecto: Predicción de Precios de Bitcoin con ARIMA
Descripción: Obtiene datos históricos de Bitcoin usando yfinance, 
realiza un análisis exploratorio de la serie temporal, ajusta un modelo ARIMA 
y predice los precios futuros.
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import datetime

# 1. Descargar datos históricos de Bitcoin (BTC-USD)
# Definir período: últimos 5 años
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=5*365)

# Descarga los datos de Bitcoin
btc_data = yf.download("BTC-USD", start=start_date, end=end_date)
btc_data = btc_data[['Adj Close']]  # Usamos el precio ajustado

# Renombrar la columna
btc_data.rename(columns={'Adj Close': 'Precio'}, inplace=True)
print("Datos históricos de Bitcoin:")
print(btc_data.head())

# 2. Análisis exploratorio y visualización de la serie temporal
plt.figure(figsize=(12, 6))
plt.plot(btc_data.index, btc_data['Precio'], color='blue', label='Precio Bitcoin')
plt.title("Precio Histórico de Bitcoin (BTC-USD)")
plt.xlabel("Fecha")
plt.ylabel("Precio (USD)")
plt.legend()
plt.tight_layout()
plt.savefig("bitcoin_precio_historico.png")
plt.show()

# 3. Verificar la estacionariedad de la serie (usando la gráfica y la prueba de Dickey-Fuller)
from statsmodels.tsa.stattools import adfuller

result = adfuller(btc_data['Precio'])
print("\nPrueba de Dickey-Fuller:")
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")
# Si p-value > 0.05 la serie es no estacionaria

# En la mayoría de los casos, la serie de precios de Bitcoin es no estacionaria,
# por lo que aplicaremos diferenciación.
btc_data['Diferencia'] = btc_data['Precio'].diff()
btc_data.dropna(inplace=True)

plt.figure(figsize=(12, 6))
plt.plot(btc_data.index, btc_data['Diferencia'], color='green', label='Diferencia de Precio')
plt.title("Serie Diferenciada de Bitcoin")
plt.xlabel("Fecha")
plt.ylabel("Cambio en el Precio (USD)")
plt.legend()
plt.tight_layout()
plt.savefig("bitcoin_diferenciado.png")
plt.show()

# Realizamos la prueba de Dickey-Fuller sobre la serie diferenciada
result_diff = adfuller(btc_data['Diferencia'])
print("\nPrueba de Dickey-Fuller (Serie Diferenciada):")
print(f"ADF Statistic: {result_diff[0]}")
print(f"p-value: {result_diff[1]}")

# 4. Ajustar un modelo ARIMA
# Usaremos la serie diferenciada para ajustar un modelo ARIMA(p,d,q). Aquí d=1 (ya que diferenciamos una vez).
# Es recomendable probar distintos valores, pero usaremos ARIMA(1,1,1) como ejemplo.
model = ARIMA(btc_data['Precio'], order=(1, 1, 1))
model_fit = model.fit()
print("\nResumen del modelo ARIMA(1,1,1):")
print(model_fit.summary())

# 5. Realizar predicciones futuras
# Predecir los próximos 30 días
forecast_steps = 30
forecast = model_fit.forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=btc_data.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
forecast_series = pd.Series(forecast, index=forecast_index)

plt.figure(figsize=(12, 6))
plt.plot(btc_data.index, btc_data['Precio'], label="Precio Histórico", color='blue')
plt.plot(forecast_series.index, forecast_series, label="Predicción 30 días", color='red', linestyle='--')
plt.title("Predicción de Precios de Bitcoin con ARIMA")
plt.xlabel("Fecha")
plt.ylabel("Precio (USD)")
plt.legend()
plt.tight_layout()
plt.savefig("bitcoin_forecast.png")
plt.show()

# 6. Guardar los resultados en un archivo CSV (opcional)
forecast_series.to_csv("bitcoin_forecast.csv", header=["Precio_Predicho"])

print("\nLa predicción de precios para los próximos 30 días se ha guardado en 'bitcoin_forecast.csv'.")
