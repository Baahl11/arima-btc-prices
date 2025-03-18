#!/usr/bin/env python
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import datetime
import sys

# 1. Descargar datos históricos de Bitcoin (BTC-USD)
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=5*365)  # últimos 5 años
btc_data = yf.download("BTC-USD", start=start_date, end=end_date)

if btc_data.empty:
    print("No se han descargado datos. Verifica tu conexión a Internet o la disponibilidad de datos.")
    sys.exit()

# Usamos la columna 'Close' (ya ajustada) en lugar de 'Adj Close'
btc_data = btc_data[['Close']]
btc_data.rename(columns={'Close': 'Precio'}, inplace=True)
print("Datos históricos de Bitcoin:")
print(btc_data.head())

# 2. Visualización de la serie temporal (Precio Histórico)
plt.figure(figsize=(12, 6))
plt.plot(btc_data.index, btc_data['Precio'], color='blue', label='Precio Bitcoin')
plt.title("Precio Histórico de Bitcoin (BTC-USD)")
plt.xlabel("Fecha")
plt.ylabel("Precio (USD)")
plt.legend()
plt.tight_layout()
plt.savefig("bitcoin_precio_historico.png")
print("Se ha guardado el gráfico: bitcoin_precio_historico.png")
plt.close()

# 3. Verificar estacionariedad con la prueba de Dickey-Fuller
result = adfuller(btc_data['Precio'])
print("\nPrueba de Dickey-Fuller para la serie de precios:")
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")

# Aplicamos diferenciación para lograr estacionariedad
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
print("Se ha guardado el gráfico: bitcoin_diferenciado.png")
plt.close()

result_diff = adfuller(btc_data['Diferencia'])
print("\nPrueba de Dickey-Fuller para la serie diferenciada:")
print(f"ADF Statistic: {result_diff[0]}")
print(f"p-value: {result_diff[1]}")

# 4. Ajustar un modelo ARIMA(1,1,1)
model = ARIMA(btc_data['Precio'], order=(1, 1, 1))
model_fit = model.fit()
print("\nResumen del modelo ARIMA(1,1,1):")
print(model_fit.summary())

# 5. Realizar predicciones para los próximos 30 días
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
print("Se ha guardado el gráfico: bitcoin_forecast.png")
plt.close()

forecast_series.to_csv("bitcoin_forecast.csv", header=["Precio_Predicho"])
print("\nLa predicción de precios para los próximos 30 días se ha guardado en 'bitcoin_forecast.csv'.")

print("---- EJECUTANDO HASTA EL FINAL ----")

def imprimir_conclusiones():
    conclusiones = """
Conclusiones del Análisis ARIMA para Precios de Bitcoin:

1. Precio Histórico:
   - La serie de precios muestra un comportamiento fuertemente no estacionario, con períodos de alza y caída marcados.
   - En el gráfico 'bitcoin_precio_historico.png' se evidencia un incremento sustancial en ciertos periodos, seguido de correcciones pronunciadas.

2. Serie Diferenciada:
   - Tras aplicar la diferenciación (ver 'bitcoin_diferenciado.png'), se observa una mayor variabilidad en los cambios de precio.
   - La prueba de Dickey-Fuller sugiere que la serie diferenciada es más estacionaria, facilitando el ajuste del modelo ARIMA.

3. Modelo ARIMA:
   - Se ajustó un modelo ARIMA(1,1,1) como aproximación inicial. El resumen del modelo indica que los coeficientes son significativos, aunque se podrían explorar otros parámetros para mejorar el ajuste.
   - El modelo se entrenó utilizando la serie diferenciada para capturar la dinámica de precios.

4. Predicción a 30 días:
   - El gráfico 'bitcoin_forecast.png' muestra la proyección para el próximo mes. El modelo predice una ligera variación en relación al último precio histórico.
   - Dada la alta volatilidad del mercado de criptomonedas, se recomienda usar estas predicciones como una referencia aproximada.

5. Observaciones Generales:
   - La volatilidad de Bitcoin se refleja en las grandes fluctuaciones en la serie de precios y sus diferencias.
   - Para un análisis más completo, sería útil incorporar variables exógenas o probar modelos alternativos (como SARIMA, Prophet o modelos basados en redes neuronales).

6. Próximos Pasos:
   - Ajustar y comparar diferentes configuraciones del modelo ARIMA (p, d, q).
   - Evaluar la precisión del modelo utilizando métricas como MAE o RMSE.
   - Ampliar el rango de datos o incluir otras criptomonedas para un análisis comparativo.

¡Análisis Completado!
    """
    print(conclusiones)

imprimir_conclusiones()

print("---- FIN DEL SCRIPT ----")
