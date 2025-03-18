
# Predicción de Precios de Bitcoin con ARIMA

Este proyecto utiliza un modelo ARIMA para predecir los precios futuros de Bitcoin (BTC-USD). Se descargan datos históricos utilizando la librería yfinance, se realiza un análisis exploratorio de la serie temporal, se ajusta un modelo ARIMA(1,1,1) y se generan predicciones para los próximos 30 días. Además, se generan gráficos que muestran el precio histórico, la serie diferenciada y la predicción.


## Requisitos

- Python 3.x
- yfinance
- pandas
- matplotlib
- statsmodels

Para instalar las dependencias, ejecuta:

    pip install yfinance pandas matplotlib statsmodels

## Uso

1. Clona o descarga este repositorio.
2. Asegúrate de tener instaladas las dependencias.
3. Ejecuta el script principal:

       python bitcoin_arima.py

4. Revisa los archivos generados en el directorio:
    - bitcoin_precio_historico.png
    - bitcoin_diferenciado.png
    - bitcoin_forecast.png
    - bitcoin_forecast.csv

## Análisis y Conclusiones

### Precio Histórico

- La serie de precios de Bitcoin muestra una alta volatilidad, con períodos de alza y caídas marcadas, lo que indica un comportamiento fuertemente no estacionario.
- El gráfico `bitcoin_precio_historico.png` refleja estos movimientos y tendencias generales.

### Serie Diferenciada

- Al aplicar la diferenciación, la serie se vuelve más estacionaria, lo cual es necesario para el modelado de series temporales.
- La prueba de Dickey-Fuller sobre la serie diferenciada confirma que la transformación mejora la estacionariedad, facilitando el ajuste del modelo ARIMA.
- El gráfico `bitcoin_diferenciado.png` muestra la variabilidad en los cambios diarios de precio.

### Modelo ARIMA

- Se ha ajustado un modelo ARIMA(1,1,1) como aproximación inicial para capturar la dinámica de la serie.
- El resumen del modelo (imprimido en consola) indica que los coeficientes son significativos, lo que sugiere un ajuste razonable.

### Predicción a 30 Días

- La predicción para los próximos 30 días se visualiza en `bitcoin_forecast.png` y se guarda en `bitcoin_forecast.csv`.
- El modelo predice una ligera variación respecto al último precio histórico, considerando la alta volatilidad del mercado de criptomonedas.
- Estas predicciones deben interpretarse como una referencia aproximada, dada la naturaleza volátil del mercado.

### Conclusiones Generales

1. **Volatilidad elevada:** Bitcoin muestra fluctuaciones marcadas en su precio, lo que se refleja en la serie histórica.
2. **Necesidad de diferenciación:** La diferenciación es crucial para transformar la serie en estacionaria y poder aplicar modelos de series temporales.
3. **Modelo ARIMA como aproximación inicial:** Aunque el modelo ARIMA(1,1,1) ofrece una buena primera aproximación, se recomienda explorar otros parámetros o modelos alternativos para mejorar la precisión.
4. **Predicción con cautela:** Dada la naturaleza del mercado de criptomonedas, las predicciones deben utilizarse como referencia y complementarse con análisis adicionales.
5. **Próximos Pasos:**
   - Explorar modelos alternativos como SARIMA, Prophet o redes neuronales.
   - Optimizar los parámetros del modelo (p, d, q) y evaluar la precisión usando métricas como MAE o RMSE.
   - Ampliar el análisis incorporando variables exógenas o comparando diferentes criptomonedas.

## Próximos Pasos

- **Explorar modelos alternativos:** Probar modelos como SARIMA, Prophet o incluso enfoques basados en redes neuronales.
- **Optimización del modelo:** Ajustar y comparar diferentes configuraciones de parámetros (p, d, q) para mejorar el ajuste.
- **Ampliar el análisis:** Incluir variables exógenas y utilizar técnicas de validación para evaluar la precisión del modelo.

## Autor

- Baahl11

## Licencia

Este proyecto está bajo la licencia MIT.
EOF
