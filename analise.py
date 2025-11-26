import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima


df = pd.read_csv('data/data_2000_2019.csv', sep=',')

# =========================================================
# ============ TRANFORMANDO EM UMA TIME SERIES ============

df['data'] = pd.to_datetime({
    'year': df['year'],
    'month': df['month'],
    'day': 1
})

df.index = df['data']
df_dengue_brasil = pd.DataFrame(index=df['data'].unique(),
                                data=df['dengue_cases'].groupby('data').sum())
df_dengue_brasil.index = pd.to_datetime(df_dengue_brasil.index)

# =========================================================
# ============ TRATANDO VALORES NULOS E ZEROS =============

df_dengue_brasil = df_dengue_brasil.fillna(value=0)

df_dengue_brasil = df_dengue_brasil.asfreq('MS', fill_value=0)
df_dengue_brasil['dengue_cases'] = df_dengue_brasil['dengue_cases'] + 1

# =========================================================
# ============ DEMONSTRACAO SERIE SEM ANALIZE =============

df_dengue_brasil.plot(figsize=(12, 6))
plt.title("Casos de dengue no Brasil de 2000 a 2019")

# =========================================================
# ============ DEMONSTRACAO DECOMPOSICA DA SERIE ==========

decomposicao_casos = seasonal_decompose(df_dengue_brasil['dengue_cases'],
                                        model='multiplicative',
                                        period=12)

fig_decomposicao = decomposicao_casos.plot()
fig_decomposicao.set_size_inches(12, 8)
plt.subplots_adjust(top=0.9, right=0.85)
fig_decomposicao.suptitle('Decomposição Completa da Série', fontsize=16)

# =========================================================
# ============ DEMONSTRACAO DA TENDENCIA DA SERIE =========

trend = decomposicao_casos.trend

mascara_na = ~np.isnan(trend)
trend = trend[mascara_na]

x_trend = np.arange(len(trend))
y_trend = trend

z = np.polyfit(x_trend, y_trend, 1)
tendencia = np.poly1d(z)

plt.figure(figsize=(12, 6))
plt.plot(trend.index, y_trend)
plt.plot(trend.index, tendencia(x_trend), "r--", linewidth=2)
plt.title("Tendencia da Serie Temporal")

# =========================================================
# ============ DEMONSTRACAO DA SASIONALIDADE DA SERIE =====

plt.figure(figsize=(12, 6))
decomposicao_casos.seasonal.iloc[:24].plot(marker='o')
plt.title("Sasionalidade da Serie Temporal (Zoom 2 anos)")
plt.grid(True, alpha=0.3)
plt.show()

# =========================================================
# ============ PREVIVASÃO DA SERIE TEMPORAL ===============

meses_teste = 69 # quantidade adequada para eliminar outlier
conjunto_treinamento = df_dengue_brasil.iloc[:-meses_teste, :]
conjunto_teste = df_dengue_brasil.iloc[-meses_teste:, :]



# ============ Holt-Winters
modelo_holtWinters = ExponentialSmoothing(endog=conjunto_treinamento.dengue_cases,
                                         trend='add',
                                         seasonal='mul',
                                         seasonal_periods=12).fit()

prediction_holtWinters = modelo_holtWinters.forecast(steps=len(conjunto_teste))

conjunto_treinamento['dengue_cases'].iloc[:].plot(figsize = (8, 6))
conjunto_teste['dengue_cases'][:].plot(legend=True, label='Conjunto Teste', color='orange')
prediction_holtWinters.plot(legend = True, label='Previsão', color='green')
plt.title('previsão Holt-Winters')
plt.figure(figsize=(12, 6))

# ============ Arima
modelo_arima = auto_arima(y=conjunto_treinamento['dengue_cases'], m=12)

prediction_arima = pd.Series(modelo_arima.predict(n_periods=len(conjunto_teste)))
conjunto_treinamento['dengue_cases'][:].plot(figsize= (8, 6), color='orange')
conjunto_teste['dengue_cases'][:].plot(color='green')
prediction_arima.plot(legend = True, label='Previsão', figsize=(12, 6))
plt.title('Previão Arima')
plt.show()

# ============ previsão do "futuro"
modelo_arima_futuru = auto_arima(y=df_dengue_brasil['dengue_cases'], m=12)

prediction_arima_futuru = pd.Series(modelo_arima_futuru.predict(n_periods=48))
df_dengue_brasil['dengue_cases'][:].plot(figsize= (8, 6))
prediction_arima_futuru.plot(legend = True, label='Previsão')
plt.title("""Previsão Utilizando o Modelo Arima Para os "Futuros" 4 Anos""")
plt.show()
