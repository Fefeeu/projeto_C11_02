# 🦟 Análise Temporal de Casos de Dengue no Brasil (2000-2019)

Este projeto aplica técnicas de **Análise de Séries Temporais** para investigar o comportamento dos casos de dengue no Brasil ao longo de duas décadas, identificando padrões sazonais, tendências epidemiológicas e gerando previsões para anos futuros.

## 🎯 Objetivos

- Transformar dados brutos em uma série temporal contínua e tratada.
- Decompor a série para entender **Tendência**, **Sazonalidade** e **Ruído**.
- Analisar o crescimento do nível endêmico da doença por meio de regressão linear sobre a tendência.
- Criar modelos preditivos (*forecast*) robustos para prever novos surtos.

## 🛠️ Tecnologias utilizadas

- **Python 3**
- [Pandas](https://pandas.pydata.org/) — manipulação e limpeza de dados (ETL)
- [Matplotlib](https://matplotlib.org/) — visualização de dados
- [Statsmodels](https://www.statsmodels.org/) — decomposição sazonal (`seasonal_decompose`) e suavização exponencial (Holt-Winters)
- [Pmdarima](https://alkaline-ml.com/pmdarima/) — modelagem SARIMA automatizada (`auto_arima`)
- [NumPy](https://numpy.org/) — transformações matemáticas (log/exp) para estabilização de variância

## 📁 Estrutura do repositório

```
projeto_C11_02/
├── data/
│   └── data_2000_2019.csv   # base histórica de casos de dengue (2000-2019)
├── analise.py                # script principal com toda a análise
├── .gitignore
└── README.md
```

## ⚙️ Como executar

1. Clone o repositório:
   ```bash
   git clone https://github.com/Fefeeu/projeto_C11_02.git
   cd projeto_C11_02
   ```

2. Instale as dependências:
   ```bash
   pip install pandas numpy matplotlib statsmodels pmdarima
   ```

3. Rode a análise:
   ```bash
   python analise.py
   ```

O script vai gerar, em sequência, os seguintes gráficos:
1. Série histórica de casos de dengue no Brasil (2000-2019)
2. Decomposição da série (tendência, sazonalidade e resíduo)
3. Tendência da série com linha de regressão linear
4. Sazonalidade detalhada (zoom nos 2 primeiros anos)
5. Previsão via **Holt-Winters** comparada ao conjunto de teste
6. Previsão via **SARIMA** (`auto_arima`) comparada ao conjunto de teste
7. Previsão para os **próximos 4 anos** ("futuro") usando SARIMA treinado na série completa

## 📊 Metodologia

- Os dados brutos (ano/mês/casos) são convertidos em uma série temporal mensal contínua, com valores nulos preenchidos com zero.
- É aplicado um deslocamento de +1 nos valores para permitir a decomposição multiplicativa mesmo em meses com zero casos.
- A decomposição sazonal separa a série em tendência, sazonalidade e ruído, permitindo visualizar o crescimento do nível endêmico da doença ao longo do tempo.
- Dois modelos de previsão são comparados: **Holt-Winters** (suavização exponencial) e **SARIMA** (via `auto_arima`), ambos avaliados contra um conjunto de teste separado do final da série.

## 📄 Fonte dos dados

Os dados utilizados estão no arquivo `data/data_2000_2019.csv`, contendo o número de casos de dengue por ano e mês no Brasil entre 2000 e 2019.
