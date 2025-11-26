# %%
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/data_2000_2019.csv')
df['data'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

df_dengue = df.groupby('data')['dengue_cases'].sum()
df_dengue = pd.DataFrame(df_dengue)
# df_dengue = df[['data', 'dengue_cases']].set_index('data')
# df_dengue.head()
df_dengue.plot()