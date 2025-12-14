# Аналитика кредитных заявок  
## Выявление минимального набора признаков для получения кредита

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 100)


# ## 1. Загрузка и объединение данных

# Даны две таблицы:
# - `application_info.csv` — характеристики заявок,
# - `default_flg.csv` — информация о дефолте.

# Таблицы объединяются по идентификатору заявки `id`.

app = pd.read_csv('application_info.csv')
dfl = pd.read_csv('default_flg.csv')

df = app.merge(dfl, on='id')

df.head()

## 2. Очистка данных

# - пропуски удаляются,
# - используется только выборка с известным значением `default_flg`.


df = df.dropna().copy()
df['default_flg'] = df['default_flg'].astype(int)

print('Размер выборки:', df.shape)
print('Средний дефолт-рейт:', round(df["default_flg"].mean(), 4))

# ## 3. Базовый уровень риска

# Рассчитаем общий дефолт-рейт — он будет использоваться как точка сравнения 
# для отдельных признаков.

base_dr = df['default_flg'].mean()
base_dr


# ## 4. Анализ отдельных признаков

# Для каждого признака считаем дефолт-рейт по значениям и оцениваем:
# - есть ли выраженная разница в риске,
# - можно ли использовать признак как простой фильтр.

def simple_analysis(col):
    tab = (
        df.groupby(col)['default_flg']
        .agg(['count','mean'])
        .rename(columns={'mean':'default_rate'})
        .sort_values('default_rate')
    )
    return tab

display(simple_analysis('appl_rej_cnt'))
display(simple_analysis('out_request_cnt').head(10))
display(simple_analysis('region_rating'))

# ### Комментарий

# - Наличие хотя бы одного отказа (`appl_rej_cnt > 0`) резко увеличивает риск дефолта.
# - Большое число внешних запросов также связано с повышенным риском.
# - Региональный рейтинг демонстрирует монотонную связь с дефолтом.

# Анализ Score_bki через корзины
df['score_bin'] = pd.qcut(df['Score_bki'], 5, duplicates='drop')

df.groupby('score_bin')['default_flg'].agg(['count','mean'])

# ### Комментарий

# Кредитный скор (`Score_bki`) является наиболее информативным признаком:
# при ухудшении значения дефолт-рейт возрастает монотонно.

# ## 5. Выбор минимального набора признаков

# На основе анализа отдельных признаков отбираем **минимальный набор**, который:
# - сильно влияет на риск,
# - легко интерпретируется,
# - не требует сложных расчётов.

# Выбранные признаки:
# 1. `Score_bki` — основной индикатор кредитного риска,
# 2. `appl_rej_cnt` — история отказов,
# 3. `out_request_cnt` — кредитная активность,
# 4. `region_rating` — региональный фактор.


rule = (
    (df['Score_bki'] <= -1.7) &
    (df['appl_rej_cnt'] == 0) &
    (df['out_request_cnt'] <= 3) &
    (df['region_rating'] >= 50)
)

approved = df[rule]

print('Доля заявок, проходящих фильтр:', round(len(approved)/len(df), 3))
print('Дефолт-рейт в отобранной группе:', round(approved["default_flg"].mean(), 3))

# ## 6. Итоговые выводы

# 1. Для базовой фильтрации заявок достаточно **небольшого числа признаков**.
# 2. Наиболее значимый фактор — кредитный скор (`Score_bki`).