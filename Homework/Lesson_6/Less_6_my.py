import itertools

import pandas as pd
import numpy as np

# Модель второго уровня
from lightgbm import LGBMClassifier

import os, sys

module_path = os.path.abspath(os.path.join(os.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)

# Написанные нами функции
from metrics import precision_at_k, recall_at_k
from utils import prefilter_items
from recommenders import MainRecommender

# Read data
data = pd.read_csv('C:/Users/Вадим/Desktop/GeekBrains/Recommendation-systems/Lectures/Lecture_2/webinar_2/webinar_2'
                   '/data/retail_train.csv')
item_features = pd.read_csv(
    'C:/Users/Вадим/Desktop/GeekBrains/Recommendation-systems/Lectures/Lecture_2/webinar_2/webinar_2'
    '/data/product.csv')
user_features = pd.read_csv(
    'C:/Users/Вадим/Desktop/GeekBrains/Recommendation-systems/Lectures/Lecture_2/webinar_2/webinar_2'
    '/data/hh_demographic.csv')

# Process features dataset
ITEM_COL = 'item_id'
USER_COL = 'user_id'
# column processing
item_features.columns = [col.lower() for col in item_features.columns]
user_features.columns = [col.lower() for col in user_features.columns]

item_features.rename(columns={'product_id': ITEM_COL}, inplace=True)
user_features.rename(columns={'household_key': USER_COL}, inplace=True)

# Split dataset for train, eval, test
# Важна схема обучения и валидации!
# -- давние покупки -- | -- 6 недель -- | -- 3 недель --
# подобрать размер 2-ого датасета (6 недель) --> learning curve (зависимость метрики recall@k от размера датасета)


VAL_MATCHER_WEEKS = 6
VAL_RANKER_WEEKS = 3

# берем данные для тренировки matching модели
data_train_matcher = data[data['week_no'] < data['week_no'].max() - (VAL_MATCHER_WEEKS + VAL_RANKER_WEEKS)]

# берем данные для валидации matching модели
data_val_matcher = data[(data['week_no'] >= data['week_no'].max() - (VAL_MATCHER_WEEKS + VAL_RANKER_WEEKS)) &
                        (data['week_no'] < data['week_no'].max() - (VAL_RANKER_WEEKS))]

# берем данные для тренировки ranking модели
data_train_ranker = data_val_matcher.copy()  # Для наглядности. Далее мы добавим изменения, и они будут отличаться

# берем данные для теста ranking, matching модели
data_val_ranker = data[data['week_no'] >= data['week_no'].max() - VAL_RANKER_WEEKS]


def print_stats_data(df_data, name_df):
    print(name_df)
    print(f"Shape: {df_data.shape} Users: {df_data[USER_COL].nunique()} Items: {df_data[ITEM_COL].nunique()}")


print_stats_data(data_train_matcher, 'train_matcher')
print_stats_data(data_val_matcher, 'val_matcher')
print_stats_data(data_train_ranker, 'train_ranker')
print_stats_data(data_val_ranker, 'val_ranker')

print('#видим разброс по пользователям и товарам')

print(data_train_matcher.head(2))

# Prefilter items
n_items_before = data_train_matcher['item_id'].nunique()

data_train_matcher = prefilter_items(data_train_matcher, item_features=item_features, take_n_popular=5000)

n_items_after = data_train_matcher['item_id'].nunique()
print('Decreased # items from {} to {}'.format(n_items_before, n_items_after))

# Make cold-start to warm-start
# ищем общих пользователей
common_users = data_train_matcher.user_id.values

data_val_matcher = data_val_matcher[data_val_matcher.user_id.isin(common_users)]
data_train_ranker = data_train_ranker[data_train_ranker.user_id.isin(common_users)]
data_val_ranker = data_val_ranker[data_val_ranker.user_id.isin(common_users)]

print_stats_data(data_train_matcher, 'train_matcher')
print_stats_data(data_val_matcher, 'val_matcher')
print_stats_data(data_train_ranker, 'train_ranker')
print_stats_data(data_val_ranker, 'val_ranker')

# Init/train recommender
recommender = MainRecommender(data_train_matcher)

# Варианты, как получить кандидатов
# Можно потом все эти варианты соединить в один
# (!) Если модель рекомендует < N товаров, то рекомендации дополняются топ-популярными товарами до N

# Берем тестового юзера 2375
print(recommender.get_als_recommendations(2375, N=5))
print(recommender.get_own_recommendations(2375, N=5))
print(recommender.get_similar_items_recommendation(2375, N=5))
print(recommender.get_similar_users_recommendation(2375, N=5))

# Eval recall of matching
ACTUAL_COL = 'actual'

result_eval_matcher = data_val_matcher.groupby(USER_COL)[ITEM_COL].unique().reset_index()
result_eval_matcher.columns = [USER_COL, ACTUAL_COL]
print(result_eval_matcher.head(2))

# N = Neighbors
N_PREDICT = 50

# для понятности расписано все в строчку, без функций, ваша задача уметь оборачивать все это в функции
result_eval_matcher['own_rec'] = \
    result_eval_matcher[USER_COL].apply(lambda x: recommender.get_own_recommendations(x, N=N_PREDICT))
result_eval_matcher['sim_item_rec'] = \
    result_eval_matcher[USER_COL].apply(lambda x: recommender.get_similar_items_recommendation(x, N=50))
result_eval_matcher['als_rec'] = \
    result_eval_matcher[USER_COL].apply(lambda x: recommender.get_als_recommendations(x, N=50))
result_eval_matcher['sim_user_rec'] = \
    result_eval_matcher[USER_COL].apply(lambda x: recommender.get_similar_users_recommendation(x, N=50))


# Пример оборачивания
# сырой и простой пример как можно обернуть в функцию
def evalRecall(df_result, target_col_name, recommend_model):
    result_col_name = 'result'
    df_result[result_col_name] = df_result[target_col_name].apply(lambda x: recommend_model(x, N=25))
    return df_result.apply(lambda row: recall_at_k(row[result_col_name], row[ACTUAL_COL], k=N_PREDICT), axis=1).mean()


evalRecall(result_eval_matcher, USER_COL, recommender.get_own_recommendations)


def calc_recall(df_data, top_k):
    for col_name in df_data.columns[2:]:
        yield col_name, df_data.apply(lambda row: recall_at_k(row[col_name], row[ACTUAL_COL], k=top_k), axis=1).mean()


def calc_precision(df_data, top_k):
    for col_name in df_data.columns[2:]:
        yield col_name, df_data.apply(lambda row: precision_at_k(row[col_name], row[ACTUAL_COL], k=top_k),
                                      axis=1).mean()


# Recall@50 of matching
TOPK_RECALL = 50
sorted(calc_recall(result_eval_matcher, TOPK_RECALL), key=lambda x: x[1], reverse=True)

# Precision@5 of matching
TOPK_PRECISION = 5
sorted(calc_precision(result_eval_matcher, TOPK_PRECISION), key=lambda x: x[1], reverse=True)

# Ranking part
# Обучаем модель 2-ого уровня на выбранных кандидатах

# - Обучаем на data_train_ranking
# - Обучаем *только* на выбранных кандидатах
# - Я *для примера* сгенерирую топ-50 кадидиатов через get_own_recommendations
# - (!) Если юзер купил < 50 товаров, то get_own_recommendations дополнит рекоммендации топ-популярными


# Подготовка данных для трейна
# взяли пользователей из трейна для ранжирования
df_match_candidates = pd.DataFrame(data_train_ranker[USER_COL].unique())
df_match_candidates.columns = [USER_COL]

# собираем кандитатов с первого этапа (matcher)
df_match_candidates['candidates'] = \
    df_match_candidates[USER_COL].apply(lambda x: recommender.get_own_recommendations(x, N=N_PREDICT))

print(df_match_candidates.head(2))

df_items = \
    df_match_candidates.apply(lambda x: pd.Series(x['candidates']), axis=1).stack().reset_index(level=1, drop=True)
df_items.name = 'item_id'
df_match_candidates = df_match_candidates.drop('candidates', axis=1).join(df_items)
print(df_match_candidates.head(4))

# Check warm start
print_stats_data(df_match_candidates, 'match_candidates')

### Создаем трейн сет для ранжирования с учетом кандидатов с этапа 1
df_ranker_train = data_train_ranker[[USER_COL, ITEM_COL]].copy()
df_ranker_train['target'] = 1  # тут только покупки
print(df_ranker_train.head())

# Не хватает нулей в датасете, поэтому добавляем наших кандитатов в качество нулей
df_ranker_train = df_match_candidates.merge(df_ranker_train, on=[USER_COL, ITEM_COL], how='left')

# чистим дубликаты
df_ranker_train = df_ranker_train.drop_duplicates(subset=[USER_COL, ITEM_COL])

df_ranker_train['target'].fillna(0, inplace=True)

print(df_ranker_train.target.value_counts())
print(df_ranker_train.head(2))

# (!) На каждого юзера 50 item_id-кандидатов
print(df_ranker_train['target'].mean())

# - Пока для простоты обучения выберем LightGBM c loss = binary. Это классическая бинарная классификация
# - Это пример *без* генерации фич

# Подготавливаем фичи для обучения модели
print(item_features.head(2))
print(user_features.head(2))

df_ranker_train = df_ranker_train.merge(item_features, on='item_id', how='left')
df_ranker_train = df_ranker_train.merge(user_features, on='user_id', how='left')

print(df_ranker_train.head(2))

# ** Фичи user_id: **
# - Средний чек
# - Средняя сумма покупки 1 товара в каждой категории
# - Кол - во покупок в каждой категории
# - Частотность покупок раз / месяц
# - Долю покупок в выходные
# - Долю покупок утром / днем / вечером

# ** Фичи item_id **:
# - Кол - во покупок в неделю
# - Среднее кол - во покупок 1 товара в категории в неделю
# - (Кол - во покупок в неделю) / (Среднее ол-во покупок 1 товара в категории в неделю)
# - Цена(Можно посчитать из retil_train.csv)
# - Цена / Средняя цена товара в категории

# ** Фичи пары user_id - item_id **
# - (Средняя сумма покупки 1 товара в каждой категории (берем категорию item_id)) - (Цена item_id)
# - (Кол - во покупок юзером конкретной категории в неделю) - (Среднее
#     кол-во покупок всеми юзерами конкретной категории в неделю)
# - (Кол - во покупок юзером конкретной категории в неделю) / (Среднее кол-во покупок всеми юзерами конкретной
#     категории в неделю)

print(df_ranker_train.head())

X_train = df_ranker_train.drop('target', axis=1)
y_train = df_ranker_train[['target']]

cat_feats = X_train.columns[2:].tolist()
X_train[cat_feats] = X_train[cat_feats].astype('category')

print(cat_feats)

# Обучение модели ранжирования

lgb = LGBMClassifier(objective='binary',
                     max_depth=8,
                     n_estimators=300,
                     learning_rate=0.05,
                     categorical_column=cat_feats,
                     n_jobs=-1,
                     #                      verbose=0
                     )

lgb.fit(X_train, y_train)

train_preds = lgb.predict_proba(X_train)

df_ranker_predict = df_ranker_train.copy()
df_ranker_predict['proba_item_purchase'] = train_preds[:, 1]
print(df_ranker_predict['proba_item_purchase'][:10])

result_eval_ranker = data_val_ranker.groupby(USER_COL)[ITEM_COL].unique().reset_index()
result_eval_ranker.columns = [USER_COL, ACTUAL_COL]
print(result_eval_ranker.head(2))

result_eval_ranker['own_rec'] = \
    result_eval_ranker[USER_COL].apply(lambda x: recommender.get_own_recommendations(x, N=N_PREDICT))


def rerank(user_id):
    return df_ranker_predict[df_ranker_predict[USER_COL] == user_id].sort_values('proba_item_purchase',
                                                                                 ascending=False).head(
        5).item_id.tolist()


# max_depth = [1, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
max_depth = [i for i in range(5, 31)]  # 15
num_leaves = [i for i in range(8, 51, 2)]  # 50
min_child_samples = 9  # [i for i in range(4, 31, 2)]
n_estimators = 600  # [i for i in range(100, 601, 50)]
lrate = np.linspace(0.01, 0.9, 15, endpoint=True)
params = list(itertools.product(max_depth, num_leaves, repeat=1))

# params = list(itertools.product(max_depth, num_leaves, min_child_samples, n_estimators, lrate, repeat=1))
print(params)

result_dict = {}
i = 0

for max_depth, num_leaves in params:
    lgb = LGBMClassifier(objective='binary',
                         boosting_type='gbdt',
                         max_depth=max_depth,
                         num_leaves=num_leaves,
                         min_child_samples=9,
                         n_estimators=600,
                         learning_rate=0.32785714285714285,
                         categorical_column=cat_feats,
                         n_jobs=-1,
                         )

    lgb.fit(X_train, y_train)
    train_preds = lgb.predict_proba(X_train)
    df_ranker_predict = df_ranker_train.copy()
    df_ranker_predict['proba_item_purchase'] = train_preds[:, 1]
    result_eval_ranker['reranked_own_rec'] = result_eval_ranker[USER_COL].apply(lambda user_id: rerank(user_id))
    tuple_res = sorted(calc_precision(result_eval_ranker, TOPK_PRECISION), key=lambda x: x[1], reverse=True)[0]
    iteration_tuple = (("max_depth", max_depth),
                       ("num_leaves", num_leaves),
                       ("method", tuple_res[0]),
                       ("precision", tuple_res[1]))
    result_dict[str(i)] = dict(iteration_tuple)
    i += 1


print(result_dict)

print()