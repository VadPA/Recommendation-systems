import random

import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender
from implicit.nearest_neighbours import bm25_weight, tfidf_weight
from utils import prefilter_items


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS

    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self, df_data, weighting=True, N=5, num_threads=4, n_factors=40, regularization=0.001, iterations=15):
        # your_code. Это необязательная часть. Но если вам удобно что-либо посчитать тут - можно это сделать
        self.iterations = iterations
        self.regularization = regularization
        self.n_factors = n_factors
        self.num_threads = num_threads
        self.N = N

        # Топ покупок каждого юзера
        self.popularity = df_data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.popularity.sort_values('quantity', ascending=False, inplace=True)
        self.popularity = self.popularity[self.popularity['item_id'] != 999999]

        # Топ покупок по всему датасету
        self.popularity_all = df_data.groupby('item_id')['quantity'].count().reset_index()
        self.popularity_all.sort_values('quantity', ascending=False, inplace=True)
        self.popularity_all = self.popularity_all[self.popularity_all['item_id'] != 999999]
        self.popularity_all = self.popularity_all.item_id.tolist()

        self.user_item_matrix = self.prepare_matrix(df_data)  # pd.DataFrame

        self.user_item_matrix_tfidf = tfidf_weight(self.user_item_matrix.T).T

        self.id_to_itemid, self.id_to_userid, \
        self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)

        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        self.sparse_user_item = csr_matrix(self.user_item_matrix)

        self.model = self.fit(self.sparse_user_item, self.num_threads, self.n_factors,
                              self.regularization, self.iterations)
        self.model_tfidf = self.fit_tfidf(self.user_item_matrix_tfidf, self.num_threads, self.n_factors,
                                          self.regularization, self.iterations)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix, self.num_threads)

    @staticmethod
    def prepare_matrix(df_data: pd.DataFrame):
        """Создаём user_item матрицу"""
        user_item_matrix = pd.pivot_table(df_data,
                                          index='user_id', columns='item_id',
                                          values='quantity',
                                          aggfunc='count',
                                          fill_value=0
                                          )
        user_item_matrix = user_item_matrix.astype(float)
        return user_item_matrix

    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix, num_threads):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=1, num_threads=num_threads)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return own_recommender

    @staticmethod
    def fit(sparse_user_item, num_threads, n_factors, regularization, iterations):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)
        model.fit(sparse_user_item.T.tocsr())  # csr_matrix(user_item_matrix).T.tocsr()

        return model

    @staticmethod
    def fit_tfidf(user_item_matrix_tfidf, num_threads, n_factors, regularization, iterations,
                  calculate_training_loss=True):
        """Обучает ALS"""

        model_tfidf = AlternatingLeastSquares(factors=n_factors,
                                              regularization=regularization,
                                              iterations=iterations,
                                              calculate_training_loss=calculate_training_loss,
                                              num_threads=num_threads)

        model_tfidf.fit(csr_matrix(user_item_matrix_tfidf).T.tocsr())  #

        return model_tfidf

    def get_similar_item(self, item_id):
        """Находит товар, похожий на item_id"""
        n = 2
        while True:
            recs = self.model.similar_items(self.itemid_to_id[item_id], n)
            top_rec = recs[n - 1][0]
            if self.id_to_itemid[top_rec] != 999999:
                break
            else:
                n += 1
        return self.id_to_itemid[top_rec]

    def extend_with_top_popular(self, recommendations):
        """Если кол-во рекоммендаций < N, то дополняем их топ-популярными"""

        if len(recommendations) < self.N:
            recommendations.extend(self.popularity_all[len(recommendations): self.N + 1])
            # recommendations = recommendations[:N]

        return recommendations

    def get_recommendations(self, user):
        """Рекомендуем топ-N товаров"""

        if user not in self.userid_to_id:
            user = random.choice(list(self.userid_to_id.keys()))

        res = [self.id_to_itemid[rec[0]] for rec in
               self.model.recommend(userid=self.userid_to_id[user],
                                    user_items=self.sparse_user_item,
                                    N=self.N,
                                    filter_already_liked_items=False,
                                    filter_items=[self.itemid_to_id[999999]],
                                    recalculate_user=True)]
        return res

    def get_own_recommendations(self, user):
        """Рекомендуем товары среди тех, которые юзер уже купил"""

        return self.get_recommendations(user)

    def get_similar_items_recommendation(self, user):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        top_users_popularity = self.popularity[self.popularity['user_id'] == user].head(self.N)

        res = top_users_popularity['item_id'].apply(lambda x: self.get_similar_item(x)).tolist()

        if len(res) < self.N:
            res = self.extend_with_top_popular(res)

        return res

    def get_similar_users_recommendation(self, user):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        res = []

        # Находим топ-N похожих пользователей
        similar_users = self.model.similar_users(self.userid_to_id[user], N=self.N + 1)
        similar_users = [self.id_to_userid[rec[0]] for rec in similar_users]
        similar_users = similar_users[1:]

        for _user_id in similar_users:
            res.extend(self.get_own_recommendations(_user_id))

        if len(res) < self.N:
            res = self.extend_with_top_popular(res)

        return res
