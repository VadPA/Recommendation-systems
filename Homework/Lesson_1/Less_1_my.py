import pandas as pd
import numpy as np


def hit_rate(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    flags = np.isin(bought_list, recommended_list)

    hit_rate = (flags.sum() > 0) * 1

    return hit_rate


def hit_rate_at_k(recommended_list, bought_list, k=5):
    return hit_rate(recommended_list[:k], bought_list)


def precision(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    flags = np.isin(bought_list, recommended_list)

    precision = flags.sum() / len(recommended_list)

    return precision


def precision_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    bought_list = bought_list
    recommended_list = recommended_list[:k]

    flags = np.isin(bought_list, recommended_list)

    precision = flags.sum() / len(recommended_list)

    return precision


def money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    bought_list = bought_list
    recommended_list = recommended_list[:k]

    flags = np.isin(recommended_list, bought_list) * 1

    money_precision = round(sum(flags * prices_recommended[:k]) / sum(prices_recommended[:k]), 3)

    return money_precision


def recall(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    flags = np.isin(bought_list, recommended_list)

    recall = flags.sum() / len(bought_list)

    return recall


def recall_at_k(recommended_list, bought_list, k=5):
    return recall(recommended_list[:k], bought_list)


def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]
    prices_recommended = np.array(prices_recommended)[:k]
    prices_bought = np.array(prices_bought)

    flags = np.isin(bought_list, recommended_list) * 1

    money_recall = round(sum(flags * prices_bought) / prices_bought.sum(), 3)

    return money_recall


def ap_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]

    flags = np.isin(recommended_list, bought_list)

    if sum(flags) == 0:
        return 0

    sum_ = 0
    for i in range(0, k):

        if flags[i] == True:
            p_k = precision_at_k(recommended_list, bought_list, k=i + 1)
            sum_ += p_k

    result = sum_ / sum(flags)

    return result


def ap_k_v2(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]

    relevant_indexes = np.nonzero(np.isin(recommended_list, bought_list))[0]
    if len(relevant_indexes) == 0:
        return 0

    amount_relevant = len(relevant_indexes)

    sum_ = sum(
        [precision_at_k(recommended_list, bought_list, k=index_relevant + 1) for index_relevant in relevant_indexes])

    return sum_ / amount_relevant


def map_k(recommended_list, users, k=5):
    result = [ap_k(recommended_list, el) for el in users['purchased_items']]
    return round(sum(result) / users.shape[0], 3)


def reciprocal_rank(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]

    relevant_indexes = np.nonzero(np.isin(recommended_list, bought_list))[0]

    if len(relevant_indexes) >= 1:
        own_relevant_index = relevant_indexes[0] + 1
    else:
        return 0

    return 1 / (own_relevant_index)


def mean_reciprocal_rank(recommended_list, users, k=5):
    result = [reciprocal_rank(recommended_list, el, k) for el in users['purchased_items']]
    return round(sum(result) / users.shape[0], 3)


recommended_list = [143, 156, 1134, 991, 27, 1543, 3345, 533, 11, 43]  # id товаров рекомендованных
bought_list = [521, 32, 143, 991]  # id товаров купленных
prices_recommended = [400, 60, 40, 40, 90, 55, 70, 25, 10, 35]  # цена рекомендованных товаров
prices_bought = [20, 140, 40, 40]  # цена купленных товаров

df_ = [{"userID": 0, "purchased_items": [521, 32, 143, 991], "prices_bought": [20, 140, 40, 40]},
       {"userID": 1, "purchased_items": [27, 15, 100, 1543], "prices_bought": [90, 100, 70, 55]},
       {"userID": 2, "purchased_items": [156, 32, 143, 991], "prices_bought": [60, 30, 40, 40]}, ]

df = pd.DataFrame(df_, columns=['userID', 'purchased_items', 'prices_bought'])
print(df)

print(hit_rate(recommended_list, bought_list))

print(hit_rate_at_k(recommended_list, bought_list, 3))

print(precision(recommended_list, bought_list))

print(money_precision_at_k(recommended_list, bought_list, prices_recommended, 5))

print(money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, 5))

print(ap_k_v2(recommended_list, bought_list, k=5))

print(ap_k(recommended_list, bought_list, k=5))

print(map_k(recommended_list, users=df, k=3))

print(mean_reciprocal_rank(recommended_list, users=df, k=3))

print()
