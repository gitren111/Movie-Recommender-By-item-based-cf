import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#一、数据集加载
ra_data_path = 'ml-latest-small/ml-latest-small/ratings.csv'
ra_cache_dir = 'cache'
cache_path = os.path.join(ra_cache_dir,'rating_matrix_cache')

def load_data(ra_data_path,cache_path):
    print('开始分批加载数据集...')
    if not os.path.exists(ra_cache_dir):
        os.makedirs(ra_cache_dir)

    if os.path.exists(cache_path):
        print('加载缓冲中...')
        ratings_matrix = pd.read_pickle(cache_path)
        print('从缓存加载数据集完毕')
    else:
        dtype = {'userId': np.int32, 'movieId': np.int32, 'rating': np.float32}
        print('加载新数据中')
        #加载前三列数据：用户ID、电影ID、评分
        ratings = pd.read_csv(ra_data_path,dtype=dtype,usecols=range(3))
        #转换为用户-电影评分矩阵
        ratings_matrix = pd.pivot_table(data=ratings,index=['userId'],columns=['movieId'],values='rating')
        ratings_matrix.to_pickle(cache_path)
        print('数据加载完毕')
    return ratings_matrix
ratings_matrix = load_data(ra_data_path,cache_path)
print(ratings_matrix.shape)
print(ratings_matrix.head())

#二、相似度计算
"""
用“用户-评分”矩阵和皮尔森相关系数公式，计算基于客户的相似度矩阵
"""
def compute_cosine_similarity(ratings_matrix):
    item_similarity_cache_path = os.path.join(ra_cache_dir,'item_similarity_cache')

    #基于皮尔森相关系数计算相似度
    if not os.path.exists(ra_cache_dir):
        os.makedirs(ra_cache_dir)
    if os.path.exists(item_similarity_cache_path):
        print('正从缓存加载物品相似度矩阵')
        similarity = pd.read_pickle(item_similarity_cache_path)
    else:
        print('开始计算物品相似度矩阵')
        similarity = cosine_similarity(ratings_matrix.T.fillna(0))
        similarity.to_pickle(item_similarity_cache_path)
    print('物品相似度矩阵加载完毕')
    return similarity

item_similar = compute_cosine_similarity(ratings_matrix)
print('用户相似度矩阵预览和大小')
print(item_similar.shape)
print(item_similar.head())


#三、预测评分函数（基于物品协同过滤）：item-based CF
#1、item-based CF：基于物品协同过滤算法预测评分
def predict_itemBasedCF(uid,iid,ratings_matrix,item_similar,threshold=0.1):
    #找到与电影u相似的近邻电影(正相关)
    similar_items = item_similar[uid].drop(labels=[uid]).dropna()
    similar_items = similar_items.where(similar_items>threshold).dropna()

    if similar_items.empty is True:
        raise Exception('电影{}没有相似的电影'.format(uid))

    #从近邻电影中找出u客户评分过电影，构建有效电影序列
    id_item = set(ratings_matrix[uid].dropna().index) & set(similar_items.index)
    finally_similar_items = similar_items.loc[list(id_item)]
    #print(f'finally_similar_items{finally_similar_items}')

    #（5）公式计算
    sum_up = 0
    sum_down = 0
    for sim_itemid,similarity in finally_similar_items.items():
        #用户u对近邻相似电影评分
        user_rated_movies = ratings_matrix.loc[uid,sim_itemid]

        # 分子计算
        if pd.notna(user_rated_movies):
            sum_up += similarity * user_rated_movies  # 防止迭代到nan值导致分子为nan
        # 分母计算
        sum_down += abs(similarity)

    # 最后将所有相似物品累加再除分母相似度累加
    if sum_down == 0:
        print('分母为0不可计算')
        raise None
    predict_rating = sum_up / sum_down
    predict_rating = min(predict_rating, 5)
    return round(predict_rating, 2)

#3、根据筛选条件来预测评分
def itemBase_predict_all(uid,item_ids,ratings_matrix,item_similarity):
    for iid in item_ids:
        if pd.notna(ratings_matrix.loc[uid,iid]):
            continue#如果已经评分就跳过该电影不进行预测
        try:
            rating = predict_itemBasedCF(uid,iid,ratings_matrix,item_similarity)
        except Exception as e:
            print('基于物品的协同过滤算法无法预测结果')
        else:
            yield uid,iid,rating


def predict_all(uid,ratings_matrix,similar_matrix,filter_rule=None):
    #获取用户评分数据
    user_ratings = ratings_matrix.loc[uid]
    #筛选用户未评分的电影
    unrated_movies = user_ratings[user_ratings.isna()].index

    if not filter_rule:
        #筛选用户没有评分的电影
        item_ids = unrated_movies

    #筛选掉非热门电影
    elif filter_rule == 'unhot':
        count = ratings_matrix.count()
        hot_movies = count.where(count>10).dropna().index
        # 确保只预测用户未评分且是热门电影
        item_ids = set(unrated_movies)&set(hot_movies)

    # 筛选低评分的电影
    elif filter_rule == 'rated':
        user_ratings = ratings_matrix.loc[uid]
        high_rate__movies = user_ratings[user_ratings>2.5].index
        # 确保只预测用户未评分且是高分电影
        item_ids = set(unrated_movies) & set(high_rate__movies)

    #筛选掉非热门且无评分电影
    elif set(filter_rule) == set(['unhot','rated']):
        count = ratings_matrix.count()
        ids1 = count.where(count>10).dropna().index
        user_ratings = ratings_matrix.loc[uid]
        ids2 = user_ratings[user_ratings>2.5].index
        item_ids = set(ids1)&set(ids2)&set(unrated_movies)
    else:
        raise Exception('无效的筛选参数')
    yield from itemBase_predict_all(uid, item_ids, ratings_matrix, similar_matrix)

#四、预测电影评分

def topk_itemBase_predict(k,filter_rule=None):
    results = predict_all(21,ratings_matrix,item_similar,filter_rule)
    results = sorted(results,key=lambda x:x[2],reverse=True)[:k]
    print('基于物品的协同过滤评分预测，预测用户评分最高的10部电影为：')
    for uid,itemid,predict_rating in results:
        print('预测出用户{}对电影{}的评分为：{:.2f}'.format(uid, itemid, predict_rating))
    return results
topk_itemBase_predict(100,filter_rule='unhot')

