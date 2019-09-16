#-*- coding:utf8-*-
# -*- coding:utf-8 -*-
"""
这是加载模型，和给出训练结果的代码
 Desc  :
    Loading ALS model to predict news recommendation.
"""
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark import SparkContext


class Load_Pre_ALs():
    def __init__(self):
        # 返回的推荐结果数
        self.k = 10
        # 模型路径
        self.path_model = "../als_model"
        self.sc = SparkContext("local", "recommendation")

    # 加载模型
    def load_model(self):
        try:
            model = MatrixFactorizationModel.load(self.sc, self.path_model)
            print(model)
            return model
        except Exception:
            print("模型加载出错")
            return {}

    # 得到预测结果
    def rec(self, model, user_id):
        # 参数k为返回结果数
        try:
            recommendedResult = model.recommendProducts(user_id, self.k)
            return recommendedResult
        except Exception as err:
            # logging.info()
            return []


if __name__ == '__main__':
    als = Load_Pre_ALs()
    print("加载模型.......")
    model = als.load_model()
    print("预测推荐结果......")
    rec_id_result = als.rec(model, 7763690)
    print(rec_id_result)