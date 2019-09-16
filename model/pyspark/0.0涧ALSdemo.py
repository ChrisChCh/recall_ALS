# -*- coding:utf-8 -*-
"""
这是训练和保存模型的代码
 Desc  :
     Training and saving ALS model.

"""
from pyspark.sql import SparkSession
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import logging
import pyspark
from pyspark import SparkContext


class News_ALS():
    def __init__(self):
        # 返回的推荐结果数
        self.k = 10
        # 灌入算法数据文件路径
        self.path_id = "ratings_id.csv"
        # self.path_device = "../data/ratings_device.csv"
        self.path_model = "../als_model"
        # conf = SparkConf().setAppName("miniProject").setMaster("local[*]")
        # self.sc = SparkContext.getOrCreate(conf)
        self.sc = SparkContext("local", "recommendation")

    # 加载数据&训练模型&保存模型
    def pyspark_als_id_model(self):
        self.spark = SparkSession.builder.appName("news_recommendation").master("local").getOrCreate()
        # SparkSession.Builder
        df = self.spark.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(
            self.path_id)
        # print(df)
        # 读取数据
        df_singer_recommend = df.select("user_id", "article_id", "ratings")
        # 转RDD
        singer_rdd = df_singer_recommend.rdd
        # 缓存
        trainingRDD = singer_rdd.cache()
        # 加载模型
        rank = 10
        numIterations = 10
        # 训练
        model_als = ALS.train(trainingRDD, rank, numIterations)
        try:
            model_als.save(self.sc, path=self.path_model)
        except Exception as err:
            logging.error("save model is fail :{}".format(err))

    # # 保存模型
    # def save_model(self,model_als):
    #     try:
    #         model = self.pyspark_als_id_model()
    #         model.save_model(model_als, self.path_model)
    #     except Exception:
    #         print("保存模型时出错")

    # 使用device号进行als算法。这块不需要。
    # def pyspark_als_device(self,device):
    #     spark = SparkSession.builder.appName("singer_recommendation").master("local").getOrCreate()
    #     df = spark.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(self.path_device)
    #     print(df)
    #     # 读取数据
    #     df_singer_recommend = df.select("device", "article_id", "ratings")
    #     # 转RDD
    #     singer_rdd = df_singer_recommend.rdd
    #     # 缓存
    #     trainingRDD = singer_rdd.cache()
    #     # 加载模型
    #     rank = 10
    #     numIterations = 10
    #     # 训练
    #     model = ALS.train(trainingRDD, rank, numIterations)
    #     # 参数k为返回结果数
    #     recommendedResult = model.recommendProducts(device, self.k)
    #     print(recommendedResult)


if __name__ == '__main__':
    als = News_ALS()
    print("加载数据，训练模型，保存模型.......")
    als.pyspark_als_id_model()