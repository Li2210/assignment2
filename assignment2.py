#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml.feature import Word2Vec
from pyspark.ml.recommendation import ALS
import pandas as pd

# https://stackoverflow.com/questions/57451719/since-spark-2-3-the-queries-from-raw-json-csv-files-are-disallowed-when-the-ref
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

FILE_NAME = "tweets.json"

original_data = spark.read.option("multiline","true").json(FILE_NAME).cache()


# In[2]:


# user_reply:(user_id, replyto_id)
user_replyTo = original_data.select("user_id", "replyto_id").where("replyto_id is not null")

# user_retweet:(user_id, retweet_id)
user_retweet = original_data.select("user_id", "retweet_id").where("retweet_id is not null")

# combine two data:(user_id, relatedTweetId)
user_combine = user_retweet.union(user_replyTo).toDF("user_id","relatedTweetId")

user_data = user_combine.rdd.map(lambda row:(row[0], str(row[1]))).groupByKey().mapValues(list)                            .toDF(["user_id", "data"])

# user_data.show(200,truncate=False)


# In[3]:


# idf train data
# https://spark.apache.org/docs/latest/ml-features.html#tf-idf
hashingTF = HashingTF(inputCol="data", outputCol="rawFeatures",numFeatures=131072)
featurizedData = hashingTF.transform(user_data)

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

clearData = rescaledData.drop("data").drop("rawFeatures")

# clearData.show(truncate=False)


# In[19]:


# 2179124364
selectId = 2179124364

target_user = clearData.filter("user_id =" + str(selectId)).collect()[0][1]

other_user = clearData.filter("user_id !=" + str(selectId)).rdd.map(lambda row : (row[0], row[1]))

def consine(a,b):
    return a.dot(b) / (a.norm(2) * b.norm(2))

found_user = other_user.map(lambda row:(row[0], consine(target_user, row[1])))
found_user.sortBy(lambda row:row[1], ascending=False).take(5)


# In[20]:


# Word2Vec
tweet_text = original_data.select('id', split("text",' ').alias("words"))

word2Vec = Word2Vec(vectorSize=300, minCount=1, inputCol="words", outputCol="model")
model = word2Vec.fit(tweet_text)

result = model.transform(tweet_text).drop("words")
tweet_word = result.rdd.map(lambda row:(row[0], row[1])).reduceByKey(lambda v1, v2: v1 + v2)

# word_vec = model.getVectors()
# tweet_word = tweet_text.select("id", posexplode("words").alias("pos", "word")).join(word_vec, "word")\
#                        .select("id","vector").rdd.map(lambda row:(row[0], row[1]))\
#                        .reduceByKey(lambda v1, v2: v1 + v2)
# tweet_word.take(10)

user_vec = user_combine.rdd.map(lambda row: (row[1], row[0]))                            .join(tweet_word)                            .map(lambda row:(row[1][0], row[1][1]))                            .reduceByKey(lambda v1,v2: v1 + v2)
# user_vec.take(10)
target_user_vec = user_vec.filter(lambda user_vec_pair:user_vec_pair[0] == selectId).collect()[0][1]
other_user_vec = user_vec.filter(lambda user_vec_pair:user_vec_pair[0] != selectId).map(lambda row : (row[0], row[1]))

found_user_word = other_user_vec.map(lambda row:(row[0], consine(target_user_vec, row[1])))
found_user_word.sortBy(lambda row:row[1], ascending=False).take(5)

# tweet_text.show(10, truncate=False)


# In[6]:


# get user_id 
all_user_ids = original_data.select("user_id").distinct().rdd.map(lambda row:row[0]).collect()

# create dictionary for user_id
user_id_to_index = dict()
for user_id in all_user_ids:
    user_id_to_index[user_id] = len(user_id_to_index)

def extract_id(rows):
    return [(mention_user[0]) for mention_user in rows[0]]

# extract mentioned user id
all_mention_users = original_data.select("user_mentions")                                  .filter("user_mentions is not null")                                  .rdd.flatMap(extract_id)                                  .distinct().collect()


# create dictionary for mentioned user id
mention_id_to_index = dict()
for mention_user_id in all_mention_users:
    mention_id_to_index[mention_user_id] = len(mention_id_to_index)

sc.broadcast(user_id_to_index)
sc.broadcast(mention_id_to_index)


# In[8]:


def extract_data(rows):
    user_id = rows[0]
    return [(user_id, mention_id_to_index[mention_user[0]], 1) for mention_user in rows[1]]

def create_key(rows):
    user_id = rows[0] 
    mention_user = rows[1] 
    counter = rows[2]
    return ((user_id,mention_user), counter)

mention_user = original_data.select("user_id","user_mentions").where("user_mentions is not null")
# extract data
clear_data = mention_user.rdd.map(lambda row:(user_id_to_index[row[0]],row[1])).flatMap(extract_data)
# combine user_id mention_user_id as a key
key_data = clear_data.map(create_key)
# calculate                    
user_mention_data = key_data.reduceByKey(lambda counter1, counter2 : counter1 + counter2)                             .map(lambda row : (row[0][0], row[0][1], row[1]))                             .toDF(["user_id", "mention_user_id", "counter"])

# https://spark.apache.org/docs/latest/ml-collaborative-filtering.html
# train data
als = ALS(userCol="user_id", itemCol="mention_user_id", ratingCol="counter",coldStartStrategy="drop")
model = als.fit(user_mention_data)       
model_recommend = model.recommendForAllUsers(5).collect()

# print result
for rows in model_recommend:
    print(all_user_ids[rows[0]], ":", end=" ")
    for data in rows[1]:
        print(all_mention_users[data[0]], end="   ")
    print()


# In[ ]:




