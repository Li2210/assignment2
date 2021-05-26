#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import Word2Vec
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer, IndexToString
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import *

# https://stackoverflow.com/questions/57451719/since-spark-2-3-the-queries-from-raw-json-csv-files-are-disallowed-when-the-ref
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

FILE_NAME = "tweets.json"

original_data = spark.read.option("multiline","true").json(FILE_NAME).cache()

selectId = "480875170"

# Q1

# find all replies
user_replyTo = original_data.select("user_id", "replyto_id").where("replyto_id is not null")
# find all retweets
user_retweet = original_data.select("user_id", "retweet_id").where("retweet_id is not null")

# combine two data:(user_id, relatedTweetId)
user_combine = user_retweet.union(user_replyTo)
user_data = user_combine.rdd.map(lambda row:(row[0], str(row[1]))).groupByKey().mapValues(list)

def changeToString(row):
    user_id = row[0]
    data = " ".join(row[1])
    return (user_id, data)
    
    
user_tweet_string = user_data.map(changeToString)

train_data = user_tweet_string.toDF(["user_id", "data"])

# idf train data
# https://spark.apache.org/docs/latest/ml-features.html#tf-idf
tokenizer = Tokenizer(inputCol="data", outputCol="words")
wordsData = tokenizer.transform(train_data)

hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures",numFeatures=131072)
featurizedData = hashingTF.transform(wordsData)
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# drop useless columns
clearData = rescaledData.drop("data").drop("words").drop("rawFeatures")

target_user = clearData.filter("user_id =" + selectId).collect()[0][1]
other_user = clearData.filter("user_id !=" + selectId).rdd.map(lambda row : (row[0], row[1]))

def cosine(a,b):
    return a.dot(b) / (a.norm(2) * b.norm(2))

found_user = other_user.map(lambda row:(row[0], cosine(target_user, row[1])))
idf_result = found_user.sortBy(lambda row:row[1], ascending=False).take(5)



# Word2Vec
text_data = user_tweet_string.toDF(["user_id", "text"])
word_data = text_data.select('user_id', split("text",' ').alias("words"))
# word_data.show(100, truncate=False)

# train data
word2Vec = Word2Vec(vectorSize=300, minCount=1, inputCol="words", outputCol="vector")
model = word2Vec.fit(word_data)
result = model.transform(word_data).drop("words")

target_user_1 = result.filter("user_id =" + selectId).collect()[0][1]
other_user_1 = result.filter("user_id !=" + selectId).rdd.map(lambda row : (row[0], row[1]))
found_user_1 = other_user_1.map(lambda row:(row[0], cosine(target_user_1, row[1])))
word_result = found_user_1.sortBy(lambda row:row[1], ascending=False).take(5)


# get user_id 
user_ids = original_data.select("user_id").rdd.map(lambda row:row[0]).distinct().collect()

# create dictionary for user_id
user_id_mapper = dict()
for user_id in user_ids:
    user_id_mapper[user_id] = len(user_id_mapper)

def extract_id(rows):
    return [(mention_user[0]) for mention_user in rows[0]]

# extract mentioned user id
mention_user_info = original_data.select("user_mentions").where("user_mentions is not null")
mention_user_ids = mention_user_info.rdd.flatMap(extract_id).distinct().collect()
# create dictionary for mentioned user id
mention_user_id_mapper = dict()
for mention_user_id in mention_user_ids:
    mention_user_id_mapper[mention_user_id] = len(mention_user_id_mapper)

# sc.broadcast(user_id_mapper)
# sc.broadcast(mention_user_id_mapper)

def extract_data(rows):
    user_id = rows[0]
    return [(user_id, mention_user_id_mapper[mention_user[0]], 1) for mention_user in rows[1]]

def create_key(rows):
    user_id = rows[0] 
    mention_user = rows[1] 
    counter = rows[2]
    return ((user_id,mention_user), counter)

mention_user = original_data.select("user_id","user_mentions").where("user_mentions is not null")
# extract data
clear_data = mention_user.rdd.map(lambda row:(user_id_mapper[row[0]],row[1])).flatMap(extract_data)
# combine user_id mention_user_id as a key
key_data = clear_data.map(create_key)
# calculate                    
clear_key = key_data.reduceByKey(lambda a,b : a + b).map(lambda row : (row[0][0], row[0][1], row[1]))
train_data = clear_key.toDF(["user_id", "mention_user_id", "counter"])

# https://spark.apache.org/docs/latest/ml-collaborative-filtering.html
# train data
als = ALS(userCol="user_id", itemCol="mention_user_id", ratingCol="counter",coldStartStrategy="drop")
model = als.fit(train_data)
model_recommend = model.recommendForAllUsers(5).collect()

print("Top five users with similar interest based on TF-IDF:")
print(idf_result)

print("Top five users with similar interest based on Word2Vec:")
print(word_result)

print("User recomendation is:")
counter = 0

for rows in model_recommend:
    if(counter < 20):
        print(user_ids[rows[0]], ":", end=" ")
        for data in rows[1]:
            print(mention_user_ids[data[0]], end="   ")
        print()
        counter += 1

sc.stop()


# In[ ]:




