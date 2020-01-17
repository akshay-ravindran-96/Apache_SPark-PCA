#!/usr/bin/env python
# coding: utf-8

# In[1]:


import findspark


# In[2]:


from pyspark.sql import SparkSession


# In[3]:


spark = SparkSession.builder.master("local").appName("Linear Regression Model").config("spark.executor.memory", "1gb").getOrCreate()


# In[4]:


sc =  spark.sparkContext


# In[24]:


rdd = sc.textFile('cal_housing.data')


# In[14]:


header = sc.textFile('cal_housing.domain')


# In[15]:


header.collect()


# In[16]:


rdd.take(2)


# In[17]:


rdd.take(100)


# In[26]:


rdd = rdd.map(lambda line: line.split(","))


# In[27]:


rdd.take(2)


# In[28]:


rdd.first()


# In[29]:


rdd.top(2)


# In[30]:


from pyspark.sql import Row


# In[31]:


df = rdd.map(lambda line: Row(longitude=line[0], 
                              latitude=line[1], 
                              housingMedianAge=line[2],
                              totalRooms=line[3],
                              totalBedRooms=line[4],
                              population=line[5], 
                              households=line[6],
                              medianIncome=line[7],
                              medianHouseValue=line[8])).toDF()


# In[32]:


df.show()


# In[33]:


df.head(10)


# In[35]:


df.tail(1)


# In[36]:


df.dtypes


# In[37]:


df.printSchema()


# In[38]:


from pyspark.sql.types import *


# In[39]:


df = df.withColumn("longitude", df["longitude"].cast(FloatType()))    .withColumn("latitude", df["latitude"].cast(FloatType()))    .withColumn("housingMedianAge",df["housingMedianAge"].cast(FloatType()))    .withColumn("totalRooms", df["totalRooms"].cast(FloatType())) \ 
   .withColumn("totalBedRooms", df["totalBedRooms"].cast(FloatType())) \ 
   .withColumn("population", df["population"].cast(FloatType())) \ 
   .withColumn("households", df["households"].cast(FloatType())) \ 
   .withColumn("medianIncome", df["medianIncome"].cast(FloatType())) \ 
   .withColumn("medianHouseValue", df["medianHouseValue"].cast(FloatType()))


# In[40]:


df = df.withColumn("longitude", df["longitude"].cast(FloatType())).withColumn("latitude", df["latitude"].cast(FloatType())).withColumn("housingMedianAge",df["housingMedianAge"].cast(FloatType())).withColumn("totalRooms", df["totalRooms"].cast(FloatType())).withColumn("totalBedRooms", df["totalBedRooms"].cast(FloatType())).withColumn("population", df["population"].cast(FloatType())).withColumn("households", df["households"].cast(FloatType())).withColumn("medianIncome", df["medianIncome"].cast(FloatType())).withColumn("medianHouseValue", df["medianHouseValue"].cast(FloatType()))


# In[41]:


def convertColumn(df, names, newType):
  for name in names: 
     df = df.withColumn(name, df[name].cast(newType))
  return df 


# In[42]:


columns = ['households', 'housingMedianAge', 'latitude', 'longitude', 'medianHouseValue', 'medianIncome', 'population', 'totalBedRooms', 'totalRooms']
df = convertColumn(df, columns, FloatType())


# In[43]:


df.printSchema()


# In[44]:


df.select('population','totalBedRooms').show(10)


# In[45]:


df.groupBy("housingMedianAge").count().sort("housingMedianAge",ascending=False).show()


# In[46]:


df.describe().show()


# In[47]:


print(df.describe())


# In[48]:


df.hist(figsize = (10,10))
plt.show()


# In[49]:


df = df.withColumn("medianHouseValue", col("medianHouseValue")/100000)

# Show the first 2 lines of `df`
df.take(2)


# In[50]:


from pyspark.sql.functions import *


# In[51]:


df = df.withColumn("medianHouseValue", col("medianHouseValue")/100000)

# Show the first 2 lines of `df`
df.take(2)


# In[52]:


roomsPerHousehold = df.select(col("totalRooms")/col("households"))

# Divide `population` by `households`
populationPerHousehold = df.select(col("population")/col("households"))

# Divide `totalBedRooms` by `totalRooms`
bedroomsPerRoom = df.select(col("totalBedRooms")/col("totalRooms"))

# Add the new columns to `df`
df = df.withColumn("roomsPerHousehold", col("totalRooms")/col("households"))    .withColumn("populationPerHousehold", col("population")/col("households"))    .withColumn("bedroomsPerRoom", col("totalBedRooms")/col("totalRooms"))
   
# Inspect the result
df.first()


# In[53]:


df = df.select("medianHouseValue", 
              "totalBedRooms", 
              "population", 
              "households", 
              "medianIncome", 
              "roomsPerHousehold", 
              "populationPerHousehold", 
              "bedroomsPerRoom")


# In[54]:


df.take(2)


# In[55]:


from pyspark.ml.linalg import DenseVector

# Define the `input_data` 
input_data = df.rdd.map(lambda x: (x[0], DenseVector(x[1:])))

# Replace `df` with the new DataFrame
df = spark.createDataFrame(input_data, ["label", "features"])


# In[56]:


from pyspark.ml.feature import StandardScaler

# Initialize the `standardScaler`
standardScaler = StandardScaler(inputCol="features", outputCol="features_scaled")

# Fit the DataFrame to the scaler
scaler = standardScaler.fit(df)

# Transform the data in `df` with the scaler
scaled_df = scaler.transform(df)

# Inspect the result
scaled_df.take(2)


# In[57]:


train_data, test_data = scaled_df.randomSplit([.8,.2],seed=1234)


# In[58]:


from pyspark.ml.regression import LinearRegression

# Initialize `lr`
lr = LinearRegression(labelCol="label", maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Fit the data to the model
linearModel = lr.fit(train_data)


# In[59]:


predicted = linearModel.transform(test_data)

# Extract the predictions and the "known" correct labels
predictions = predicted.select("prediction").rdd.map(lambda x: x[0])
labels = predicted.select("label").rdd.map(lambda x: x[0])

# Zip `predictions` and `labels` into a list
predictionAndLabel = predictions.zip(labels).collect()

# Print out first 5 instances of `predictionAndLabel` 
predictionAndLabel[:5]


# In[60]:


linearModel.coefficients


# In[61]:


linearModel.intercept


# In[62]:


linearModel.summary.rootMeanSquaredError


# In[63]:


linearModel.summary.r2


# In[64]:


spark.stop()


# In[ ]:




