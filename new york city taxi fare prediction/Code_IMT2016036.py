import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))

train =  pd.read_csv('../input/train.csv', nrows = 20000000)

train.head()

train=train.dropna()

train.shape

train.columns

train = train.drop(((train[train['pickup_latitude']<-90])|(train[train['pickup_latitude']>90])).index, axis=0)
train = train.drop(((train[train['pickup_longitude']<-180])|(train[train['pickup_longitude']>180])).index, axis=0)
train = train.drop(((train[train['dropoff_latitude']<-90])|(train[train['dropoff_latitude']>90])).index, axis=0)
train = train.drop(((train[train['dropoff_longitude']<-180])|(train[train['dropoff_longitude']>180])).index, axis=0)

train = train.drop(train[train['fare_amount']<0].index, axis=0)

train['pickup_datetime'] = train['pickup_datetime'].str.replace(" UTC", "")
train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
train['hour_of_day'] = train.pickup_datetime.dt.hour
train['week'] = train.pickup_datetime.dt.week
train['month'] = train.pickup_datetime.dt.month
train["year"] = train.pickup_datetime.dt.year
train['day_of_year'] = train.pickup_datetime.dt.dayofyear
train['week_of_year'] = train.pickup_datetime.dt.weekofyear
train["weekday"] = train.pickup_datetime.dt.weekday
train["quarter"] = train.pickup_datetime.dt.quarter
train["day_of_month"] = train.pickup_datetime.dt.day

train=train.drop_duplicates(subset=[ 'fare_amount', 'pickup_datetime', 'pickup_longitude',
       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
       'passenger_count'], keep='first', inplace=False)

def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))  

train['trip_distance'] = distance(train.pickup_latitude, train.pickup_longitude ,train.dropoff_latitude, train.dropoff_longitude)

train.head()

f, ax = plt.subplots()
figure = sns.countplot(x = 'passenger_count', data=train, order=train['passenger_count'].unique())
ax = ax.set(ylabel="Count", xlabel="passenger_count")
figure.grid(False)
plt.title('passenger_count')
plt.show()

sns.jointplot(x='passenger_count', y='fare_amount', data=train)

sns.jointplot(x='trip_distance', y='fare_amount', data=train)

sns.jointplot(x='hour_of_day', y='fare_amount', data=train)

sns.jointplot(x='week', y='fare_amount', data=train)

sns.jointplot(x='month', y='fare_amount', data=train)

sns.jointplot(x='year', y='fare_amount', data=train)

sns.jointplot(x='day_of_year', y='fare_amount', data=train)

sns.jointplot(x='week_of_year', y='fare_amount', data=train)

sns.jointplot(x='weekday', y='fare_amount', data=train)

sns.jointplot(x='quarter', y='fare_amount', data=train)

sns.jointplot(x='day_of_month', y='fare_amount', data=train)

train.head()