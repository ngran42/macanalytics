# Databricks notebook source
import logging

# disable informational messages from fbprophet
logging.getLogger('py4j').setLevel(logging.ERROR)

# COMMAND ----------

import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from fbprophet import Prophet 
%matplotlib inline

# COMMAND ----------

df = spark.read.format ("csv").load("dbfs:/mnt/customer-analytics/demand_sales1_prophet.csv", header=True)

# COMMAND ----------

from pyspark.sql.types import IntegerType

df = df.withColumn("y", df["y"].cast(IntegerType()))

# COMMAND ----------

from pyspark.sql.types import DateType

df = df.withColumn("ds", df["ds"].cast(DateType()))

# COMMAND ----------

series = df.toPandas()

# COMMAND ----------

pd.set_option('display.float_format', lambda x: '%.2f' % x)

# COMMAND ----------

len(series)

# COMMAND ----------

series.head()

# COMMAND ----------

series.tail()

# COMMAND ----------

from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
import itertools

# COMMAND ----------

series['ds'] = pd.to_datetime(series['ds'])

# COMMAND ----------

series.info

# COMMAND ----------

series.describe()

# COMMAND ----------

train['ds'] = pd.to_datetime(train['ds'], format='%Y-%m-%d') 

# COMMAND ----------

# Filter data between two dates 
filtered_train1 = train.loc[(train['ds'] >= '2015-07-05') 
                     & (train['ds'] < '2017-07-05')] 

# COMMAND ----------

filtered_test1 = train.loc[(train['ds'] >= '2017-07-05') 
                     & (train['ds'] < '2018-01-02')] 

# COMMAND ----------

#Plotting data
train1.y.plot(figsize=(15,8), title= 'Daily Sales', fontsize=14)
test1.y.plot(figsize=(15,8), title= 'Daily Sales', fontsize=14)
plt.show()

# COMMAND ----------

help(Prophet)

# COMMAND ----------

help(Prophet.fit)

# COMMAND ----------

holidays = pd.DataFrame({
  'holiday': 'holiday_season',
  'ds': pd.to_datetime(['2015-11-26', '2015-11-27', '2015-11-30', '2015-12-24', '2015-12-25','2015-12-26',
                        '2016-11-24', '2016-11-25', '2016-11-28', '2016-12-24', '2016-12-25','2016-12-26',
                        '2017-11-23', '2017-11-24', '2017-11-27', '2017-12-24', '2017-12-25','2017-12-26',
                        '2018-11-22', '2018-11-23', '2018-11-26', '2018-12-24', '2018-12-25','2018-12-26',
                        '2019-11-28', '2019-11-29', '2019-12-2',  '2019-12-24', '2019-12-25','2019-12-26',
                        '2020-11-26', '2020-11-27', '2020-11-30', '2020-12-24', '2020-12-25','2020-12-26']),
  'lower_window': -1,
  'upper_window': 0,
})

# COMMAND ----------

m = Prophet(holidays=holidays, 
            daily_seasonality = True,
           interval_width=0.95)

m.add_country_holidays(country_name='US')

# COMMAND ----------

m.train_holiday_names

# COMMAND ----------

#m= Prophet()
m.fit(series)

# COMMAND ----------

#forecast = model.predict(test1) 
#forecast.tail()

# COMMAND ----------

future = m.make_future_dataframe(periods=180)

# COMMAND ----------

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# COMMAND ----------

fig1 = m.plot(forecast)

# COMMAND ----------

fig2 = m.plot_components(forecast)

# COMMAND ----------

from fbprophet.plot import plot_plotly, plot_components_plotly

plot_plotly(m, forecast)

# COMMAND ----------

plot_components_plotly(m, forecast)

# COMMAND ----------

#display(forecast)

# COMMAND ----------

display(spark.createDataFrame(forecast))

# COMMAND ----------

#export forecast dataframe to csv
spark.createDataFrame(forecast).coalesce(1).write.format('csv').save('dbfs:/mnt/customer-analytics/NGrannum/forecast.csv')

# COMMAND ----------

df_cv = cross_validation(m, initial='400 days', period='90 days', horizon = '180 days')

# COMMAND ----------

df_p = performance_metrics(df_cv)


# COMMAND ----------

df_p

# COMMAND ----------

fig3 = plot_cross_validation_metric(df_cv, metric='mape')

# COMMAND ----------

def create_param_combinations(**param_dict):
    param_iter = itertools.product(*param_dict.values())
    params =[]
    for param in param_iter:
        params.append(param) 
    params_df = pd.DataFrame(params, columns=list(param_dict.keys()))
    return params_df

def single_cv_run(history_df, metrics, param_dict):
    m = Prophet(**param_dict)
    m.add_country_holidays(country_name='US')
    m.fit(history_df)
    df_cv = cross_validation(m, initial='400 days', period='90 days', horizon = '180 days')
    df_p = performance_metrics(df_cv).mean().to_frame().T
    df_p['params'] = str(param_dict)
    df_p = df_p.loc[:, metrics]
    return df_p

param_grid = {  
                'changepoint_prior_scale': [0.005, 0.05, 0.5, 5],
                'changepoint_range': [0.8, 0.9],
                'seasonality_prior_scale':[0.1, 1, 10.0],
                'holidays_prior_scale':[0.1, 1, 10.0],
                'seasonality_mode': ['multiplicative', 'additive'],
                'growth': ['linear', 'logistic'],
                'yearly_seasonality': [5, 10, 20]
              }

metrics = ['horizon', 'rmse', 'mape', 'params'] 

results = []


params_df = create_param_combinations(**param_grid)
for param in params_df.values:
    param_dict = dict(zip(params_df.keys(), param))
    cv_df = single_cv_run(df,  metrics, param_dict)
    results.append(cv_df)
results_df = pd.concat(results).reset_index(drop=True)
best_param = results_df.loc[results_df['mape'] == min(results_df['mape']), ['params']]
print(f'\n The best param combination is {best_param.values[0][0]}')
results_df

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

m= Prophet(
    changepoint_prior_scale=5,
    changepoint_range=0.8,
    seasonality_prior_scale=10,
    holidays_prior_scale= 1,
    seasonality_mode='additive',
    #growth='logistic', 
    growth='linear',
    yearly_seasonality= 10
)
m.add_country_holidays(country_name='US')
m.fit(df)
future = m.make_future_dataframe(periods=180) 
#future['cap'] = 60000
#future['floor'] = 0
forecast = m.predict(future)
fig21 = m.plot(forecast)
fig22 = m.plot_components(forecast)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

model = Prophet()
model.fit()

# COMMAND ----------

recalc_dates = features.resample('Q',level='date').mean().index.values[:-1]
#print('recalc_dates:')
#print(recalc_dates)
#print()

models = pd.Series(index=recalc_dates)
for date in recalc_dates:
    X_train = features.xs(slice(None,date),level='date',drop_level=False)
    y_train = outcome.xs(slice(None,date),level='date',drop_level=False)
    model = LinearRegression()
    model.fit(X_train,y_train)
    models.loc[date] = model
    
    
    print("Training on the first {} records, through {}"\
          .format(len(y_train),y_train.index.get_level_values('date').max()))
    #print("Coefficients: {}".format((model.coef_)))
    #print()

# COMMAND ----------

def extract_coefs(models):
    coefs = pd.DataFrame()
    for i,model in enumerate(models):
        model_coefs = pd.Series(model.coef_,index=['f01','f02','f03','f04']) #extract coefficients for model
        model_coefs.name = models.index[i] # name it with the recalc date
        coefs = pd.concat([coefs,model_coefs],axis=1)
    return coefs.T
extract_coefs(models).plot(title='Coefficients for Expanding Window Model')

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

