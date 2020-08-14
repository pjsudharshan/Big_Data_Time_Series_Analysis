#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import geojsonio
from datetime import timedelta, date, datetime
import time
import plotly.express as px
import glob as gb
import json
import pytz

import pmdarima as pm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[2]:


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 15

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE+2)  # fontsize of the figure title


# # Data Exploration

# In[3]:


#Sample of format of each line as read from the input file
col_list = ['gridID', 'timeInterval', 'countryCode', 'smsIn', 'smsOut', 'callIn', 'callOut', 'internet']
read_data = pd.read_csv('dataverse_files/sms-call-internet-mi-2013-11-01.txt' ,sep='\t',header=None, names=col_list,                         parse_dates=True)


# In[4]:


read_data.head(10)


# In[5]:


start_time = time.time()

#Initialize an empty dataframe to append daily and hourly resampled data
dailyGridActivity = pd.DataFrame()
hourlyGridActivity = pd.DataFrame()

#Create a list of 62 data file names placed under directory  "dataverse_files" with extension .txt
filenames = gb.glob("dataverse_files/*.txt")

#Set the column names for the data read
col_list = ['gridID', 'timeInterval', 'countryCode', 'smsIn', 'smsOut', 'callIn', 'callOut', 'internet']

#Read each tab separated file into Pandas DataFrame 
for file in filenames:
    read_data = pd.read_csv(file, sep='\t',header=None, names=col_list, parse_dates=True)
    
    #Convert timeInterval column which has Epoch timestamps to UTC and then convert to Milan's local timezone
    read_data['startTime'] = pd.to_datetime(read_data.timeInterval, unit='ms', utc=True).dt.tz_convert('CET').dt.tz_localize(None)
    
    #Drop timeInterval & countryCode columns
    read_data.drop(columns=['timeInterval','countryCode'], inplace=True)
    
    #Groupby gridID and startTime, startTime which is 10 min apart is resampled to daily aggregation 
    read_data_daily = read_data.groupby(['gridID', pd.Grouper(key='startTime', freq='D')]).sum()
    dailyGridActivity = pd.concat([dailyGridActivity,read_data_daily]).groupby(['gridID', 'startTime']).sum()
    
    #Groupby gridID and startTime, startTime which is 10 min apart is resampled to hourly aggregation 
    read_data_hourly = read_data.groupby(['gridID', pd.Grouper(key='startTime', freq='H')]).sum()
    hourlyGridActivity = pd.concat([hourlyGridActivity,read_data_hourly]).groupby(['gridID', 'startTime']).sum()
    
#Get Grid wise total volume of the activities over the 2months
totalGridActivity = dailyGridActivity.groupby('gridID').sum()
    
print('%3.2f s' %(time.time() - start_time))


# In[6]:


hourlyGridActivity.head(5000)


# In[7]:


dailyGridActivity.head(500)


# In[8]:


totalGridActivity.head(100)


# In[9]:


#Create additional columns hours:hour of the day, weekdayFlag: weekend or weekday information
dailyGridActivity['weekdayFlag'] = dailyGridActivity.index.get_level_values(1)
dailyGridActivity['weekdayFlag'] = dailyGridActivity['weekdayFlag'].dt.weekday

hourlyGridActivity['weekdayFlag'] = hourlyGridActivity.index.get_level_values(1)
hourlyGridActivity['weekdayFlag'] = hourlyGridActivity['weekdayFlag'].dt.weekday

hourlyGridActivity['hours'] = hourlyGridActivity.index.get_level_values(1)
hourlyGridActivity['hours'] = hourlyGridActivity['hours'].dt.hour

dailyGridActivity['sms'] = dailyGridActivity['smsIn'] + dailyGridActivity['smsOut']
dailyGridActivity['call'] = dailyGridActivity['callIn'] + dailyGridActivity['callOut']

hourlyGridActivity['sms'] = hourlyGridActivity['smsIn'] + hourlyGridActivity['smsOut']
hourlyGridActivity['call'] = hourlyGridActivity['callIn'] + hourlyGridActivity['callOut']

totalGridActivity['sms'] = totalGridActivity['smsIn']+totalGridActivity['smsOut']
totalGridActivity['call'] = totalGridActivity['callIn']+totalGridActivity['callOut']


# In[10]:


dailyGridActivity.head()


# In[11]:


hourlyGridActivity.head()


# In[12]:


totalGridActivity.head()


# In[13]:


hist = totalGridActivity['sms'].hist(bins=1000)


# In[14]:


hist = totalGridActivity['sms'].plot.kde()


# ## Test the plot with matplotlib

# In[15]:


plt.figure(figsize=(10,5))
plt.hist(totalGridActivity['sms'], bins='auto', density=True)
plt.xlabel('SMS activity')
plt.ylabel('PDF')
plt.title(r'Total SMS activity (sms_in+sms_out)')
plt.tight_layout()
plt.savefig('sav_images/totalsms.svg',transparent=True)
plt.show()


# ## Seaborn provides combined kde and histogram plot

# In[16]:


plt.figure(figsize=(10,5))
ax = sns.distplot(totalGridActivity['smsIn'], bins='auto', kde=True,                   norm_hist=True, kde_kws={'color':'k','label':'KDE'},                   hist_kws={'color':'blue','alpha':0.5,'label':'Actual data'})
plt.xlabel('SMS activity')
plt.ylabel('PDF')
plt.title(r'Incoming SMS activity')
plt.tight_layout()
plt.savefig('sav_images/sms_in_kde.svg',transparent=True)
plt.show()


# In[17]:


plt.figure(figsize=(10,5))
ax = sns.distplot(totalGridActivity['smsOut'], bins='auto', kde=True,                   norm_hist=True, kde_kws={'color':'k','label':'KDE'},                   hist_kws={'color':'blue','alpha':0.5,'label':'Actual data'})
plt.xlabel('SMS activity')
plt.ylabel('PDF')
plt.title(r'Outgoing SMS activity')
plt.tight_layout()
plt.savefig('sav_images/sms_out_kde.svg',transparent=True)
plt.show()


# In[18]:


plt.figure(figsize=(10,5))
ax = sns.distplot(totalGridActivity['sms'], bins='auto', kde=True,                   norm_hist=True, kde_kws={'color':'k','label':'KDE'},                   hist_kws={'color':'blue','alpha':0.5,'label':'Actual data'})
plt.xlabel('SMS activity')
plt.ylabel('PDF')
plt.title(r'Total SMS activity (sms_in+sms_out)')
plt.tight_layout()
plt.savefig('sav_images/sms_total_kde.svg',transparent=True)
plt.show()


# In[19]:


plt.figure(figsize=(10,5))
ax = sns.distplot(totalGridActivity['call'], bins='auto', kde=True,                   norm_hist=True, kde_kws={'color':'k','label':'KDE'},                   hist_kws={'color':'blue','alpha':0.5,'label':'Actual data'})
plt.xlabel('Call activity')
plt.ylabel('PDF')
plt.title(r'Total Call activity (call_in+call_out)')
plt.tight_layout()
plt.savefig('sav_images/call_total_kde.svg',transparent=True)
plt.show()


# In[20]:


plt.figure(figsize=(10,5))
ax = sns.distplot(totalGridActivity['internet'], bins='auto', kde=True,                   norm_hist=True, kde_kws={'color':'k','label':'KDE'},                   hist_kws={'color':'blue','alpha':0.5,'label':'Actual data'})
plt.xlabel('Internet Traffic')
plt.ylabel('PDF')
plt.title(r'Internet Traffic activity')
plt.tight_layout()
plt.savefig('sav_images/internet_kde.svg',transparent=True)
plt.show()


# ## Top 10 grids

# In[21]:


smsInGridActivity = totalGridActivity[['smsIn']].sort_values(by = 'smsIn',ascending=False)
smsOutGridActivity = totalGridActivity[['smsOut']].sort_values(by = 'smsOut',ascending=False)
callInGridActivity = totalGridActivity[['callIn']].sort_values(by = 'callIn',ascending=False)
callOutGridActivity = totalGridActivity[['callOut']].sort_values(by = 'callOut',ascending=False)

smsGridActivity = totalGridActivity[['sms']].sort_values(by = 'sms',ascending=False)
callGridActivity = totalGridActivity[['call']].sort_values(by = 'call',ascending=False)
internetGridActivity = totalGridActivity[['internet']].sort_values(by = 'internet',ascending=False)

#Fetch the top 10 grids from the sorted grids and display them
top10 = pd.DataFrame()
top10['smsIn'] = smsInGridActivity[:10].index.values
top10['smsOut'] = smsOutGridActivity[:10].index.values
top10['callIn'] = callInGridActivity[:10].index.values
top10['callOut'] = callOutGridActivity[:10].index.values
top10['sms'] = smsGridActivity[:10].index.values
top10['call'] = callGridActivity[:10].index.values
top10['internet'] = internetGridActivity[:10].index.values


# In[22]:


top10


# * For the internet traffic, the area with the highest total traffic during the two-month period: 5161
# * For the smsin, smsout, total sms activity, the area with the highest total traffic during the two-month period: 5059
# * For the callin, callout, total call activity, the area with the highest total traffic during the two-month period: 5059

# In[23]:


#Plot top 10 grids and its volume for SMS activity
ax = smsGridActivity[:10].plot(kind='barh', figsize=(10,5))
ax.invert_yaxis()
ax.set_xlabel("Total SMS Activity")
ax.set_ylabel("Top 10 Grids")
fig = ax.get_figure()
fig.savefig('sav_images/sms_top10.svg',transparent=True)


# In[24]:


#Token for Mapbox API

token = r'pk.eyJ1IjoicGpzdWRoYXJzaGFuIiwiYSI6ImNrY295YTZ4aTBwczEydHF5aG1qM2M4NTMifQ.8MXIaUGorClC0Z4SG9NduQ'


# In[25]:


#Read GeoJSON file
milan = gpd.read_file('milano-grid.geojson')


# In[ ]:


#Visualize the region of interest on map

fig = px.choropleth_mapbox(smsGridActivity[:10].reset_index(),
                           geojson=milan,
                           locations='gridID',
                           color='sms',
                           zoom=12.3, center = {"lat": 45.4646, "lon": 9.1885},
                           opacity=0.7)
fig.update_layout(mapbox_style="dark", mapbox_accesstoken=token)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.write_image('sav_images/sms_top10_visual.svg')
fig.show()


# In[26]:


#Plot top 10 grids and its volume for Call activity
ax = callGridActivity[:10].plot(kind='barh', figsize=(10,5))
ax.invert_yaxis()
ax.set_xlabel("Total Call Activity")
ax.set_ylabel("Top 10 Grids")
fig = ax.get_figure()
fig.savefig('sav_images/call_top10.svg',transparent=True)


# In[ ]:


#Visualize the region of interest on map

fig = px.choropleth_mapbox(callGridActivity[:10].reset_index(),
                           geojson=milan,
                           locations='gridID',
                           color='call',
                           zoom=12.3, center = {"lat": 45.4646, "lon": 9.1885},
                           opacity=0.7)
fig.update_layout(mapbox_style="dark", mapbox_accesstoken=token)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.write_image('sav_images/call_top10_visual.svg')
fig.show()


# In[27]:


#Plot top 10 grids and its volume for Internet activity
ax = internetGridActivity[:10].plot(kind='barh', figsize=(10,5))
ax.invert_yaxis()
ax.set_xlabel("Internet Activity")
ax.set_ylabel("Top 10 Grids")
fig = ax.get_figure()
fig.savefig('sav_images/internet_top10.svg',transparent=True)


# In[ ]:


#Visualize the region of interest on map

fig = px.choropleth_mapbox(internetGridActivity[:10].reset_index(),
                           geojson=milan,
                           locations='gridID',
                           color='internet',
                           zoom=12.3, center = {"lat": 45.4646, "lon": 9.1885},
                           opacity=0.7)
fig.update_layout(mapbox_style="dark", mapbox_accesstoken=token)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.write_image('sav_images/internet_top10_visual.svg')
fig.show()


# ## Visualization of SMS, Call and Internet activities

# ### Doesn't display well

# In[28]:


#get unique values across all the columns in dataframe top10
topgrids = pd.unique(top10.values.ravel())


# In[ ]:


#New browser opens with the top 10 grids
topGridgeojson = milan.loc[milan['cellId'].isin(topgrids)]
topGridgeojson = topGridgeojson.to_json()

#Uncomment below line to display on browser
# _ = geojsonio.display(topGridgeojson)


# ### Change to Plotly library - which can provide better visualiztion

# In[ ]:


#Internet Activity Visualization

internetGridActivity_rest = internetGridActivity.reset_index()
internetGridActivity_min = internetGridActivity_rest['internet'].min()
internetGridActivity_max = internetGridActivity_rest['internet'].max()

fig = px.choropleth_mapbox(internetGridActivity_rest, geojson=milan, locations='gridID', color='internet',
                           color_continuous_scale="thermal",
                           range_color=(internetGridActivity_min, internetGridActivity_max),
                           zoom=10.5, center = {"lat": 45.4646, "lon": 9.1885},
                           opacity=0.7,
                           labels={'internet':'Internet Traffic'}
                          )
fig.update_layout(mapbox_style="dark", mapbox_accesstoken=token)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.write_image('sav_images/internet_visual.svg')
fig.show()


# In[ ]:


#Call Activity Visualization

callGridActivity_rest = callGridActivity.reset_index()
callGridActivity_min = callGridActivity_rest['call'].min()
callGridActivity_max = callGridActivity_rest['call'].max()

fig = px.choropleth_mapbox(callGridActivity_rest, geojson=milan, locations='gridID', color='call',
                           color_continuous_scale="thermal",
                           range_color=(callGridActivity_min, callGridActivity_max),
                           zoom=10.5, center = {"lat": 45.4646, "lon": 9.1885},
                           opacity=0.7,
                           labels={'call':'Call Activity'}
                          )
fig.update_layout(mapbox_style="dark", mapbox_accesstoken=token)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.write_image('sav_images/call_visual.svg')
fig.show()


# In[ ]:


#SMS Activity Visualization

smsGridActivity_rest = smsGridActivity.reset_index()
smsGridActivity_min = smsGridActivity_rest['sms'].min()
smsGridActivity_max = smsGridActivity_rest['sms'].max()

fig = px.choropleth_mapbox(smsGridActivity_rest, geojson=milan, locations='gridID', color='sms',
                           color_continuous_scale="thermal",
                           range_color=(smsGridActivity_min, smsGridActivity_max),
                           zoom=10.5, center = {"lat": 45.4646, "lon": 9.1885},
                           opacity=0.7,
                           labels={'sms':'SMS Activity'}
                          )
fig.update_layout(mapbox_style="dark", mapbox_accesstoken=token)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.write_image('sav_images/sms_visual.svg')
fig.show()


# ## Three regions of interest - Square id 5161 (Top internet activity), 4159 and 4556

# ### Daily Analysis

# In[29]:


#Three regions of interest
sqid_intr = ['5161','4159','4556']
daily5161 = dailyGridActivity.loc[5161][['sms','call','internet']]
daily4159 = dailyGridActivity.loc[4159][['sms','call','internet']]
daily4556 = dailyGridActivity.loc[4556][['sms','call','internet']]


# In[30]:


#We are interested only in first two weeks
days_intr = 14
start_date = date(2013, 11, 1)
end_date = start_date + timedelta(days_intr-1)

daily5161_intr = daily5161.loc[start_date:end_date]
daily4159_intr = daily4159.loc[start_date:end_date]
daily4556_intr = daily4556.loc[start_date:end_date]

sqid_daily_intr = [daily5161_intr, daily4159_intr, daily4556_intr]


# In[31]:


plt.figure(figsize=(15,5))
for i,sqid in enumerate(sqid_daily_intr):
    plt.plot(sqid.index, sqid.sms, label=sqid_intr[i])
plt.legend(loc='best')
plt.xlabel('Date')
plt.ylabel('Total SMS activity')
plt.title('Comparison of SMS activity')
plt.tight_layout()
plt.savefig('sav_images/sms_timeseries.svg',transparent=True)
plt.show()


# In[32]:


plt.figure(figsize=(15,5))
for i,sqid in enumerate(sqid_daily_intr):
    plt.plot(sqid.index, sqid.call, label=sqid_intr[i])
plt.legend(loc='best')
plt.xlabel('Date')
plt.ylabel('Total Call activity')
plt.title('Comparison of Call activity')
plt.tight_layout()
plt.savefig('sav_images/call_timeseries.svg',transparent=True)
plt.show()


# In[33]:


plt.figure(figsize=(15,5))
for i,sqid in enumerate(sqid_daily_intr):
    plt.plot(sqid.index, sqid.internet, label=sqid_intr[i])
plt.legend(loc='best')
plt.xlabel('Date')
plt.ylabel('Total Internet Traffic')
plt.title('Comparison of Internet Traffic')
plt.tight_layout()
plt.savefig('sav_images/internet_timeseries.svg',transparent=True)
plt.show()


# In[ ]:


#Visualize the region of interest on map

fig = px.choropleth_mapbox(pd.DataFrame(sqid_intr, columns=['sqid']),
                           geojson=milan,
                           locations='sqid',
                           color='sqid',
                           zoom=12, center = {"lat": 45.4646, "lon": 9.1885},
                           opacity=0.7)
fig.update_layout(mapbox_style="dark", mapbox_accesstoken=token)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.write_image('sav_images/sqid_intr_visual.svg')
fig.show()


# ### Hourly Analysis

# In[34]:


pd.options.mode.chained_assignment = None

#Get hourly data for the three grids 
hourly5161 = hourlyGridActivity.loc[5161]
hourly4159 = hourlyGridActivity.loc[4159]
hourly4556 = hourlyGridActivity.loc[4556]

#We are interested only in first two weeks
hourly5161_intr = hourly5161.loc[start_date:end_date]
hourly4159_intr = hourly4159.loc[start_date:end_date]
hourly4556_intr = hourly4556.loc[start_date:end_date]

#Add column to identify Day of the week for each observation
hourly5161_intr['dayOfWeek'] = hourly5161_intr.index.day_name()
hourly4159_intr['dayOfWeek'] = hourly4159_intr.index.day_name()
hourly4556_intr['dayOfWeek'] = hourly4556_intr.index.day_name()

#Use pivot_table() reshape the dataframe with Mean values of the acitivities for Day of the week as columns and hours in a day as index
hourly5161_sms_intr = hourly5161_intr.pivot_table(index=hourly5161_intr.index.hour,
                                                            columns='dayOfWeek', values='sms', aggfunc='mean')
hourly5161_call_intr = hourly5161_intr.pivot_table(index=hourly5161_intr.index.hour,
                                                            columns='dayOfWeek', values='call', aggfunc='mean')
hourly5161_internet_intr = hourly5161_intr.pivot_table(index=hourly5161_intr.index.hour,
                                                            columns='dayOfWeek', values='internet', aggfunc='mean')

hourly4159_sms_intr = hourly4159_intr.pivot_table(index=hourly4159_intr.index.hour,
                                                            columns='dayOfWeek', values='sms', aggfunc='mean')
hourly4159_call_intr = hourly4159_intr.pivot_table(index=hourly4159_intr.index.hour,
                                                            columns='dayOfWeek', values='call', aggfunc='mean')
hourly4159_internet_intr = hourly4159_intr.pivot_table(index=hourly4159_intr.index.hour,
                                                            columns='dayOfWeek', values='internet', aggfunc='mean')

hourly4556_sms_intr = hourly4556_intr.pivot_table(index=hourly4556_intr.index.hour,
                                                            columns='dayOfWeek', values='sms', aggfunc='mean')
hourly4556_call_intr = hourly4556_intr.pivot_table(index=hourly4556_intr.index.hour,
                                                            columns='dayOfWeek', values='call', aggfunc='mean')
hourly4556_internet_intr = hourly4556_intr.pivot_table(index=hourly4556_intr.index.hour,
                                                            columns='dayOfWeek', values='internet', aggfunc='mean')


# In[35]:


hourly5161_sms_intr = hourly5161_sms_intr[['Friday','Saturday','Sunday','Monday','Tuesday','Wednesday','Thursday']]
hourly4159_sms_intr = hourly4159_sms_intr[['Friday','Saturday','Sunday','Monday','Tuesday','Wednesday','Thursday']]
hourly4556_sms_intr = hourly4556_sms_intr[['Friday','Saturday','Sunday','Monday','Tuesday','Wednesday','Thursday']]


# In[36]:


#From the reshaped dataframe plot heatmap for SMS activity of the three grids

fig, ax = plt.subplots(1,3,figsize=(15,5))
ax[0].set_title("SMS Activity [5161]")
ax[1].set_title("SMS Activity [4159]")
ax[2].set_title("SMS Activity [4556]")

sns.heatmap(hourly5161_sms_intr, ax=ax[0])
ax[0].set_ylabel("Hours in a Day")
ax[0].set_xlabel("Days of Week")
sns.heatmap(hourly4159_sms_intr, ax=ax[1])
ax[1].set_ylabel("Hours in a Day")
ax[1].set_xlabel("Days of Week")
sns.heatmap(hourly4556_sms_intr, ax=ax[2])
ax[2].set_ylabel("Hours in a Day")
ax[2].set_xlabel("Days of Week")
fig.tight_layout()
plt.savefig('sav_images/sms_hourly.svg',transparent=True)
plt.show()


# In[37]:


hourly5161_call_intr = hourly5161_call_intr[['Friday','Saturday','Sunday','Monday','Tuesday','Wednesday','Thursday']]
hourly4159_call_intr = hourly4159_call_intr[['Friday','Saturday','Sunday','Monday','Tuesday','Wednesday','Thursday']]
hourly4556_call_intr = hourly4556_call_intr[['Friday','Saturday','Sunday','Monday','Tuesday','Wednesday','Thursday']]


# In[38]:


#From the reshaped dataframe plot heatmap for Call activity of the three grids

fig, ax = plt.subplots(1,3,figsize=(15,5))
ax[0].set_title("Call Activity [5161]")
ax[1].set_title("Call Activity [4159]")
ax[2].set_title("Call Activity [4556]")

sns.heatmap(hourly5161_call_intr, ax=ax[0])
ax[0].set_ylabel("Hours in a Day")
ax[0].set_xlabel("Days of Week")
sns.heatmap(hourly4159_call_intr, ax=ax[1])
ax[1].set_ylabel("Hours in a Day")
ax[1].set_xlabel("Days of Week")
sns.heatmap(hourly4556_call_intr, ax=ax[2])
ax[2].set_ylabel("Hours in a Day")
ax[2].set_xlabel("Days of Week")
fig.tight_layout()
plt.savefig('sav_images/call_hourly.svg',transparent=True)
plt.show()


# In[39]:


hourly5161_internet_intr = hourly5161_internet_intr[['Friday','Saturday','Sunday','Monday','Tuesday','Wednesday','Thursday']]
hourly4159_internet_intr = hourly4159_internet_intr[['Friday','Saturday','Sunday','Monday','Tuesday','Wednesday','Thursday']]
hourly4556_internet_intr = hourly4556_internet_intr[['Friday','Saturday','Sunday','Monday','Tuesday','Wednesday','Thursday']]


# In[40]:


#From the reshaped dataframe plot heatmap for internet traffic of the three grids

fig, ax = plt.subplots(1,3,figsize=(15,5))
ax[0].set_title("Internet Traffic [5161]")
ax[1].set_title("Internet Traffic [4159]")
ax[2].set_title("Internet Traffic [4556]")

sns.heatmap(hourly5161_internet_intr, ax=ax[0])
ax[0].set_ylabel("Hours in a Day")
ax[0].set_xlabel("Days of Week")
sns.heatmap(hourly4159_internet_intr, ax=ax[1])
ax[1].set_ylabel("Hours in a Day")
ax[1].set_xlabel("Days of Week")
sns.heatmap(hourly4556_internet_intr, ax=ax[2])
ax[2].set_ylabel("Hours in a Day")
ax[2].set_xlabel("Days of Week")
fig.tight_layout()
plt.savefig('sav_images/internet_hourly.svg',transparent=True)
plt.show()


# In[41]:


#Plot box plot of SMS activity for each day of the week

fig, ax = plt.subplots(1,3,figsize=(19,4))
ax[0].set_title("SMS Activity [5161]")
ax[1].set_title("SMS Activity [4159]")
ax[2].set_title("SMS Activity [4556]")

sns.boxplot(hourly5161_intr['dayOfWeek'], hourly5161_intr['sms'], ax=ax[0])
sns.boxplot(hourly4159_intr['dayOfWeek'], hourly4159_intr['sms'], ax=ax[1])
sns.boxplot(hourly4556_intr['dayOfWeek'], hourly4556_intr['sms'], ax=ax[2])
fig.tight_layout()
plt.savefig('sav_images/sms_daily.svg',transparent=True)
plt.show()


# In[42]:


#Plot box plot of SMS activity for each day of the week

fig, ax = plt.subplots(1,3,figsize=(19,4))
ax[0].set_title("Call Activity [5161]")
ax[1].set_title("Call Activity [4159]")
ax[2].set_title("Call Activity [4556]")

sns.boxplot(hourly5161_intr['dayOfWeek'], hourly5161_intr['call'], ax=ax[0])
sns.boxplot(hourly4159_intr['dayOfWeek'], hourly4159_intr['call'], ax=ax[1])
sns.boxplot(hourly4556_intr['dayOfWeek'], hourly4556_intr['call'], ax=ax[2])
fig.tight_layout()
plt.savefig('sav_images/call_daily.svg',transparent=True)
plt.show()


# In[43]:


#Plot box plot of SMS activity for each day of the week

fig, ax = plt.subplots(1,3,figsize=(19,4))
ax[0].set_title("Internet Traffic [5161]")
ax[1].set_title("Internet Traffic [4159]")
ax[2].set_title("Internet Traffic [4556]")

sns.boxplot(hourly5161_intr['dayOfWeek'], hourly5161_intr['internet'], ax=ax[0])
sns.boxplot(hourly4159_intr['dayOfWeek'], hourly4159_intr['internet'], ax=ax[1])
sns.boxplot(hourly4556_intr['dayOfWeek'], hourly4556_intr['internet'], ax=ax[2])
fig.tight_layout()
plt.savefig('sav_images/internet_daily.svg',transparent=True)
plt.show()

