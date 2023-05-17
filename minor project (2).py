#!/usr/bin/env python
# coding: utf-8

# In[8]:


#Importing Librariees
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

get_ipython().system('pip install chart_studio')
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import plot

#for offline plotting
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


# In[9]:


df = pd.read_excel(r"C:\Users\bhagy\Desktop\corizo minor project.xlsx")
df


# In[10]:


df.info()


# In[11]:


print(f'Dataframe contains stock prices between {df.Date.min()} {df.Date.max()}')
print(f'Total days = {(df.Date.max() - df.Date.min()).days} days')


# In[12]:


df.describe()


# In[13]:


df[['Open','High','Low','Close','Adj Close']].plot(kind='box')


# In[14]:


#Setting the layout for our plot
layout = go.Layout(
       title='Stock Prices',
       xaxis=dict(
           title='Date',
           titlefont=dict(
                 family='Courier New, monospace',
                 size=18,
                 color='#7f7f7f'
           )
       ),
       yaxis=dict(
           title='Price',
           titlefont=dict(
                 family='Courier New, monospace',
                 size=18,
                 color='#7f7f7f'
           )
       )
)

df_data = [{'x' : df['Date'], 'y' : df['Close']}]
plot = go.Figure(data=df_data, layout=layout)


# In[16]:


#plot(plot) #Plotting offline
iplot(plot)


# In[17]:


#Building the regression model
from sklearn.model_selection import train_test_split

#for preprocessing
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import StandardScaler

#for model evaluation
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score


# In[18]:


#Split the data into train and test sets
X = np.array(df.index).reshape(-1,1)
Y = df['Close']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)


# In[19]:


#Feature Scaling
scaler = StandardScaler().fit(X_train)


# In[20]:


from sklearn.linear_model import LinearRegression


# In[21]:


#Creating a linear Model
lm = LinearRegression()
lm.fit(X_train, Y_train)


# In[22]:


#plot actual and predicted values for train dataset
trace0 = go.Scatter(
     x = X_train.T[0],
     y = Y_train,
    mode = 'markers',
    name = 'actual'
)
trace1 = go.Scatter(
     x = X_train.T[0],
     y = lm.predict(X_train).T,
     mode = 'lines',
     name = 'Predicted'
)
df_data = [trace0, trace1]
layout.xaxis.title.text = 'Day'
plot2 = go.Figure(data=df_data, layout=layout)


# In[23]:


iplot(plot2)


# In[24]:


#Calculate Scores for Model Evaluation
scores = f'''
{'Metrics'.ljust(10)}{'Train'.center(20)}{'Test'.center(20)}
{'r2_score'.ljust(10)}{r2_score(Y_train, lm.predict(X_train))}\t{r2_score(Y_test, lm.predict(X_test))}
{'MSE'.ljust(10)}{mse(Y_train, lm.predict(X_train))}\t{mse(Y_test, lm.predict(X_test))}
'''
print(scores)


# In[ ]:




