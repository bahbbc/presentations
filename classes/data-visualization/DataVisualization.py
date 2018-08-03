
# coding: utf-8

# # Visualização de dados para tomada de decisão

# ![](https://media.giphy.com/media/zw69pUViBZCZW/giphy.gif)

# In[1]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().magic('matplotlib inline')


# In[2]:


df = pd.read_csv('kaggle-survey-2017/multipleChoiceResponses.csv', encoding="ISO-8859-1")


# In[3]:


exchange = pd.read_csv('kaggle-survey-2017/conversionRates.csv', encoding="ISO-8859-1", low_memory=False)


# In[4]:


df = pd.merge(left=df, right=exchange, how='left', 
              left_on='CompensationCurrency', right_on='originCountry')


# In[5]:


df.columns


# In[6]:


df.shape


# ## Histogramas

# Transformar Age para inteiro para poder enxergar os numeros melhor

# In[7]:


df['Age'] = df['Age'].fillna(0).astype(int)


# Vamos ver um histograma da idade dos participantes

# In[8]:


_ = sns.countplot(x = 'Age', data=df)


# Ficou horrível...
# 
# Vamos adicionar o titulo e aumentar o gráfico

# In[9]:


plt.subplots(figsize=(20,15))
plot = sns.countplot(y="Age", data=df).set_title("Count of respondents by age")


# In[10]:


plt.subplots(figsize=(10,8))
_ = sns.distplot(df['Age']).set_title("Count of respondents by age")


# Distplot não aceita Nulos

# In[11]:


plt.subplots(figsize=(10,8))
_ = plt.hist(df['Age'], normed=True, alpha=0.5)
_ = plt.title("Count of respondents by age")


# ## Boxplot

# In[12]:


_ = sns.boxplot(df['Age']).set_title("Count of respondents by age")


# In[13]:


money_index = df['CompensationAmount'].notnull()


# In[14]:


compensation_check = df[money_index]


# ## Scatterplots (Dispersão)

# In[15]:


df.describe()


# In[16]:


compensation_check['GenderSelect'].value_counts()


# In[17]:


df['exchangeRate'] = df['exchangeRate'].fillna(0)
df['CompensationAmount'] = df['CompensationAmount'].fillna(0)


# In[18]:


df['CompensationAmount'] = df.CompensationAmount.apply(lambda x: 0 if (pd.isnull(x) or (x=='-') or (x==0))
                                                       else float(x.replace(',',''))) 
df['CompensationAmount'] = df['CompensationAmount']*df['exchangeRate']
df = df[df['CompensationAmount']>0]


# In[22]:


df['CompensationAmount'].describe()


# In[ ]:


sns.regplot(x="CompensationAmount", y="tip", data=tips)


# In[20]:


sns.swarmplot(x="GenderSelect", y="CompensationAmount", data=df)


# In[21]:


f = {'CompensationAmount':['median','count']}


temp_df = df.groupby('GenderSelect').agg(f).sort_values(by=[('CompensationAmount','median')], ascending=False)

