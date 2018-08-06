
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


# In[6]:


df.columns


# In[7]:


df.shape


# ## Histogramas

# Transformar Age para inteiro para poder enxergar os numeros melhor

# In[8]:


df['Age'] = df['Age'].fillna(0).astype(int)


# Vamos ver um histograma da idade dos participantes

# In[9]:


_ = sns.countplot(x = 'Age', data=df)


# Ficou horrível...
# 
# Vamos adicionar o titulo e aumentar o gráfico

# In[10]:


plt.subplots(figsize=(20,15))
plot = sns.countplot(y="Age", data=df).set_title("Count of respondents by age")


# In[11]:


plt.subplots(figsize=(10,8))
_ = sns.distplot(df['Age']).set_title("Count of respondents by age")


# In[52]:


_ = sns.distplot(df['Age'], kde=False).set_title("Count of respondents by age")


# Distplot não aceita Nulos

# In[12]:


plt.subplots(figsize=(10,8))
_ = plt.hist(df['Age'], normed=True, alpha=0.5)
_ = plt.title("Count of respondents by age")


# In[57]:


sns.countplot(y="MajorSelect", data=df, palette="Greens_d")


# E se trocarmos os y por um x?

# In[68]:


sns.countplot(y="FormalEducation", data=df)


# In[70]:


df['PastJobTitlesSelect'].value_counts()


# Atividade: Construir uma função que agrupe os tipos de empregos e plot em um barplot.

# In[73]:


df['FirstTrainingSelect'].value_counts()


# In[74]:


df['MLSkillsSelect'].value_counts()


# In[75]:


df['MLTechniquesSelect'].value_counts()


# In[76]:


df['TimeGatheringData'].value_counts()


# In[77]:


df['AlgorithmUnderstandingLevel'].value_counts()


# In[78]:


df['WorkChallengesSelect'].value_counts()


# In[79]:


df['RemoteWork'].value_counts()


# In[81]:


df['LearningDataScienceTime'].value_counts()


# ## Boxplot

# In[13]:


_ = sns.boxplot(df['Age']).set_title("Count of respondents by age")


# In[14]:


money_index = df['CompensationAmount'].notnull()


# In[15]:


compensation_check = df[money_index]


# In[16]:


df.describe()


# In[17]:


compensation_check['GenderSelect'].value_counts()


# In[18]:


df['exchangeRate'] = df['exchangeRate'].fillna(0)
df['CompensationAmount'] = df['CompensationAmount'].fillna(0)


# In[19]:


df['CompensationAmount'] = df.CompensationAmount.apply(lambda x: 0 if (pd.isnull(x) or (x=='-') or (x==0))
                                                       else float(x.replace(',',''))) 
df['CompensationAmount'] = df['CompensationAmount']*df['exchangeRate']
df = df[df['CompensationAmount']>0]


# In[20]:


df['CompensationAmount'].describe()


# In[25]:


sns.boxplot(x="GenderSelect", y="CompensationAmount",
            data=df)
sns.despine(offset=10, trim=True)


# Tem um outlier nesse conj. de dados que está atrapalhando a nossa visualização... Podemos removê-lo usando boolean indexes. Vamos usar pessoas que ganham até 2000000.

# In[38]:


sns.boxplot(x="GenderSelect", y="CompensationAmount",
            data=df[df['CompensationAmount'] < 2000000])
sns.despine(offset=10, trim=True)


# Agora vamos colocar os titulos em 45º

# In[45]:


sns.boxplot(x="GenderSelect", y="CompensationAmount",
            data=df[df['CompensationAmount'] < 2000000])
sns.despine(offset=10, trim=True)
plt.xticks(rotation=15)


# ## Scatterplots (Dispersão)

# In[55]:


sns.jointplot(x="LearningCategorySelftTaught", y="LearningCategoryWork", data=df);


# In[71]:


sns.kdeplot(df['LearningCategorySelftTaught'])
sns.kdeplot(df['LearningCategoryWork'])
sns.kdeplot(df['LearningCategoryOnlineCourses'])
sns.kdeplot(df['LearningCategoryUniversity'])
#sns.kdeplot(df['LearningCategoryKaggle'])
sns.kdeplot(df['LearningCategoryOther'])
plt.legend();


# Desafio:
# 
# Fazer um Heatmap. Siga os passos [desse tutorial](https://seaborn.pydata.org/examples/many_pairwise_correlations.html). Atenção! Use apenas as variáveis numéricas

# ## Gráficos do Kaggle

# http://blog.kaggle.com/2017/10/30/introducing-kaggles-state-of-data-science-machine-learning-report-2017/

# Joyplots -> http://blog.kaggle.com/2017/07/20/joyplots-tutorial-with-insect-data/
# 
# Plots de mapas -> http://blog.kaggle.com/2016/11/30/seventeen-ways-to-map-data-in-kaggle-kernels/
