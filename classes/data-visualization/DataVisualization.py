
# coding: utf-8

# # Visualização de dados para tomada de decisão

# ![](https://media.giphy.com/media/zw69pUViBZCZW/giphy.gif)

# In[91]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import re
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

# Vamos analisar a idade dos cientistas de dados dessa pesquisa. Qual a idade média? Quantos anos tem a pessoa mais velha dessa pesquisa? 

# Para conseguir usar o `countplot` vamos transformar `Age` para inteiro para poder enxergar os numeros melhor

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


# E se eu não quiser um eixo x mais limpo? Só para ver a distribuição em si?

# In[11]:


plt.subplots(figsize=(10,8))
_ = sns.distplot(df['Age']).set_title("Count of respondents by age")


# E para remover a curva de tendencia?

# In[52]:


_ = sns.distplot(df['Age'], kde=False).set_title("Count of respondents by age")


# **Nota**: Distplot não aceita Nulos. O grande número de pessoas que ficaram com idade zero na verdade são pessoas que não preencheram. 

# ### Desafio 1
# 
# Ao invés de substituir os valores nulos pelo número zero, substitua-os pelo valor médio da idade no dataset. Plot a idade novamente. Além disso, troque as cores do gráfico. Para isso use [o guia de paletas do seaborn](https://seaborn.pydata.org/tutorial/color_palettes.html#palette-tutorial).
# 
# ![monstros_sa](https://media.giphy.com/media/zxxXYJqTlpBnO/giphy.gif)

# ### Como seria o mesmo histograma usando apenas matplotlib?

# In[12]:


plt.subplots(figsize=(10,8))
_ = plt.hist(df['Age'], normed=True, alpha=0.5)
_ = plt.title("Count of respondents by age")


# ### Quais são as áreas de graduação dos cientistas de dados?

# In[84]:


sns.countplot(y="MajorSelect", data=df, palette="Greens_d").set_title("Count of respondents by major")


# Para ficar mais facil de ver podemos ordenar as barras

# In[88]:


_ = sns.countplot(y="MajorSelect", data=df, palette="Greens_d", order=df['MajorSelect'].value_counts().index) .set_title("Count of respondents by major")


# Agora ficou bem mais fácil de tirar conclusões sobre os cursos.
# 
# A maioria dos cientistas de dados estudou ciência da computação, matemática ou engenharia.

# E se trocarmos os y por um x?

# ### Qual o maior grau de educação dos cientistas de dados?

# In[89]:


_ = sns.countplot(y="FormalEducation", data=df, palette="Greens_d", order=df['FormalEducation'].value_counts().index) .set_title("Count of respondents by formal education")


# ### Desafio 2
# ##### Quais os empregos anteriores dos cientistas de dados?
# 
# Para fazer esse desafio você vai consultar a coluna `PastJobTitlesSelect`. Veja que essa coluna possui varios valores. Você precisará criar um método para reduzir a granularidade dessa coluna.

# Dica: A solução fica mais fácil se você usar [expressões regulares](https://pt.wikipedia.org/wiki/Express%C3%A3o_regular). Para testá-las use [esse site](https://regexr.com/)

# ![finn_mathematical](https://media.giphy.com/media/ccQ8MSKkjHE2c/giphy.gif)

# In[70]:


df['PastJobTitlesSelect'].value_counts()


# In[103]:


df['PastJobTitlesSelect'] = df['PastJobTitlesSelect'].fillna('NULL')


# In[105]:


past_job_category = []
for s in df['PastJobTitlesSelect']:
    past_job_category.append(re.sub(r'(?=,).*', '', s))
    
df['new_job_category'] = past_job_category


# In[107]:


df[['PastJobTitlesSelect', 'new_job_category']].head(5)


# In[108]:


df['new_job_category'].value_counts()


# In[109]:


_ = sns.countplot(y="new_job_category", data=df, palette="Greens_d", order=df['new_job_category'].value_counts().index) .set_title("Count of respondents by previous job category")


# ### Será que o trabalho remoto impacta no tempo que um cientista passa coletando dados?

# In[124]:


df['RemoteWork'].value_counts()


# In[129]:


df['TimeGatheringData'].value_counts()


# In[128]:


df['TimeGatheringData'] = df['TimeGatheringData'].fillna(-1)


# In[132]:


sns.swarmplot(x="RemoteWork", y="TimeGatheringData", data=df)


# Parece que não muda muito... 

# ### E se eu quiser saber se o tempo que a pessoa passa gerando visualizações impacta no tempo que ela gasta em visualização em um projeto?

# In[134]:


df['WorkToolsSelect'].value_counts()


# In[161]:


df['WorkDataVisualizations'] = df['WorkDataVisualizations'].fillna('NULL')
work_visualization = []
for s in df['WorkDataVisualizations']:
    work_visualization.append(re.sub(' of projects', '', s))
    
df['work_visualization'] = work_visualization


# In[162]:


df['work_visualization'].value_counts()


# In[165]:


plt.subplots(figsize=(10,8))
sns.swarmplot(x="work_visualization", y="TimeVisualizing", data=df, 
              order=['100%', '51-75%', '26-50%', '10-25%', 'Less than 10%', 'None', 'NULL'])


# ### Desafio 3
# 
# Fazer um Heatmap mostrando a [correlação](https://pt.wikipedia.org/wiki/Coeficiente_de_correla%C3%A7%C3%A3o_de_Pearson) dos tempos das etapas de um projeto de Data Science. 
# 
# São elas:
# 
#  - TimeGatheringData
#  - TimeVisualizing
#  - TimeModelBuilding
#  - TimeFindingInsights
#  - TimeProduction
# 
# Siga os passos [desse tutorial](https://seaborn.pydata.org/examples/many_pairwise_correlations.html). Atenção! Use apenas essas variáveis.

# ![crazy_finn](https://media.giphy.com/media/KI9oNS4JBemyI/giphy.gif)

# In[170]:


from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white")

# Generate a large random dataset
rs = np.random.RandomState(33)
d = df[['TimeGatheringData', 'TimeVisualizing', 'TimeModelBuilding', 'TimeFindingInsights', 'TimeProduction']]
# Compute the correlation matrix
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# ### E se eu quiser ter uma ideia do tempo que é investido criando-se modelos?

# ## Boxplot

# In[166]:


_ = sns.boxplot(df['TimeModelBuilding']).set_title("Time spent by building models")


# Eu também posso usar boxplots com variáveis categóricas...

# ### E Se eu quiser verificar o salário das pessoas por gênero?

# Primeiramente, vamos usar apenas as pessoas que tenham valores de salário que é representado pela variável `CompensationAmount`

# In[172]:


money_index = df['CompensationAmount'].notnull()
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

# ### E se eu quiser ver a distribuição da probabilidade das pessoas que aprenderam algo (da profissão) no Trabalho e que foram auto didatas? 

# In[55]:


sns.jointplot(x="LearningCategorySelftTaught", y="LearningCategoryWork", data=df);


# E o que eu posso fazer se eu quiser ver as probabilidades de todas as categorias `LearningCategory(...)` todas juntas?

# In[71]:


sns.kdeplot(df['LearningCategorySelftTaught'])
sns.kdeplot(df['LearningCategoryWork'])
sns.kdeplot(df['LearningCategoryOnlineCourses'])
sns.kdeplot(df['LearningCategoryUniversity'])
#sns.kdeplot(df['LearningCategoryKaggle'])
sns.kdeplot(df['LearningCategoryOther'])
plt.legend();


# ## Desafio 4
# 
# Existem ainda várias perguntas que ficaram sem resposta, do tipo:
# 
#  1. Quais os maiores desafios de um cientista de dados? (`WorkChallengesSelect`)
#  - Quais os algoritmos mais utilizados em data science? (`WorkAlgorithmsSelect`)
#  - Quais os setores que mais empregam cientistas de dados? (`EmployerIndustry`)
#  - Qual o tamanho das empresas que contratam cientistas de dados? (`EmployerSize`)
#  
# Organizem-se em duplas para resolver esses desafios.

# ![challenge](https://media.giphy.com/media/d4zHnLjdy48Cc/giphy.gif)

# ## Gráficos mais complexos

#  - Uma das análises desse dataset no blog do kaggle -> http://blog.kaggle.com/2017/10/30/introducing-kaggles-state-of-data-science-machine-learning-report-2017/
#  - Joyplots -> http://blog.kaggle.com/2017/07/20/joyplots-tutorial-with-insect-data/
#  - Plots de mapas -> http://blog.kaggle.com/2016/11/30/seventeen-ways-to-map-data-in-kaggle-kernels/
