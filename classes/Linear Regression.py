
# coding: utf-8

# # Predição de Valor de imóvel com *Regressão Linear*

# ## O que é Regressão Linear?

# ![regression_line](https://cdn-images-1.medium.com/max/1600/1*eeIvlwkMNG1wSmj3FR6M2g.gif)

# ![formula](https://image.slidesharecdn.com/8-1209490505240696-9/95/multiple-linear-regression-16-638.jpg?cb=1489720634)

# ## Quando eu uso uma regressão?

# Quando você está trabalhando com variáveis contínuas. 
# 
# **Exemplo**: Você sabe o valor da sua casa? Como você construiria um modelo para prever o valor dos imóveis da sua cidade?

# 
# ![Question](https://media.giphy.com/media/3o7buirYcmV5nSwIRW/giphy.gif)

# In[147]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn
import seaborn as sns
import numpy as np
get_ipython().magic('matplotlib inline')


# Vamos usar o [Boston Housing Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/) para prever valores de imóveis em Boston 

# Primeiro carregue o dataset. Vamos dar uma roubada e usar o dataset da biblioteca do sklearn.

# In[148]:


from sklearn.datasets import load_boston
boston = load_boston()


# In[149]:


print(boston)


# O dataset vem em formato de dicionário - Precisamos transformar ele em DataFrame. Mas, antes vamos ver as descrições das variáveis. 

# Como vejo as chaves de um dicionário?

# In[150]:


print(boston.keys())


# Vamos verificar a chave que descreve como o dataset está estruturado. É a chave `DESCR`.

# In[151]:


print(boston.DESCR)


# Transformar o dicionário em um dataframe:

# Jeito 1 - Assim, não dá para ver os nomes dos atributos:

# In[152]:


boston_data = pd.DataFrame(boston.data)
boston_data.head()


# Jeito 2 - Com os nomes dos atributos

# In[153]:


boston_data = pd.DataFrame(boston.data, columns=boston.feature_names)


# In[154]:


boston_data.head()


# Qual dessas variáveis parece ser a mais importante? Conseguimos fazer alguma coisa para testar como a nossa variável resposta se comporta com outras variáveis?

# Primeiro precisamos colocar a variável resposta no dataset

# In[155]:


boston_data['target'] = boston.target


# In[156]:


boston_data.head()


# Vamos ver se fizemos tudo certinho verificando o tamanho do Dataframe.

# In[157]:


boston_data.shape


# A descrição não estava mentido. Existem 506 registros e 14 colunas. Fizemos tudo certo até aqui

# Agora sim!
# ![cat_approves](https://media.giphy.com/media/eUQVeW0WEwGxq/giphy.gif)

# ## Bora explorar os dados!

# ![cat_evil](https://media.giphy.com/media/LkjlH3rVETgsg/giphy.gif)

# ### Vamos dar uma olhada na variável resposta

# In[158]:


sns.distplot(boston_data.target)


# ## Simple linear regression

# Agora vamos escolher um atributo que acreditamos ser o mais relevante. E rodar uma regressão linear simples com esse atributo. Mas, como vamos escolher esse atributo entre os 13? A primeira coisa que podemos fazer é levantar algumas hipóteses.

# Inicialmente vamos ver a distribuição de cada variável com o `describe`

# In[159]:


boston_data.describe().T


# Quais variáveis são categóricas? Quais variáveis são numéricas?

# Por enquanto vamos trabalhar só com os atributos em negrito. Mais especificamente, neste primeiro momento, vamos ver o atributo `RM` - o número de quartos

# **Attribute Information (in order)** 
# 
#  - CRIM     per capita crime rate by town
#  - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
#  - INDUS    proportion of non-retail business acres per town
#  - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
#  - NOX      nitric oxides concentration (parts per 10 million)
#  - ** RM       average number of rooms per dwelling**
#  - AGE      proportion of owner-occupied units built prior to 1940
#  - DIS      weighted distances to five Boston employment centres
#  - RAD      index of accessibility to radial highways
#  - TAX      full-value property-tax rate per $10,000
#  - ** PTRATIO  pupil-teacher ratio by town **
#  - B       1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
#  
#  - ** LSTAT lower status of the population **
#  - MEDV     Median value of owner-occupied homes in $1000's

# ### RM  -  average number of rooms per dwelling

# Distribuição da variável

# In[160]:


sns.distplot(boston_data.RM)


# Vamos testar a correlação dessa variável com o valor dos imóveis.

# In[161]:


_ = sns.regplot(x="RM", y="target", data=boston_data)


# Vamos ver a correlção de Pearson usando o método `corr` do pandas

# In[162]:


boston_data.target.corr(boston_data.RM)


# Agora vamos testar fazer um preditor de valor de imóveis usando apenas o número de quartos

# Precisamos separar o dataset em treino e teste e também pegar a variável resposta

# In[163]:


Y = boston_data['target']
X = boston_data.RM.to_frame()


# In[164]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = 42)


# In[165]:


from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, Y_train)

Y_pred = lm.predict(X_test)


# Vamos fazer um regplot para ver como ficaram as nossas predições?

# In[166]:


_ = sns.regplot(x=Y_test, y=Y_pred)


# In[167]:


beta1=lm.coef_
intercepto=lm.intercept_
print(beta1)
print(intercepto)


# In[168]:


X_test.iloc[0]


# In[169]:


Y_pred[0]


# Using the linear regression equation we can get the same results as the predicition function.

# In[207]:


intercepto + (beta1[0] * X_test.iloc[0][0])


# Qual a vantagem de conhecer a fórmula? 
# 
# Nesse modelo consiguimos ver que a cada número de quartos o valor do imóvel cresce 9.19 pontos - Exatamente o valor do beta1[0].

# Vamos verificar o erro dessa solução:

# Falta adicionar o r_squared

# In[171]:


mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
print(mse)


# In[172]:


sklearn.metrics.r2_score(Y_test, Y_pred)  


# Quanto menor o erro quadrático médio, melhor.

# Vamos fazer um modelo com outra variável agora. Agora vamos utilizar o **LSTAT - % lower status of the population**
# 

# ### LSTAT - % lower status of the population

# ![cat_typing](https://media.giphy.com/media/ule4vhcY1xEKQ/giphy.gif)

# Distribuição da variável

# In[173]:


sns.distplot(boston_data.LSTAT)


# Gráfico de Correlação da variável com a resposta

# In[174]:


_ = sns.regplot(x="LSTAT", y="target", data=boston_data)


# Essa variável tem um comportamento igual ao da anterior?

# Correlação de pearson com a variável resposta

# In[175]:


boston_data.target.corr(boston_data.LSTAT)


# Definir novos X e Y

# In[176]:


Y = boston_data['target']
X = boston_data.LSTAT.to_frame()


# Dividir o dataset em treino e teste

# In[177]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = 42)


# Treinar o novo modelo

# In[178]:


lm.fit(X_train, Y_train)

Y_pred = lm.predict(X_test)


# Verificar as predições com um `regplot`

# In[179]:


_ = sns.regplot(x=Y_test, y=Y_pred)


# Vamos usar o r quadrático médio para avaliar qual dos modelos tem o menor erro quadrático

# In[180]:


mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
print(mse)


# Podemos usar também o r quadrado para validar o erro

# In[181]:


sklearn.metrics.r2_score(Y_test, Y_pred)  


# Esse modelo é um pouco melhor que o anterior...

# Agora... A vida num é assim né?! Vocês acham que é melhor fazer um modelo com uma variável ou com todas que eu tenho?

# ## Regressão linear multivariável

# Existe uma maneira de fazer a correlação com todas as variáveis possíveis?

# In[182]:


important_vars = boston_data[['RM', 'PTRATIO', 'LSTAT']]


# In[183]:


# https://seaborn.pydata.org/examples/many_pairwise_correlations.html


# In[184]:


corr = important_vars.corr()


# In[185]:


corr


# ### Matriz de correlação

# In[186]:


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


# Experimentem tirar o mask para ver como fica

# ### Verificar a distribuição da variável que resta - PTRATIO

# ![cat_nail_care](https://media.giphy.com/media/mlvseq9yvZhba/giphy.gif)

# In[187]:


sns.distplot(boston_data.PTRATIO)


# ### Realizar a regressão Linear para as 3 variáveis mais importantes

# In[189]:


Y = boston_data['target']
X = important_vars


# In[190]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = 42)


# In[191]:


lm.fit(X_train, Y_train)

Y_pred = lm.predict(X_test)


# Esse modelo é melhor ou pior que os anteriores? Qual o melhor modelo?

# In[192]:


mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
print(mse)


# In[193]:


sklearn.metrics.r2_score(Y_test, Y_pred)  


# ### Como fica a equação com várias variáveis?

# In[194]:


beta1=lm.coef_
intercepto=lm.intercept_
print(beta1)
print(intercepto)


# In[208]:


intercepto + (beta1[0] * X_test.iloc[0][0] + beta1[1] * X_test.iloc[0][1] + beta1[2] * X_test.iloc[0][2])


# In[209]:


Y_pred[0]


# ### E se eu colocasse todas as variáveis?

# In[212]:


Y = boston_data['target']
X = boston_data.drop('target', axis=1)


# In[215]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = 42)


# In[216]:


lm.fit(X_train, Y_train)

Y_pred = lm.predict(X_test)


# In[217]:


mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
print(mse)


# In[218]:


sklearn.metrics.r2_score(Y_test, Y_pred)  


# ## Outros desafios usando Regressão

#  - [Kaggle: a first experience on Machine Learning and Regression Challenges](https://medium.com/@pramos/kaggle-a-first-experience-on-machine-learning-and-regression-challenges-446436901b7e)
#  - [Predicting House Prices Playground Competition: Winning Kernels](http://blog.kaggle.com/2017/03/29/predicting-house-prices-playground-competition-winning-kernels/)
