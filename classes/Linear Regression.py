
# coding: utf-8

# # Predição de Valor de imóvel com *Regressão Linear*

# ## O que é Regressão Linear?

# ![regression_line](https://cdn-images-1.medium.com/max/1600/1*eeIvlwkMNG1wSmj3FR6M2g.gif)

# [ADICIONAR A FORMULA AQUI]

# ## Quando eu uso uma regressão?

# Quando você está trabalhando com variáveis contínuas. **Exemplo**: Você sabe o valor da sua casa? Como você construiria um modelo para prever o valor dos imóveis da sua cidade?

# 
# ![Question](https://media.giphy.com/media/3o7buirYcmV5nSwIRW/giphy.gif)

# In[59]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
get_ipython().magic('matplotlib inline')


# Vamos usar o Boston Housing Dataset para prever valores de imóveis em Boston 

# Primeiro carregue o dataset

# In[7]:


from sklearn.datasets import load_boston
boston = load_boston()


# In[11]:


print(boston)


# O dataset vem em formato de dicionário - Precisamos transformar ele em DataFrame. Mas, antes vamos ver as descrições das variáveis. 

# Como vejo as chaves de um dicionário?

# In[8]:


print(boston.keys())


# In[9]:


print(boston.DESCR)


# Transforma o dicionário em um dataframe:

# Jeito 1 - Assim, não dá para ver os nomes dos atributos:

# In[16]:


boston_data = pd.DataFrame(boston.data)
boston_data.head()


# Jeito 2 - Com os nomes dos atributos

# In[17]:


boston_data = pd.DataFrame(boston.data, columns=boston.feature_names)


# In[18]:


boston_data.head()


# Qual dessas variáveis parece ser a mais importante? Conseguimos fazer alguma coisa para testar como a nossa variável resposta se comporta com outras variáveis?

# Primeiro precisamos colocar a variável resposta no dataset

# In[20]:


boston_data['target'] = boston.target


# In[21]:


boston_data.head()


# Agora sim!
# ![cat_approves](https://media.giphy.com/media/eUQVeW0WEwGxq/giphy.gif)

# In[22]:


boston_data.shape


# A descrição não estava mentido. Existem 506 registros e 14 colunas. Fizemos tudo certo até aqui

# ## Simple linear regression

# Vamos ver a distribuição da variável resposta

# In[129]:


sns.distplot(boston_data.target)


# Agora vamos escolher um atributo que acreditamos ser o mais relevante. E rodar uma regressão linear simples com esse atributo. Mas, como vamos escolher esse atributo entre os 13? Vamos levantar algumas hipóteses.

# Inicialmente vamos ver a distribuição de cada variável com o `describe`

# In[27]:


boston_data.describe().T


# Por enquanto vamos trabalhar só com os atributos em negrito. Mais especificamente vamos ver o atributo `RM` - o número de quartos

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

# In[38]:


sns.distplot(boston_data.RM)


# Vamos testar a correlação dessa variável com o valor dos imóveis.

# In[43]:


_ = sns.regplot(x="RM", y="target", data=boston_data)


# Vamos ver a correlção de Pearson usando o método `corr` do pandas

# In[47]:


boston_data.target.corr(boston_data.RM)


# Agora vamos testar fazer um preditor de valor de imóveis usando apenas o número de quartos

# Precisamos separar o dataset em treino e teste

# In[74]:


Y = boston_data['target']
X = boston_data.RM.to_frame()


# In[75]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = 42)


# In[76]:


from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, Y_train)

Y_pred = lm.predict(X_test)


# Vamos fazer um regplot para ver como ficaram as nossas predições?

# In[77]:


_ = sns.regplot(x=Y_test, y=Y_pred)


# In[85]:


beta1=lm.coef_
intercepto=lm.intercept_
print(beta1)
print(intercepto)


# In[127]:


X_test.iloc[0]


# Using the formula we can get the same results as the predicition function.

# Y = a + bX

# In[125]:


intercepto + (beta1[0] * X_test.iloc[0])


# In[128]:


Y_pred[0]


# Qual a vantagem? Nesse modelo consiguimos ver que a cada número de quartos o valor do imóvel cresce 9.19 pontos - Exatamente o valor do beta1[0].

# Vamos verificar o erro dessa solução:

# In[78]:


mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
print(mse)


# Quanto menor o erro quadrático médio, melhor.

# Vamos fazer um modelo com outra variável agora. Agora vamos utilizar o **LSTAT - % lower status of the population**
# 

# ### LSTAT - % lower status of the population

# ![cat_typing](https://media.giphy.com/media/ule4vhcY1xEKQ/giphy.gif)

# Distribuição da variável

# In[133]:


sns.distplot(boston_data.LSTAT)


# Gráfico de Correlação da variável com a resposta

# In[134]:


_ = sns.regplot(x="LSTAT", y="target", data=boston_data)


# Essa variável tem um comportamento igual ao da anterior?

# Correlação de pearson com a variável resposta

# In[135]:


boston_data.target.corr(boston_data.LSTAT)


# Definir novos X e Y

# In[138]:


Y = boston_data['target']
X = boston_data.LSTAT.to_frame()


# Dividir o dataset em treino e teste

# In[140]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = 42)


# Treinar o novo modelo

# In[141]:


lm.fit(X_train, Y_train)

Y_pred = lm.predict(X_test)


# Verificar as predições com um `regplot`

# In[142]:


_ = sns.regplot(x=Y_test, y=Y_pred)


# Vamos usar o r_squared para avaliar qual dos modelos tem o menor erro quadrático

# In[144]:


mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
print(mse)


# Esse modelo é um pouco melhor que o anterior...

# Agora a vida num é assim né? Vocês acham que é melhor fazer um modelo com uma variável ou com todas que eu tenho?

# ## Regressão linear multivariável

# ### Matriz de correlação

# ### Verificar a distribuição da variável que resta - PTRATIO

# ### Realizar a regressão Linear para as 3 variáveis mais importantes

# ### Como fica a equação com várias variáveis?

# ### Qual é o melhor modelo?

# ### E se eu colocasse todas as variáveis?

# Objetivo da aula: Realizar uma análise de um problema de negócio com a regressão linear, partindo da determinação do problema até a aplicação do modelo de regressão linear para validar hipóteses e apresentar uma solução.
# 
# 
# Outcomes esperados:
# 
# Saber o objetivos, vantagens e desvantagens do uso de regressão linear como modelo
# 
# Conhecer diferentes funcionamentos da regressão linear 
# 
# Saber mensurar e comparar modelos
# 
# 
# Breve descrição: o que é regressão linear, qual o objetivo, é um modelo paramétrico (strong assumptions, strong conclusions), comentar sobre as vantagens e desvantagens (principalmente na ótica de negócios — é simples e interpretável). Ter exemplos sobre variáveis X e Y. Como funciona, qual a métrica de erro (standard error, residual sum of squares, R2), como os coeficientes são estimados, como o modelo funciona em forma de matriz, como funciona no caso de múltiplas variáveis, como mensurar e como comparar modelos. Ao final da aula, os alunos devem ser capazes de identificar aplicações de regressão linear em cenários de negócios.
# 
# 
# O que eles vão ter antes da aula é o capítulo de Regressão do DataCamp. Você pode, se quiser, complementar com algum texto ou material que explique melhor os conceitos, passando uma atividade pré-aula para os alunos. 
# Essa é exatamente a ideia por trás da nossa metodologia: todo conceito pode ser estudado antes, e durante a aula nós focamos em botar em prática o tema. Você pode, também, abordar conceitos e fazer explicações durante a prática - a teoria não precisa ser passada antes da prática, e nós sugerimos utilizar a prática como exemplo para ensinar a teoria. O importante é que os alunos saiam da aula tendo visto uma apliação completa de regressão linear e que consigam replicar sozinhos depois.

# ## Outros desafios usando Regressão

#  - [Kaggle: a first experience on Machine Learning and Regression Challenges](https://medium.com/@pramos/kaggle-a-first-experience-on-machine-learning-and-regression-challenges-446436901b7e)
#  - [Predicting House Prices Playground Competition: Winning Kernels](http://blog.kaggle.com/2017/03/29/predicting-house-prices-playground-competition-winning-kernels/)
