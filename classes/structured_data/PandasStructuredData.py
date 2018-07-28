
# coding: utf-8

# # Análise de dados estruturados

# In[1]:


import pandas as pd


# ![panda](https://media.giphy.com/media/HDR31jsQUPqQo/giphy.gif)

# ## Criar um dataframe a partir de um dicionário

# In[2]:


dict = {"country": ["Brazil", "Russia", "India", "China", "South Africa"],
       "capital": ["Brasilia", "Moscow", "New Dehli", "Beijing", "Pretoria"],
       "area": [8.516, 17.10, 3.286, 9.597, 1.221],
       "population": [200.4, 143.5, 1252, 1357, 52.98] }


# In[3]:


# Transformar o dicionário em um dataframe
brics = pd.DataFrame(dict)


# In[4]:


# Ver os primeiros registros desse dataframe
brics.head()


# ## Importar um csv com o pandas

# Vamos utilizar os dados que o Kaggle lançou no ano de 2017 sobre Cientistas de Dados e Data Science. São 5 datasets diferentes:
# 
#  - **schema.csv**: a CSV file with survey schema. This schema includes the questions that correspond to each column name in both the multipleChoiceResponses.csv and freeformResponses.csv.
#  - **multipleChoiceResponses.csv**: Respondents' answers to multiple choice and ranking questions. These are non-randomized and thus a single row does correspond to all of a single user's answers. -freeformResponses.csv: Respondents' freeform answers to Kaggle's survey questions. These responses are randomized within a column, so that reading across a single row does not give a single user's answers.
#  - **conversionRates.csv**: Currency conversion rates (to USD) as accessed from the R package "quantmod" on September 14, 2017
#  - **RespondentTypeREADME.txt**: This is a schema for decoding the responses in the "Asked" column of the schema.csv file.

# In[5]:


# Carregue o dataset multipleChoiceResponses com o pandas 
multiple_choice = pd.read_csv('kaggle-survey-2017/multipleChoiceResponses.csv')


# In[6]:


# Veja as primeiras linhas do dataset
multiple_choice.head()


# In[7]:


#Veja a quantidade de linhas e de colunas do dataset
multiple_choice.shape


# Existem 228 colunas!!!
# ![panda](https://media.giphy.com/media/14aUO0Mf7dWDXW/giphy.gif)

# Vamos ver do que se tratam essas colunas. Como são MUITAS colunas precisamos alterar a configuração padrão do pandas para visualização de linhas e colunas

# In[8]:


pd.set_option('max_rows', 200)
pd.set_option('max_columns', 1000)


# In[9]:


multiple_choice.head()


# Podemos ver só o nome das colunas também utilizando o `columns`. Para ficar mais fácil de visualizar, ao invés de retornar o array, podemos transformar esse dado em uma Series.

# In[10]:


# Use o columns no dataframe e coloque-o em uma Series para facilitar a visualização
pd.Series(multiple_choice.columns)


# Podemos ver mais detalhes do dataset com o `info()`

# In[11]:


multiple_choice.info()


# Podemos dar uma olhada nos tipos de campos que vem em cada uma das colunas númericas com um único comando

# In[12]:


multiple_choice.describe()


# E se eu quiser ver a quantidade de nulos no dataset todo?

# In[13]:


multiple_choice.isnull().sum()


# E se eu quiser fazer a porcentagem de nulos?

# In[14]:


multiple_choice.isnull().sum() / len(multiple_choice)


# O que eu devo fazer se eu quiser ver apenas coluna `JobFactorSalary`?

# In[15]:


multiple_choice['JobFactorSalary']


# Nossa quanto nulo!
# ![sad_panda](https://media.giphy.com/media/3e18NPUVzoxzO/giphy.gif)

# Vamos fazer algumas operações com o pandas para contar o número de nulos que existem nessa coluna

# In[35]:


multiple_choice['JobFactorSalary'].isnull().sum()


# Como eu faço se eu só quiser ver os 10 primeiros registros?

# In[17]:


multiple_choice['JobFactorSalary'][:10]


# E se eu quiser ver 2 colunas ao mesmo tempo? (E apenas essas 2 colunas)

# In[18]:


multiple_choice[['JobFactorSalary', 'JobFactorLearning']][:10]


# O quanto que as pessoas dessa pesquisa estão satisfeitas com o trabalhos? Conseguimos saber isso usando só o pandas?

# In[39]:


multiple_choice['JobSatisfaction'].value_counts()


# Percebemos com esse comando que as pessoas até que estão bastante satisfeitas.

# Agora vamos olhar só as pessoas que estão Super Satisfeitas (Highly Satisfied) com o seu trabalho. Como que eu posso fazer isso?

# In[20]:


# Filtre só quem está com o JobSatisfaction de 10. Guarde isso em uma variável pq é bastante dado
highly_satisfied = multiple_choice[multiple_choice['JobSatisfaction'] == '10 - Highly Satisfied']


# In[21]:


# veja o tamanho do dataset. Ele bateu com a quantidade de pessoas que estão altamente satisfeitas?
highly_satisfied.shape


# In[22]:


# Veja os primeiros 3 registros (todas as colunas) das pessoas altamentes satisfeitas
highly_satisfied[:3]


# E se eu quiser ver as pessoas altamente satisfeitas e que trabalham com python?

# In[24]:


highly_satisfied = multiple_choice['JobSatisfaction'] == '10 - Highly Satisfied'
pythonist = multiple_choice['LanguageRecommendationSelect'] == 'Python'
highly_satisfied_and_pythonist = multiple_choice[highly_satisfied & pythonist]


# In[25]:


highly_satisfied_and_pythonist.shape


# E se tentassemos com a idade? Ver só que está abaixo de 30 anos

# In[26]:


highly_satisfied = multiple_choice['JobSatisfaction'] == '10 - Highly Satisfied'
age = multiple_choice['Age'] < 30.0
highly_satisfied_and_age = multiple_choice[highly_satisfied & age]


# In[27]:


highly_satisfied_and_age.shape


# Quais são as linguagens que a galera altamente satisfeita recomenda?

# In[28]:


multiple_choice[highly_satisfied]['LanguageRecommendationSelect'].value_counts()


# In[53]:


highly_satisfied_languages = multiple_choice[highly_satisfied]['LanguageRecommendationSelect'].value_counts()
language_counts = multiple_choice['JobSatisfaction'][highly_satisfied].notnull().sum()


# In[55]:


language_counts


# In[60]:


(highly_satisfied_languages / language_counts) * 100


# E se eu quiser ordenar esses valores? Do menor para o maior?

# In[61]:


pd.Series((highly_satisfied_languages / language_counts) * 100).sort_values()


# ![arrested_panda](https://media.giphy.com/media/N6funLtVsHW0g/giphy.gif)

# In[ ]:


## adicionar a coluna da currency


# Objetivo da aula: Aprender a estruturar e manipular dados a partir de diferentes fontes,
# utilizando o Pandas e fazer análise desses dados utilizando os métodos e funções da
# biblioteca
# 
# Outcomes esperados:
# 
# ● Discutir as aplicações do uso da biblioteca Pandas em projetos reais de Ciência de
# Dados, trazendo exemplos de empresas de tecnologia
# 
# ● Aprofundar os métodos e funções da biblioteca Pandas utilizados para manipulação
# e análise de dados estruturados
# 
# ● Praticar a importação, leitura, organização e manipulação de dados de diferentes
# fontes para utilizar o Pandas para análise desse dataset
# 
# Perguntas-chave:
# 
# ● Como preparar os dados estruturados para manipular utilizando o Pandas?
# 
# ● Quais são os métodos e funções mais utilizados do Pandas em projetos de Ciência
# de Dados?
# 
# ● Em que estágios de um projeto a biblioteca pode ajudar o Cientista de Dados a
# manipular dados estruturados?
