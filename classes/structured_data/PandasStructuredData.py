
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
#  - **multipleChoiceResponses.csv**: Respondents' answers to multiple choice and ranking questions. These are non-randomized and thus a single row does correspond to all of a single user's answers. 
#  -**freeformResponses.csv:** Respondents' freeform answers to Kaggle's survey questions. These responses are randomized within a column, so that reading across a single row does not give a single user's answers.
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


# Nossa quanto nulo!
# ![sad_panda](https://media.giphy.com/media/3e18NPUVzoxzO/giphy.gif)

# O que eu devo fazer se eu quiser ver apenas coluna `JobFactorSalary`?

# In[15]:


multiple_choice['JobFactorSalary']


# Vamos fazer algumas operações com o pandas para contar o número de nulos que existem nessa coluna

# In[16]:


multiple_choice['JobFactorSalary'].isnull().sum()


# Como eu faço se eu só quiser ver os 10 primeiros registros?

# In[17]:


multiple_choice['JobFactorSalary'][:10]


# E se eu quiser ver 2 colunas ao mesmo tempo? (E apenas essas 2 colunas)

# In[18]:


multiple_choice[['JobFactorSalary', 'JobFactorLearning']][:10]


# O quanto que as pessoas dessa pesquisa estão satisfeitas com o trabalhos? Conseguimos saber isso usando só o pandas?

# In[19]:


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

# In[23]:


highly_satisfied = multiple_choice['JobSatisfaction'] == '10 - Highly Satisfied'
pythonist = multiple_choice['LanguageRecommendationSelect'] == 'Python'
highly_satisfied_and_pythonist = multiple_choice[highly_satisfied & pythonist]


# In[24]:


highly_satisfied_and_pythonist.shape


# E se tentassemos com a idade? Ver só que está abaixo de 30 anos

# In[25]:


highly_satisfied = multiple_choice['JobSatisfaction'] == '10 - Highly Satisfied'
age = multiple_choice['Age'] < 30.0
highly_satisfied_and_age = multiple_choice[highly_satisfied & age]


# In[26]:


highly_satisfied_and_age.shape


# Quais são as linguagens que a galera altamente satisfeita recomenda?

# In[27]:


multiple_choice[highly_satisfied]['LanguageRecommendationSelect'].value_counts()


# In[28]:


highly_satisfied_languages = multiple_choice[highly_satisfied]['LanguageRecommendationSelect'].value_counts()
language_counts = multiple_choice['JobSatisfaction'][highly_satisfied].notnull().sum()


# In[29]:


language_counts


# In[30]:


(highly_satisfied_languages / language_counts) * 100


# E se eu quiser ordenar esses valores? Do menor para o maior?

# In[31]:


pd.Series((highly_satisfied_languages / language_counts) * 100).sort_values()


# ### Desafio 1

# Qual o país que tem a maior quantidade de dados onde as pessoas preencheram a coluna que tem o menor número dos dados?
# 
# Dica: Você precisará ordenar os campos pela quantidade de nulos (ou não nulos) e depois ver o país dessa galera.

# ![arrested_panda](https://media.giphy.com/media/N6funLtVsHW0g/giphy.gif)

# In[32]:


multiple_choice.isnull().sum().sort_values(ascending=False)


# In[33]:


multiple_choice[multiple_choice['WorkToolsFrequencyAngoss'].notnull()]['Country'].value_counts()


# ## Selecionando por index

# E se eu quiser pegar os valores de uma linha específica do dataframe?

# In[34]:


multiple_choice.iloc[10,]


# Também posso ver só o valor de uma coluna, sem escrever o nome, somente pela sua posição

# In[35]:


multiple_choice.iloc[:,0]


# Mais detalhes sobre `loc`, `iloc` e `ix` podem ser vistas nesse [link](https://www.shanelynn.ie/select-pandas-dataframe-rows-and-columns-using-iloc-loc-and-ix/)

# ### Desafio 2

# Será que temos uma quantidade gigante de nulos nesse dataset porque as pessoas não preencheram essas perguntas por multipla escolha, por que responderam no modo livre? 
# 
# Para validar essa hipótese teremos que carregar o outro dataset, que contém as perguntas em forma livre. e juntar (Pelo menos uma das variáveis) dos dois os datasets. Eu escolhi `DataScienceIdentity`

# ![challenge_panda](https://media.giphy.com/media/K9z3im98oo9Ve/giphy.gif)

# In[36]:


free_responses = pd.read_csv('kaggle-survey-2017/freeformResponses.csv')


# In[37]:


free_responses.head()


# In[38]:


free_responses.shape


# In[39]:


free_responses.isnull().sum().sort_values()


# In[40]:


free_index = free_responses['DataScienceIdentityFreeForm'].notnull()


# In[41]:


identity_check = pd.DataFrame({'IdentityFree': free_responses[free_index]['DataScienceIdentityFreeForm'], 
              'IdentitySelect': multiple_choice[free_index]['DataScienceIdentitySelect']})


# In[42]:


identity_check.shape


# In[43]:


identity_check.isnull().sum()


# Aparentemente não foi isso que aconteceu...

# ## Alterando o dataset original

# In[44]:


df = multiple_choice.copy()


# In[45]:


df['LearningDataScience'].value_counts()


# In[46]:


def replace_value(row):
    if row == "Yes, I'm focused on learning mostly data science skills":
        return "yes"
    elif row == "Yes, but data science is a small part of what I'm focused on learning":
        return "so so"
    elif row == "No, I am not focused on learning data science skills":
        return "no"


# In[47]:


df['LearningDataScienceSimple'] = df['LearningDataScience'].apply(replace_value)


# Agora podemos ver os novos valores desses campos

# In[48]:


df['LearningDataScienceSimple'].value_counts()


# In[49]:


df.shape


# E agora que a outra coluna muito complexa não serve mais, podemos descartá-la

# In[50]:


df.drop(['LearningDataScience'], axis=1, inplace=True)


# In[51]:


df.shape


# Pela quantidade de linhas no dataset percebemos que o `value_counts()` não retorna os valores nulos - e nós temos MUITOS valores nulos nessa coluna. Podemos utilizar um método do pandas para trocar os NAs por uma categoria nossa.

# In[52]:


df['LearningDataScienceSimple'].fillna("did not answer the question", inplace=True)


# In[53]:


df['LearningDataScienceSimple'].value_counts()


# Outra forma de alterar o dataset é utilizando funções _in place_ para isso utilizaremos o `lambda`

# Por exemplo: E se eu quiser atualizar a idade dos participantes? O dataset foi coletado em 2017 e já estamos em 2018

# In[54]:


df['NewAge'] = df['Age'].apply(lambda x: x + 1)


# Vamos ver se funcionou? Vamos dar uma olhada nos primeiros 5 registros, com a coluna 'Age' e a 'NewAge' lado a lado

# In[55]:


df[['Age','NewAge']][:5]


# Ficou mais claro a proporção de pessoas que não responderam agora o/

# ### Desafio 3

# Separar os datasets pelas pessoas que os responderam. Para isso você vai ter que carregar o dataset `schema.csv`.
# 
# Como uns datasets ficariam muito pequenos, sugiro que você utilize os seus conhecimentos recém adquiridos e crie 4 datasets distintos.

# ![panda_playground](https://media.giphy.com/media/ieaUdBJJC19uw/giphy.gif)

# In[56]:


schema = pd.read_csv('kaggle-survey-2017/schema.csv')


# In[57]:


schema.head()


# In[58]:


schema.Asked.value_counts()


# In[59]:


coding_worker_column = schema[schema['Asked'] == 'CodingWorker']['Column']
all_column = schema[schema['Asked'] == 'All']['Column']
learners_column = schema[schema['Asked'] == 'Learners']['Column']
others_columns = schema[(~ schema['Asked'].isin(['CodingWorker', 'All', 'Learners']))]['Column']


# In[60]:


all_multiple_selection_cols = [c for c in all_column.values if 'freeform' not in c.lower()]
coding_worker_multiple_selection_cols = [c for c in coding_worker_column.values if 'freeform' not in c.lower()]
learners_multiple_selection_cols = [c for c in learners_column.values if 'freeform' not in c.lower()]
others_multiple_selection_cols = [c for c in others_columns.values if 'freeform' not in c.lower()]


# In[61]:


others_multiple_selection_cols


# In[63]:


all_multiple_selection = multiple_choice[all_multiple_selection_cols]
coding_worker_multiple_selection = multiple_choice[coding_worker_multiple_selection_cols]
learners_multiple_selection = multiple_choice[learners_multiple_selection_cols]
others_multiple_selection = multiple_choice[others_multiple_selection_cols]


# In[64]:


print(all_multiple_selection.shape)
print(coding_worker_multiple_selection.shape)
print(learners_multiple_selection.shape)
print(others_multiple_selection.shape)


# Lembrando que nós jogamos fora quem era 'free_form' - Logo, haverão sempre menos colunas do que vimos antes na quantidade por tipo de pessoa que respondeu.

# In[68]:


all_multiple_selection.to_csv('all_multiple_selection.csv')
coding_worker_multiple_selection.to_csv('coding_worker_multiple_selection.csv')
learners_multiple_selection.to_csv('learners_multiple_selection.csv')
others_multiple_selection.to_csv('others_multiple_selection.csv')

