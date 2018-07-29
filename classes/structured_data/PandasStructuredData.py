
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


# In[69]:


# Ver os primeiros registros desse dataframe


# ## Importar um csv com o pandas

# Vamos utilizar os dados que o Kaggle lançou no ano de 2017 sobre Cientistas de Dados e Data Science. São 5 datasets diferentes:
# 
#  - **schema.csv**: a CSV file with survey schema. This schema includes the questions that correspond to each column name in both the multipleChoiceResponses.csv and freeformResponses.csv.
#  - **multipleChoiceResponses.csv**: Respondents' answers to multiple choice and ranking questions. These are non-randomized and thus a single row does correspond to all of a single user's answers. 
#  -**freeformResponses.csv:** Respondents' freeform answers to Kaggle's survey questions. These responses are randomized within a column, so that reading across a single row does not give a single user's answers.
#  - **conversionRates.csv**: Currency conversion rates (to USD) as accessed from the R package "quantmod" on September 14, 2017
#  - **RespondentTypeREADME.txt**: This is a schema for decoding the responses in the "Asked" column of the schema.csv file.

# In[70]:


# Carregue o dataset multipleChoiceResponses com o pandas 


# In[71]:


# Veja as primeiras linhas do dataset


# In[72]:


#Veja a quantidade de linhas e de colunas do dataset


# Existem 228 colunas!!!
# ![panda](https://media.giphy.com/media/14aUO0Mf7dWDXW/giphy.gif)

# Vamos ver do que se tratam essas colunas. Como são MUITAS colunas precisamos alterar a configuração padrão do pandas para visualização de linhas e colunas

# In[8]:


pd.set_option('max_rows', 200)
pd.set_option('max_columns', 1000)


# In[73]:


# Vamos ver um pedaço do dataset


# Podemos ver só o nome das colunas também utilizando o `columns`. Para ficar mais fácil de visualizar, ao invés de retornar o array, podemos transformar esse dado em uma Series.

# In[74]:


# Use o columns no dataframe e coloque-o em uma Series para facilitar a visualização


# Podemos ver mais detalhes do dataset com o `info()`

# In[75]:


# use o info no dataframe


# Podemos dar uma olhada nos tipos de campos que vem em cada uma das colunas númericas com um único comando

# In[76]:


# veja as estatisticas dos campos númericos


# E se eu quiser ver a quantidade de nulos no dataset todo?

# In[77]:


# Veja nulos do dataset


# E se eu quiser fazer a porcentagem de nulos?

# In[78]:


# veja a porcentagem de nulos


# Nossa quanto nulo!
# ![sad_panda](https://media.giphy.com/media/3e18NPUVzoxzO/giphy.gif)

# O que eu devo fazer se eu quiser ver apenas coluna `JobFactorSalary`?

# In[79]:


# veja uma única coluna


# Vamos fazer algumas operações com o pandas para contar o número de nulos que existem nessa coluna

# In[80]:


# Veja o número de nulos apenas desta coluna


# Como eu faço se eu só quiser ver os 10 primeiros registros?

# In[81]:


# Selecione apenas os 10 primeiros registros


# E se eu quiser ver 2 colunas ao mesmo tempo? (E apenas essas 2 colunas)

# In[82]:


# Veja as colunas 'JobFactorSalary' e 'JobFactorLearning'


# O quanto que as pessoas dessa pesquisa estão satisfeitas com o trabalhos? Conseguimos saber isso usando só o pandas?

# In[83]:


# Use uma função para contar o número de cada uma das variáveis da coluna JobSatisfaction


# Percebemos com esse comando que as pessoas até que estão bastante satisfeitas.

# Agora vamos olhar só as pessoas que estão Super Satisfeitas (Highly Satisfied) com o seu trabalho. Como que eu posso fazer isso?

# In[84]:


# Filtre só quem está com o JobSatisfaction de 10. Guarde isso em uma variável pq é bastante dado


# In[85]:


# Veja o tamanho do dataset. Ele bateu com a quantidade de pessoas que estão altamente satisfeitas?


# In[86]:


# Veja os primeiros 3 registros (todas as colunas) das pessoas altamentes satisfeitas


# E se eu quiser ver as pessoas altamente satisfeitas e que trabalham com python?

# In[87]:


# Preciso usar 2 filtros. JobSatisfaction e LanguageRecommendationSelect


# In[88]:


# Só para validar que deu certo é bom ver o shape do resultado


# E se tentassemos com a idade? Ver só que está abaixo de 30 anos

# In[25]:


# Aqui precisa usar JobSatisfaction e Age


# In[89]:


# Só para validar que deu certo é bom ver o shape do resultado


# Quais são as linguagens que a galera altamente satisfeita recomenda? Como vc faria para mostrar esses valores em porcentagem?

# In[92]:


# preciso usar a variável que eu criei umas células acima e LanguageRecommendationSelect e também do número de pessoas que responderam a pergunta JobSatisfaction


# E se eu quiser ordenar esses valores? Do menor para o maior?

# ### Desafio 1

# Qual o país que tem a maior quantidade de dados onde as pessoas preencheram a coluna que tem o menor número dos dados?
# 
# Dica: Você precisará ordenar os campos pela quantidade de nulos (ou não nulos) e depois ver o país dessa galera.

# ![arrested_panda](https://media.giphy.com/media/N6funLtVsHW0g/giphy.gif)

# ## Selecionando por index

# E se eu quiser pegar os valores de uma linha específica do dataframe?

# In[94]:


# Qual comando eu usuária para pegar o registro da linha 11?


# Também posso ver só o valor de uma coluna, sem escrever o nome, somente pela sua posição

# In[95]:


# E para pegar a primeira coluna do dataset?


# Mais detalhes sobre `loc`, `iloc` e `ix` podem ser vistas nesse [link](https://www.shanelynn.ie/select-pandas-dataframe-rows-and-columns-using-iloc-loc-and-ix/)

# ### Desafio 2

# Será que temos uma quantidade gigante de nulos nesse dataset porque as pessoas não preencheram essas perguntas por multipla escolha, por que responderam no modo livre? 
# 
# Para validar essa hipótese teremos que carregar o outro dataset, que contém as perguntas em forma livre. e juntar (Pelo menos uma das variáveis) dos dois os datasets. Eu escolhi `DataScienceIdentity`

# ![challenge_panda](https://media.giphy.com/media/K9z3im98oo9Ve/giphy.gif)

# ## Alterando o dataset original

# In[96]:


# Primeiro vamos copiar o dataset original para outra variável, para não dar pau no resto da aula :P


# In[97]:


# Agora vamos ver os valores da coluna LearningDataScience


# In[46]:


def replace_value(row):
    if row == "Yes, I'm focused on learning mostly data science skills":
        return "yes"
    elif row == "Yes, but data science is a small part of what I'm focused on learning":
        return "so so"
    elif row == "No, I am not focused on learning data science skills":
        return "no"


# In[98]:


# Agora vamos aplicar a função replace_value a todas as linhas do LearningDataScience. 
# Crie uma nova coluna chamada LearningDataScienceSimple


# Agora podemos ver os novos valores desses campos

# In[99]:


# Agora vamos ver os valores da coluna LearningDataScienceSimple


# In[49]:


# Qual o shape do dataset atual?


# E agora que a outra coluna muito complexa não serve mais, podemos descartá-la

# In[50]:


# Descarte a coluna LearningDataScience


# In[100]:


# Qual o shape do dataset atual?


# Pela quantidade de linhas no dataset percebemos que o `value_counts()` não retorna os valores nulos - e nós temos MUITOS valores nulos nessa coluna. Podemos utilizar um método do pandas para trocar os NAs por uma categoria nossa.

# In[101]:


# Substitua os nulos pela frase "did not answer the question"


# In[102]:


# Agora vamos ver os valores da coluna LearningDataScienceSimple


# Outra forma de alterar o dataset é utilizando funções _in place_. Para isso utilizaremos o `lambda`

# Por exemplo: E se eu quiser atualizar a idade dos participantes? O dataset foi coletado em 2017 e já estamos em 2018

# In[103]:


# Ao invés de usar uma função que soma +1, vamos escrever a nossa com o lambda


# Vamos ver se funcionou? Vamos dar uma olhada nos primeiros 5 registros, com a coluna 'Age' e a 'NewAge' lado a lado

# Ficou mais claro a proporção de pessoas que não responderam agora o/

# ### Desafio 3

# Separar os datasets pelas pessoas que os responderam. Para isso você vai ter que carregar o dataset `schema.csv`.
# 
# Como uns datasets ficariam muito pequenos, sugiro que você utilize os seus conhecimentos recém adquiridos e crie 4 datasets distintos. Além de criá-los você deve salvá-los como `.csv`. Vamos usar esses datasets na próxima aula.

# ![panda_playground](https://media.giphy.com/media/ieaUdBJJC19uw/giphy.gif)
