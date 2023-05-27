#!/usr/bin/env python
# coding: utf-8

# ## Exploratory Data Analysis : World Happiness Report

# ## Importing required Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Loading Dataset

# In[2]:


dataset = pd.read_csv("WHR.csv")


# ## Describing and Understanding the Dataset

# In[3]:


dataset.head()


# In[4]:


dataset.shape


# In[5]:


dataset.columns


# In[6]:


dataset.dtypes


# In[7]:


dataset.info()


# In[8]:


dataset.describe()


# In[9]:


dataset.isnull().sum()


# In[10]:


dataset['Country name'].unique()


# In[11]:


dataset['Country name'].unique().size


# ### Observation 1:
# 
# Data set having
# 
# 1.) 149 countries and 20 fields
# 
# 2.) No null values
# 
# 3.) No duplicate rows

# ## Creating New Dataset with the Required Fields

# In[12]:


df = dataset[['Country name', 'Regional indicator', 'Ladder score', 'Logged GDP per capita', 'Social support', 
              'Healthy life expectancy', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']]


# ## Finding Top 5 and Bottom 5 Happiest Countries and their Region

# In[13]:


df.head(5)


# In[14]:


df.tail(5)


# In[15]:


Country_LS = dict(type = 'choropleth',
           locations = df['Country name'],
           locationmode = 'country names',
           z = df['Ladder score'],
           text = df['Country name'],
           colorbar = {'title':'Ladder Score'}
           )


# In[16]:


import plotly.graph_objects as go

map_fig = go.Figure(data = Country_LS)
map_fig.update_geos(projection_type="natural earth")
map_fig.update_layout(height=300, margin={"r":0,"t":0,"l":0,"b":0})


# ### Observation 2:
# 
# 1.) Top 5 happiest countries and their regions:
# 
# - Finland - Western Europe
# - Denmark - Western Europe
# - Switzerland - Western Europe
# - Iceland - Western Europe
# - Netherlands - Western Europe
# 
# 2.) Bottom 5 happiest countries and their regions:
# 
# - Lesotho - Sub-Saharan Africa 	
# - Botswana - Sub-Saharan Africa 	
# - Rwanda - Sub-Saharan Africa 	
# - Zimbabwe -Sub-Saharan Africa 	
# - Afghanistan - South Asia

# ## Finding Most and Least Happiest Countries with respect to Regions

# In[17]:


sns.histplot(x='Ladder score',data=df,kde=True)
plt.show()


# In[18]:


sns.violinplot(y='Ladder score',data=df)
plt.show()


# In[19]:


df.head(5)


# In[20]:


df.tail(5)


# In[21]:


df['Regional indicator'].value_counts()


# In[22]:


sns.countplot(y='Regional indicator',data=df)
plt.title('Numbers of Countries in Each Region')
plt.show()


# In[23]:


reg_top = df.groupby('Regional indicator').first().reset_index()
reg_top


# In[24]:


reg_top_bar = sns.barplot(x='Ladder score',y='Country name',hue='Regional indicator',data=reg_top,orient='h')
sns.move_legend(reg_top_bar,"upper left",bbox_to_anchor=(1,1))
plt.title('Most Happiest Country in Each Region')
plt.show()


# In[25]:


reg_bottom = df.groupby('Regional indicator').last().reset_index()
reg_bottom


# In[26]:


reg_bottom_bar = sns.barplot(x='Ladder score',y='Country name',hue='Regional indicator',data=reg_bottom,orient='h')
sns.move_legend(reg_bottom_bar,'upper left',bbox_to_anchor=(1,1))
plt.title('Least Happiest Country in Each Region')
plt.show()


# In[27]:


country_above_avg = pd.DataFrame(df[['Country name','Regional indicator']][df['Ladder score']>=df['Ladder score'].mean()])
country_above_avg


# In[28]:


sns.countplot(y='Regional indicator',data=country_above_avg)
plt.title('Numbers of Countries in Each Region above Average')
plt.show()


# In[29]:


country_below_avg = pd.DataFrame(df[['Country name','Regional indicator']][df['Ladder score']<=df['Ladder score'].mean()])
country_below_avg


# In[30]:


sns.countplot(y='Regional indicator',data=country_below_avg)
plt.title('Numbers of Countries in Each Region below Average')
plt.show()


# ### Observation 3: 
# 
# 1.) Number of countries in each region:
#     
# - Central and Eastern Europe: 17
# - Commonwealth of Independent States: 12
# - East Asia: 6
# - Latin America and Caribbean: 20
# - Middle East and North Africa: 17
# - North America and ANZ: 4
# - South Asia: 7
# - Southeast Asia: 9   
# - Sub-Saharan Africa: 36
# - Western Europe: 21
# 
# 2.) Most and least happiest country in each region:
#     
# - Central and Eastern Europe: Czech Republic | North Macedonia
# - Commonwealth of Independent States: Uzbekistan | Ukraine
# - East Asia: Taiwan Province of China | China
# - Latin America and Caribbean: Costa Rica | Haiti
# - Middle East and North Africa: Israel | Yemen
# - North America and ANZ: New Zealand | United States
# - South Asia: Nepal | Afghanistan
# - Southeast Asia: Singapore | Myanmar
# - Sub-Saharan Africa: Mauritius | Zimbabwe
# - Western Europe: Finland | North Cyprus
# 
# 3.) Number of countries having above and below average of ladder score in each region
# 
# - Central and Eastern Europe: 14 | 3
# - Commonwealth of Independent States: 5 | 7
# - East Asia: 4 | 2
# - Latin America and Caribbean: 18 | 2
# - Middle East and North Africa: 5 | 12
# - North America and ANZ: 4 | 0
# - South Asia: 0 | 7
# - Southeast Asia: 3 | 6
# - Sub-Saharan Africa: 1 | 35
# - Western Europe: 21 | 0

# ## Finding Correlation between Fields and Ladder Score

# In[31]:


df.columns


# In[32]:


df.corrwith(df['Ladder score'])


# In[33]:


GDP_LS = sns.scatterplot(x='Logged GDP per capita',y='Ladder score',data=df,hue='Regional indicator')
sns.move_legend(GDP_LS,'upper left',bbox_to_anchor=(1,1))
plt.title('Correlation between GDP per capita and Ladder score')
plt.show()


# In[34]:


SS_LS = sns.scatterplot(x='Social support',y='Ladder score',data=df,hue='Regional indicator')
sns.move_legend(SS_LS,'upper left',bbox_to_anchor=(1,1))
plt.title('Correlation between Social support and Ladder score')
plt.show()


# In[35]:


HLE_LS = sns.scatterplot(x='Healthy life expectancy',y='Ladder score',data=df,hue='Regional indicator')
sns.move_legend(HLE_LS,'upper left',bbox_to_anchor=(1,1))
plt.title('Correlation between Healthy life expectancy and Ladder score')
plt.show()


# In[36]:


FLC_LS = sns.scatterplot(x='Freedom to make life choices',y='Ladder score',data=df,hue='Regional indicator')
sns.move_legend(FLC_LS,'upper left',bbox_to_anchor=(1,1))
plt.title('Correlation between Freedom to make life choices and Ladder Score')
plt.show()


# In[37]:


G_LS = sns.scatterplot(x='Generosity',y='Ladder score',data=df,hue='Regional indicator')
sns.move_legend(G_LS,'upper left',bbox_to_anchor=(1,1))
plt.title('Correlation between Generosity and Ladder Score')
plt.show()


# In[38]:


PC_LS = sns.scatterplot(x='Perceptions of corruption',y='Ladder score',data=df,hue='Regional indicator')
sns.move_legend(PC_LS,'upper left',bbox_to_anchor=(1,1))
plt.title('Correlation between Perceptions of corruption and Ladder score')
plt.show()


# In[39]:


sns.pairplot(vars=['Ladder score','Logged GDP per capita','Social support','Healthy life expectancy',
                   'Freedom to make life choices','Generosity','Perceptions of corruption'],hue='Regional indicator',data=df)
plt.show()


# In[40]:


df_cor = df[['Ladder score','Logged GDP per capita','Social support','Healthy life expectancy',
                   'Freedom to make life choices','Generosity','Perceptions of corruption']].corr()
df_cor


# In[41]:


sns.heatmap(df_cor,annot=True)
plt.show()


# ### Observation 4:
# 
# 1.) Factors that makes positive impact on the happniess of a country:
# 
# - GDP per capita
# - Social support
# - Healthy life expectancy
# - Freedom to make life choices
# 
# 2.) Factors that makes negative impact on the happniess of a country:
# 
# - Perceptions of corruption

# In[ ]:




