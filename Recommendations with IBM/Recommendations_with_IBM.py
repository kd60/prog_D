#!/usr/bin/env python
# coding: utf-8

# # Recommendations with IBM
# 
#  describes IBM's use of Recommendation X and related recommendations of the International Telegraph and Telephone Consultative Committee. After reviewing the development history of X and some of the motivations for using it, the paper gives an overview of packet-switched data networks. The reader is then given a brief technical description of Recommendation X and some other recommendations used in conjunction with X. The architectural relationships between X and IBM's Systems Network Architecture (SNA) are described for packet-switched X connections between SNA and non-SNA nodes. Specific elements of Recommendation X used in SNA nodes are defined. After several IBM products that support X and some of the related recommendations are described, IBM's equipment for testing the X interface is discussed.
# 
# 
# ## Table of Contents
# 
# I. [Exploratory Data Analysis](#Exploratory-Data-Analysis)<br>
# II. [Rank Based Recommendations](#Rank)<br>
# III. [User-User Based Collaborative Filtering](#User-User)<br>
# IV. [Matrix Factorization](#Matrix-Fact)<br>
# V. [Extras & Concluding](#conclusions)
# 
# 

# In[1]:


# pip install matplotlib


# In[2]:


# import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import project_tests as t
import pickle
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('data/user-item-interactions.csv')
df_articles_community = pd.read_csv('data/articles_community.csv')
del df['Unnamed: 0']
del df_articles_community['Unnamed: 0']


# In[3]:


# Show df to To see dataframe of user item interactions
df.head(5)


# In[4]:


# Show df_articles_community to see the dataframe of articles community
df_articles_community.head()


# ### <a class="anchor" id="Exploratory-Data-Analysis">Part I :Analysis Data and Exploreing It </a>
# 

# In[5]:


#Show The Shape Of Data df
df.shape


# In[ ]:





# In[6]:


# get median of article_id grouping by Email
df.groupby('email')['article_id'].count().median()


# In[7]:


# get MAX of article_id grouping by Email
df.groupby('email')['article_id'].count().max()


# In[8]:


# median and maximum number of user_article interactions

median_user_article = df.groupby('email')['article_id'].count().median()
maxviewed = df.groupby('email')['article_id'].count().max()


# In[9]:


interact_form_user = df.groupby('email')['article_id'].count()
interact_form_user


# In[10]:


# Get the summary of user_interacts
interact_form_user.describe()


# In[11]:


# plot graph to view Distribution of User-Article Interactions
plt.figure(figsize=(8,6))
interact_form_user.plot(kind='hist')
plt.title('Distribution of User-Article Interactions') 
plt.xlabel('Number of User-Article Interactions');
plt.ylabel('The Frequency');


# `2.` Explore and remove duplicate articles from the **df_articles_community** dataframe.  

# In[12]:


# Show df_articles_community to see the dataframe of articles community
df_articles_community.head(5)


# In[13]:


#Show The Shape Of Data df_articles_community
df_articles_community.shape


# In[14]:


# get duplicate articles
df_articles_community.article_id.duplicated().sum()


# In[15]:


article_ids = df_articles_community['article_id']


# In[16]:


print(article_ids)


# In[17]:


df_articles_community[article_ids.isin(article_ids[article_ids.duplicated()])]


# In[18]:


# Remove any rows that have the same article_id 
df_articles_community.drop_duplicates(subset=['article_id'], keep='first', inplace=True)


# In[19]:


df_articles_community.iloc[100]


# ### `3.`  Find unique articles ,unique users ,user-article interactions in the dataset.

# In[20]:


print(df.shape)
print(df_articles_community.shape)


# In[21]:


# The number of unique articles 
df.article_id.nunique()


# In[22]:


# The number of unique articles 
df_articles_community.article_id.nunique()


# In[23]:


# The number of unique users
df.email.nunique()


# In[24]:


# The number of user-article interactions
df.shape[0]


# In[25]:


unique_articles = df.article_id.nunique() 
total_articles =  df_articles_community.article_id.nunique() 
unique_users = df.email.nunique()
user_article_interactions = df.shape[0] 


# In[26]:


print("The number of unique articles =",unique_articles)
print("The number of unique articles on the df_articles_community =",total_articles)
print("The number of unique users =",unique_users)
print("The number of user-article shape =",user_article_interactions)


# ### `4.`  Find the most viewed article_id.

# In[27]:


# The max of viewed  article_id
df.article_id.value_counts().max()


# In[28]:


most_viewed_article_id = str(df.article_id.value_counts().index[0]) 
max_views = df.article_id.value_counts().iloc[0]


# In[29]:


## No need to change the code here - this will be helpful for later parts of the notebook
# Run this cell to map the user email to a user_id column and remove the email column

def email_mapper():
    coded_dict = dict()
    cter = 1
    email_encoded = []
    
    for val in df['email']:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter+=1
        
        email_encoded.append(coded_dict[val])
    return email_encoded

email_encoded = email_mapper()
del df['email']
df['user_id'] = email_encoded

# show header
df.head()


# In[30]:


## If you stored all your results in the variable names above, 
## you shouldn't need to change anything in this cell

sol_1_dict = {
    '`50% of individuals have _____ or fewer interactions.`': median_user_article,
    '`The total number of user-article interactions in the dataset is ______.`': user_article_interactions,
    '`The maximum number of user-article interactions by any 1 user is ______.`': maxviewed,
    '`The most viewed article in the dataset was viewed _____ times.`': max_views,
    '`The article_id of the most viewed article is ______.`': most_viewed_article_id,
    '`The number of unique articles that have at least 1 rating ______.`': unique_articles,
    '`The number of unique users in the dataset is ______`': unique_users,
    '`The number of unique articles on the IBM platform`': total_articles
}

# Test your dictionary against the solution
t.sol_1_test(sol_1_dict)


# ### <a class="anchor" id="Rank">Part II: Rank-Based Recommendations</a>
# 
# Traditional recommendation algorithms use the predicted rating scores to represent the degree of user preference, called rating-based recommendation methods. Recently, ranking-based algorithms have been proposed and widely used, which use ranking to present the 
# 
# ### `1.` Get The **n** top articles ordered 

# In[31]:


def Find_Max_articles(n, df=df):
  
    get_max_articles = df['title'].value_counts().index.tolist()[:n]
    get_max_articles = [str(i) for i in get_max_articles]
    
    return get_max_articles 

def get_top_article_ids(n, df=df):
    
    get_max_articles = df['article_id'].value_counts().index.tolist()[:n]
    get_max_articles = [str(i) for i in get_max_articles]

    return get_max_articles 


# In[32]:


print("The Top Ten Articles = \n",Find_Max_articles(10))
print("************************************************************************************************************************")
print("The Top Ten Articles IDs = \n",get_top_article_ids(10))


# In[33]:


# Test your function by returning the top 5, 10, and 20 articles
top_5 = Find_Max_articles(5)
top_10 = Find_Max_articles(10)
top_20 = Find_Max_articles(20)

# Test each of your three lists from above
t.sol_2_test(Find_Max_articles)


# ### <a class="anchor" id="User-User">Part III: User-User Based Collaborative Filtering</a>

# In[34]:



def create_user_item_matrix(df):

    df_count = df.groupby(['user_id', 'article_id']).count().reset_index()
    user_item = df_count.pivot_table(values='title', index='user_id', columns='article_id') 
    user_item.replace(np.nan, 0, inplace=True) 
    user_item=user_item.applymap(lambda x: 1 if x > 0 else x) 

    return user_item

user_item = create_user_item_matrix(df)


# In[35]:


print("the user_item matrix = \n",user_item)


# In[36]:


## Tests: You should just need to run this cell.  Don't change the code.
assert user_item.shape[0] == 5149, "Oops!  The number of users in the user-article matrix doesn't look right."
assert user_item.shape[1] == 714, "Oops!  The number of articles in the user-article matrix doesn't look right."
assert user_item.sum(axis=1)[1] == 36, "Oops!  The number of articles seen by user 1 doesn't look right."
print("You have passed our quick tests!  Please proceed!")


# ### `2.` Get the  users in order from most to least similar

# In[37]:


def find_similar_users(user_id, user_item=user_item):

    users_dot_product = user_item.dot(np.transpose(user_item))
 
    similar_users = users_dot_product[user_id].sort_values(ascending = False)
   
    most_similar_users = similar_users.index.tolist()
    
    most_similar_users.remove(user_id)
       
    return most_similar_users


# In[38]:


print("The 6 most similar users_to_user 5 are: {}".format(find_similar_users(5)[:6]))
print("The 8 most similar users_to_user 3900 are: {}".format(find_similar_users(3900)[:8]))
print("The 12 most similar users_to_user 50 are: {}".format(find_similar_users(50)[:12]))


# ### `3.` Get the articles we would recommend to each user. 

# In[39]:


def get_names_of_article(article_ids, df=df):

    names_of_article = []

    for idx in article_ids:
        names_of_article.append(df[df['article_id']==float(idx)].max()['title'])
    
    return names_of_article 

def get_user_articles(user_id, user_item=user_item):

    article_ids = user_item.loc[user_id][user_item.loc[user_id] == 1].index.astype('str')
    
    names_of_article = []

    for idx in article_ids:
        names_of_article.append(df[df['article_id']==float(idx)].max()['title'])
    return article_ids, names_of_article


def user_user_recs(user_id, m = 10):
   
    recs = np.array([]) 
    
    user_articles_seen = get_user_articles(user_id)[0] 
    closest_users = find_similar_users(user_id)
    
    for others in closest_users:
        
        others_articles_seen = get_user_articles(others)[0]
        new_recs = np.setdiff1d(others_articles_seen, user_articles_seen, assume_unique=True) 
        recs = np.unique(np.concatenate([new_recs, recs], axis = 0))

        if len(recs) > m-1:
            break
            
    recs = recs[:m]
    recs.tolist()
    
    return recs 


# In[40]:


# Check Results
get_names_of_article(user_user_recs(3, 15)) 


# In[41]:


# Test your functions here - No need to change this code - just run this cell
assert set(get_names_of_article(['1024.0', '1176.0', '1305.0', '1314.0', '1422.0', '1427.0'])) == set(['using deep learning to reconstruct high-resolution audio', 'build a python app on the streaming analytics service', 'gosales transactions for naive bayes model', 'healthcare python streaming application demo', 'use r dataframes & ibm watson natural language understanding', 'use xgboost, scikit-learn & ibm watson machine learning apis']), "Oops! Your the get_names_of_article function doesn't work quite how we expect."
assert set(get_names_of_article(['1320.0', '232.0', '844.0'])) == set(['housing (2015): united states demographic measures','self-service data preparation with ibm data refinery','use the cloudant-spark connector in python notebook']), "Oops! Your the get_names_of_article function doesn't work quite how we expect."
assert set(get_user_articles(20)[0]) == set(['1320.0', '232.0', '844.0'])
assert set(get_user_articles(20)[1]) == set(['housing (2015): united states demographic measures', 'self-service data preparation with ibm data refinery','use the cloudant-spark connector in python notebook'])
assert set(get_user_articles(2)[0]) == set(['1024.0', '1176.0', '1305.0', '1314.0', '1422.0', '1427.0'])
assert set(get_user_articles(2)[1]) == set(['using deep learning to reconstruct high-resolution audio', 'build a python app on the streaming analytics service', 'gosales transactions for naive bayes model', 'healthcare python streaming application demo', 'use r dataframes & ibm watson natural language understanding', 'use xgboost, scikit-learn & ibm watson machine learning apis'])
print("If this is all you see, you passed all of our tests!  Nice job!")


# ### `4.` Now we are going to improve the consistency of the **user_user_recs** function 

# In[42]:


def get_top_sorted_users(user_id, df=df, user_item=user_item):

    neighbors_df = pd.DataFrame(columns=['neighbor_id', 'similarity'])
    neighbors_df['neighbor_id'] = user_item.index-1
    users_dot_product = user_item.dot(np.transpose(user_item))
    neighbors_df['similarity'] = users_dot_product[user_id]
    interacts_df = df.user_id.value_counts().rename_axis('neighbor_id').reset_index(name='num_interactions')
    neighbors_df = pd.merge(neighbors_df, interacts_df, on='neighbor_id', how='outer')
    neighbors_df = neighbors_df.sort_values(by=['similarity', 'num_interactions'], ascending = False)
    neighbors_df = neighbors_df.reset_index(drop=True)
    neighbors_df = neighbors_df[neighbors_df.neighbor_id != user_id]
    
    return neighbors_df


def user_user_recs_part2(user_id, m=10):
    
    recs = np.array([])
    
    user_articles_ids_seen, user_articles_names_seen = get_user_articles(user_id, user_item) 
    closest_neighs = get_top_sorted_users(user_id, df, user_item).neighbor_id.tolist()
    
    for neighs in closest_neighs:
        
        neigh_articles_ids_seen, neigh_articles_names_seen = get_user_articles(neighs, user_item) 
        new_recs = np.setdiff1d(neigh_articles_ids_seen, user_articles_ids_seen, assume_unique=True)
        recs = np.unique(np.concatenate([new_recs, recs], axis = 0))

        if len(recs) > m-1:
            break
            
    recs = recs[:m]
    recs = recs.tolist()
    
    rec_names = get_names_of_article(recs, df=df)
    
    return recs, rec_names


# In[43]:


rec_ids, rec_names = user_user_recs_part2(50, 5)
print("The top 5 recommendations for user 50 are the following article ids:")
print(rec_ids)
print()
print("The top 5 recommendations for user 50 are the following article names:")
print(rec_names)


# `5.` The functions from above are now used to correctly fill in the solutions to the dictionary below.

# In[44]:


user1_most_sim = get_top_sorted_users(1).iloc[0].neighbor_id
user131_10th_sim = get_top_sorted_users(131).iloc[9].neighbor_id


# In[45]:



sol_5_dict = {
    'The user that is most similar to user 1.': user1_most_sim, 
    'The user that is the 10th most similar to user 131': user131_10th_sim,
}

t.sol_5_test(sol_5_dict)


# ### `6.`  provide for the a new user .  

# In[46]:


new_user = '0.0'
new_user_recs = get_top_article_ids(10, df)


# In[47]:


assert set(new_user_recs) == set(['1314.0','1429.0','1293.0','1427.0','1162.0','1364.0','1304.0','1170.0','1431.0','1330.0']), "Oops!  It makes sense that in this case we would want to recommend the most popular articles, because we don't know anything about these users."

print("That's right!  Nice job!")


# ### <a class="anchor" id="Matrix-Fact">Part IV: Matrix Factorization</a>
# 
# We will use matrix factorization to make article recommendations to the users on the IBM Watson Studio platform.
# 
# Understand the pitfalls of traditional methods and pitfalls of measuring the influence of recommendation engines under traditional regression and classification techniques.
# Create recommendation engines using matrix factorization and FunkSVD
# 
# 

# In[48]:


# Load user_item_matrix
U_I_Matrix = pd.read_pickle('user_item_matrix.p')


# In[49]:


# See the user_item_matrix
U_I_Matrix.head()


# In[50]:


# Perform SVD on the User-Item Matrix Here

u, s, vt = np.linalg.svd(U_I_Matrix)


# In[51]:


s.shape, u.shape, vt.shape


# Doing SVD because no missing values.

# In[52]:


number_of_late_feat = np.arange(10,700+10,20)
sum_of_errs = []

for k in number_of_late_feat:
    
    s_new, u_new, vt_new = np.diag(s[:k]), u[:, :k], vt[:k, :]
    
    
    user_item_est = np.around(np.dot(np.dot(u_new, s_new), vt_new))
    
   
    diffs = np.subtract(U_I_Matrix, user_item_est)
    
    
    err = np.sum(np.sum(np.abs(diffs)))
    sum_of_errs.append(err)
    
    
plt.plot(number_of_late_feat, 1 - np.array(sum_of_errs)/df.shape[0]);
plt.xlabel('The Number of Latent Features');
plt.ylabel('Accuracy');
plt.title('Accuracy vs. Number of Latent Features');


# In[53]:


df_train = df.head(40000)
df_test = df.tail(5993)

def user_item_train_test(df_train, df_test):
    
    user_item_train = create_user_item_matrix(df_train)
    user_item_test = create_user_item_matrix(df_test)
    
    test_idx = user_item_test.index
    test_arts = user_item_test.columns
    
    return user_item_train, user_item_test, test_idx, test_arts

user_item_train, user_item_test, test_idx, test_arts = user_item_train_test(df_train, df_test)


# In[54]:


test_idx


# In[55]:


train_idx = user_item_train.index
train_idx 


# In[56]:


test_idx.difference(train_idx) 


# In[57]:


test_arts 


# In[58]:


train_arts = user_item_train.columns
train_arts


# In[59]:


test_arts.difference(train_arts) 


# In[60]:



a = 662 
b = 574 
c = 20 
d = 0 


sol_4_dict = {
    'How many users can we make predictions for in the test set?': c, 
    'How many users in the test set are we not able to make predictions for because of the cold start problem?': a, 
    'How many movies can we make predictions for in the test set?': b,
    'How many movies in the test set are we not able to make predictions for because of the cold start problem?': d
}

t.sol_4_test(sol_4_dict)


# In[61]:


# fit SVD on the user_item_train matrix
train_u, s_train, vt_train = np.linalg.svd(user_item_train)


# In[62]:



s_train.shape, train_u.shape, vt_train.shape


# In[63]:


number_of_late_feat = np.arange(10,700+10,20)
sum_of_errs_train = []
sum_of_errs_test = []

idx_row = user_item_train.index.isin(test_idx)
idx_col = user_item_train.columns.isin(test_arts)

test_u = train_u[idx_row, :]
vt_test = vt_train[:, idx_col]

users_can_predict = np.intersect1d(list(user_item_train.index),list(user_item_test.index))
    
for k in number_of_late_feat:
  
    s_train_new, train_u_new, vt_train_new = np.diag(s_train[:k]), train_u[:, :k], vt_train[:k, :]
    test_u_new, vt_test_new = test_u[:, :k], vt_test[:k, :]
  
    user_item_train_preds = np.around(np.dot(np.dot(train_u_new, s_train_new), vt_train_new))
    user_item_test_preds = np.around(np.dot(np.dot(test_u_new, s_train_new), vt_test_new))
  
    diffs_train = np.subtract(user_item_train, user_item_train_preds)
    diffs_test = np.subtract(user_item_test.loc[users_can_predict,:], user_item_test_preds)
    
    err_train = np.sum(np.sum(np.abs(diffs_train)))
    err_test = np.sum(np.sum(np.abs(diffs_test)))
    
    sum_of_errs_train.append(err_train)
    sum_of_errs_test.append(err_test)


# In[64]:


# Plot the accuracy
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Number of Latent Features')
ax1.set_ylabel('Accuracy for Training', color=color)
ax1.plot(number_of_late_feat, 1 - np.array(sum_of_errs_train)/df.shape[0], color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_title('Accuracy vs. Number of Latent Features')

ax2 = ax1.twinx() 

color = 'tab:green'
ax2.set_ylabel('Test Accuracy ', color=color)  
ax2.plot(number_of_late_feat, 1 - np.array(sum_of_errs_test)/df.shape[0], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  
plt.show()


# <a id='conclusions'></a>
# 
# 
# ## Conclusion
# 
# 
# Finally, we used a machine learning approach to building recommendations. I used the user-item interactions to build out a matrix decomposition. I used this decomposition to make predictations on new articles an individual might interact with, which turned out not to be that great).
# 

# In[ ]:




