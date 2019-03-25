import numpy as np 
import pandas as pd 

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers


# import training data
df = pd.read_csv('train.csv')
col = ['title', 'Category']
df = df[col]
df1 = df[0:1000]
df2 = df[100000:102000]
df3 = df[200000:202000]
df4 = df[300000:302000]
df5 = df[400000:401000]
df6 = df[500000:501000]
df7 = df[600000:603000]
frames = [df1,df2,df3,df4,df5,df6,df7]
df = pd.concat(frames)

# splitting 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(df['title'], df['Category'])

# label encode the target variable
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(df['title'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)

# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(df['title'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(df['title'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x)



# fit model
from sklearn import svm
clf = svm.SVC(gamma=0.01, C = 100)
clf.fit(xtrain_tfidf_ngram_chars,train_y)

print("Done")

print(clf.predict(xvalid_tfidf_ngram_chars))

print(clf.score(xvalid_tfidf_ngram_chars, valid_y))


######prediction#####


# import test data
df_test = pd.read_csv('test.csv')
col = ['title', 'itemid']
df_test = df_test[col]
df_test = df_test[0:5000]

#feature extraction
# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(df_test['title'])
test_data =  tfidf_vect.transform(df_test['title'])


# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(df_test['title'])
xtest_tfidf_ngram =  tfidf_vect_ngram.transform(df_test['title'])

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(df_test['title'])
xtest_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(df_test['title']) 




# predict test data
print(clf.predict(xtest_tfidf_ngram_chars))

result = clf.predict(xtest_tfidf_ngram_chars)

# matching result with itemid
df_result = pd.DataFrame({'Category':result})

df_itemid = df_test['itemid']

df_submission = pd.concat([df_itemid, df_result], axis=1, sort=False)

print(df_submission)

# Exporting to csv
df_submission.to_csv('submission', index=False)


