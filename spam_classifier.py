# Import Pandas library
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score,confusion_matrix
df = pd.read_csv('spam.csv', 
                   sep='\t', 
                   names=['label','sms_message'])

# Conversion
df['label'] = df.label.map({'ham':0, 'spam':1})

# Print dataset shape
print (df.shape)

x=df['sms_message']
y= df['label']


x_train,x_test,y_train,y_test = train_test_split(df['sms_message'],df['label'])

count_vector = CountVectorizer(stop_words = 'english')

trainning = count_vector.fit_transform(x_train)
testing = count_vector.transform(x_test)

model = MultinomialNB()

model.fit(trainning,y_train)

prediction = model.predict(testing)

confusion_matrix(y_test,prediction)
print("Accuracy : ",(accuracy_score(y_test,prediction)*100))
print("Precision : ",(precision_score(y_test,prediction)*100))
