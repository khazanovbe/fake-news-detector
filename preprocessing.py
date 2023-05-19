import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import itertools
import pickle
import math

NUMBER_OF_ROWS = 35000

def plotConfusionMatrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Reading data  and prepare dataframe
true  = pd.read_csv("./data/True.csv")
fake  = pd.read_csv("./data/Fake.csv")

true['label'] = 1
fake['label'] = 0

frames = [true.loc[:NUMBER_OF_ROWS][:], fake.loc[:NUMBER_OF_ROWS][:]]
df = pd.concat(frames)

X = df.drop('label', axis=1) 
y = df['label']

# Delete missing data
df = df.dropna()
df2 = df.copy()
df2.reset_index(inplace=True)

# Tokenization

nltk.download('stopwords')
ps = PorterStemmer()
corpus = []
for i in range(0, len(df2)):
    review = re.sub('[^a-zA-Z]', ' ', df2['text'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

#TF-IDF vectorizing

tfidf_v = TfidfVectorizer(max_features=len(corpus), ngram_range=(1,3))
X = tfidf_v.fit_transform(corpus).toarray()
y = df2['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Model training

model = LogisticRegression(random_state=0).fit(X_train, y_train)
# model = PassiveAggressiveClassifier(max_iter=1000).fit(X_train, y_train)
prediction = model.predict(X_test)
print("Accuracy: ",metrics.accuracy_score(y_test, prediction))
print("RMSE: ",math.sqrt(metrics.mean_squared_error(y_test,prediction)))

cm = metrics.confusion_matrix(y_test, prediction)
plotConfusionMatrix(cm, classes=['FAKE', 'REAL'])

pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(tfidf_v, open('tfidfvect.pkl', 'wb'))
