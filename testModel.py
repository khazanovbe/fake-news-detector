import pickle
import re
import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


true  = pd.read_csv("./data/True.csv")
fake  = pd.read_csv("./data/Fake.csv")

true['label'] = 1
fake['label'] = 0

ps = PorterStemmer()
review = re.sub('[^a-zA-Z]', ' ', fake['text'][13070])
review = review.lower()
review = review.split() 
review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
review = ' '.join(review)

model = pickle.load(open('model.pkl', 'rb'))
vect = pickle.load(open('tfidfvect.pkl', 'rb'))
valPkl = vect.transform([review]).toarray()
prediction = model.predict(valPkl)
print(prediction)

