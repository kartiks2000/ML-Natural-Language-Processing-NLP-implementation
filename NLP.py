# NATURAL LANGUAGE PROCESSING (NLP)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# We are importing a ".tsv" dataset, because it does not uses "," as separators as it could create problems in NLP algorithms instead it uses TAB "/t" as sepeartor
# We are also neglecting Quotes (") to avoid problems in NLP algorithms.
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts

# We only want the relevent and useful words, and will neglect the non-useful words like "on","the","a",....etc
# We will get rid of punchuations.
# We will also apply steming on words, taking only the root word of the actual word Ex. "Loved" or "Loving" -> "love", so here will only consider love not loved.
# Regular expression liberary
import re
# nltk liberary contains all relevant words, these words are called "stopwords", hence we need to download it.
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# "corpus" list will contain all our cleaned review text.
corpus = []
for i in range(0, 1000):
    # Taking onnly the alphates in each review
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    # converting each review into lowercase.
    review = review.lower()
    # splitting review into a list
    review = review.split()
    # Importing Word Stemming class and then using it in the root.
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # Combining the list back into a sentence.
    review = ' '.join(review)
    corpus.append(review)
    
    
# Creating a bag of words model    
    
# Now the texts is cleaned, hence we will be creating our bag of words model (tokenisation process).
# Bag of words model simply splits different riviews into different words (Only the relevant word).
# there would be a column for each word and number of times it appeared in a review, basically a Matrix will be formed.
# We will also remove the words with very less frequency of appearence.
from sklearn.feature_extraction.text import CountVectorizer
# Only taking the first 1500 higest frequency words present in the review.
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)



# Training the Naive Bayes model on the Training set
# We need to predict a binary result (0 or 1).
# Hence, we need to apply a classification model to know if Review is a negative or positive review. 
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)



# Predicting the Test set results
y_pred = classifier.predict(X_test)



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


