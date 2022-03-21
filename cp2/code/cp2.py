# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os

# <codecell>

path_p = u"txt_sentoken/pos"
path_n = u"txt_sentoken/neg"

# <codecell>

ds_p = os.listdir(path_p)
ds_n = os.listdir(path_n)

# <codecell>

texts_p = []
texts_n = []

# <codecell>

def convert_file_to_text(name):
    text = u""
    f = open(name)
    for i in f.readlines():
        text += i.decode(u"utf8")
    return text
import os

# <codecell>

path_p = u"txt_sentoken/pos"
path_n = u"txt_sentoken/neg"

# <codecell>

ds_p = os.listdir(path_p)
ds_n = os.listdir(path_n)

# <codecell>

texts_p = []
texts_n = []

# <codecell>

def convert_file_to_text(name):
    text = u""
    f = open(name)
    for i in f.readlines():
        text += i.decode(u"utf8")
    return text
        

# <codecell>

a = convert_file_to_text(os.path.join(path_p,ds_p[0]))

# <codecell>

for i in ds_p:
    texts_p.append(convert_file_to_text(os.path.join(path_p,i)))

# <codecell>

for i in ds_n:
    texts_n.append(convert_file_to_text(os.path.join(path_n,i)))

# <codecell>

len(texts_p)

# <codecell>

len(texts_n)

# <codecell>

from sklearn.feature_extraction.text import CountVectorizer 

# <codecell>

vectorizer = CountVectorizer()
mt = vectorizer.fit_transform(texts_p + texts_n)
mta = mt.toarray()

# <codecell>

#matriz de características
mt.shape

# <codecell>

mt.nnz *100.0 / (mt.shape[0] * mt.shape[1])

# <codecell>

y = [1]*1000 + [0]*1000

# <codecell>

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# <codecell>

naive_bayes = GaussianNB()
naive_bayes.fit(mta,y)
naive_bayes.score(mta,y)

# <codecell>

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(mta, y, train_size=0.60)

# <codecell>

print(X_train.shape)
print(X_test.shape)

# <codecell>

naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)
naive_bayes.score(X_test, y_test)

# <codecell>

d_tree = DecisionTreeClassifier()
d_tree.fit(X_train, y_train)
d_tree.score(X_test, y_test)

# <codecell>

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn.score(X_test, y_test)

# <codecell>

svm = SVC(kernel="linear")
svm.fit(X_train, y_train)
svm.score(X_test, y_test)

# <codecell>

svm = SVC(kernel="rbf")
svm.fit(X_train, y_train)
svm.score(X_test, y_test)

# <codecell>

from scipy.stats import ttest_ind

# <codecell>

#se hacen 30 corridas
rs = []
for i in range(30):
    X_train, X_test, y_train, y_test = train_test_split(mta, y, train_size=0.60)
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    rs.append(knn.score(X_test, y_test))

# <codecell>

rs

# <codecell>


