import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


train_data = pd.read_csv('train_data.txt', sep=':::', names=['ID', 'Title', 'Genre', 'Description'], engine='python')
test_data = pd.read_csv('test_data_solution.txt', sep=':::', names=['ID', 'Title', 'Genre', 'Description'], engine='python')

tfidf = TfidfVectorizer(stop_words='english', max_features=25000, ngram_range=(1, 2), sublinear_tf=True)

X_train = tfidf.fit_transform(train_data['Description'])
y_train = train_data['Genre']
X_test = tfidf.transform(test_data['Description'])
y_test = test_data['Genre']


model = LinearSVC(class_weight='balanced', max_iter=2000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print(f"Final Optimized Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred, zero_division=0))