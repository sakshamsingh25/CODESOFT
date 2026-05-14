import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

print("Loading spam.csv...")
df = pd.read_csv('spam.csv', encoding='latin-1')

df = df[['v1', 'v2']]
df.columns = ['label', 'message']


X = df['message']
y = df['label']


tfidf = TfidfVectorizer(stop_words='english')
X_tfidf = tfidf.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)


print("Training Spam Detector...")
model = MultinomialNB()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8. Test it yourself!
def check_spam(text):
    vector = tfidf.transform([text])
    return model.predict(vector)[0]

sample_msg = "CONGRATULATIONS! You have won a $1,000 Walmart gift card. Click here to claim now!"
print(f"\nTest Prediction: '{sample_msg}' is -> {check_spam(sample_msg)}")