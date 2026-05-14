import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


print("Loading fraudTest.csv...")
df = pd.read_csv('fraudTest.csv')

unwanted_cols = ['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'first', 'last', 
                 'street', 'city', 'state', 'zip', 'trans_num', 'unix_time', 'dob']
df = df.drop(columns=[c for c in unwanted_cols if c in df.columns])

le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    print(f"Encoding column: {col}")
    df[col] = le.fit_transform(df[col].astype(str))


X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("Training Random Forest model... (This may take 1-2 minutes)")
model = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print(f"\n--- Results for Task 2 ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))