


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import kagglehub


path = "change this"
print("Path to dataset files:", path)



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


data_path = path + "/train.csv"
df = pd.read_csv(data_path)


if 'PassengerId' in df.columns:
    df = df.drop('PassengerId', axis=1)
    
    
    
cols_to_drop = ['Name', 'Ticket', 'Cabin']
for col in cols_to_drop:
    if col in df.columns:
        df = df.drop(col, axis=1)


df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])


df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)


le = LabelEncoder()
df['Survived'] = le.fit_transform(df['Survived'])


X = df.drop('Survived', axis=1)
y = df['Survived']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Survived", "Survived"]))


cm = confusion_matrix(y_test, y_pred)


plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Iris Dataset")
plt.show()
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.flatten()  

# 1. Survived
sns.countplot(x='Survived', data=df, ax=axes[0])
axes[0].set_xticklabels(['Zmarli', 'Przeżyli'])
axes[0].set_title("Liczba osób przeżyłych i zmarłych")

# 2. Survived vs Płeć
sns.countplot(x='Survived', hue='Sex_male', data=df, ax=axes[1])
axes[1].set_xticklabels(['Zmarli', 'Przeżyli'])
axes[1].set_title("Przeżywalność wg płci")
axes[1].legend(['Kobieta', 'Mężczyzna'])

# 3. Survived vs Klasa
sns.countplot(x='Pclass', hue='Survived', data=df, ax=axes[2])
axes[2].set_title("Przeżywalność wg klasy")
axes[2].legend(['Zmarli', 'Przeżyli'])

# 4. Wiek vs Survived
sns.boxplot(x='Survived', y='Age', data=df, ax=axes[3])
axes[3].set_xticklabels(['Zmarli', 'Przeżyli'])
axes[3].set_title("Wiek a przeżycie")

# 5. Feature importance
importances = model.feature_importances_
features = X.columns
sns.barplot(x=importances, y=features, ax=axes[4])
axes[4].set_title("Feature Importance")

# 6. Korelacje
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=axes[5])
axes[5].set_title("Macierz korelacji")


for i in range(6, 9):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

