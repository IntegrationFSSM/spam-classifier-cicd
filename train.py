import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import skops.io as sio
import os

# Créer les dossiers s'ils n'existent pas
os.makedirs("Results", exist_ok=True)
os.makedirs("Model", exist_ok=True)

# 1. Charger les données
print("Loading data...")
spam_df = pd.read_csv("Data/spam.csv", encoding='latin-1')

# Ajuster selon votre dataset (vérifier les noms de colonnes)
# Si vos colonnes sont 'v1' et 'v2', utiliser ceci :
if 'v1' in spam_df.columns and 'v2' in spam_df.columns:
    spam_df = spam_df[['v1', 'v2']]
    spam_df.columns = ['label', 'text']
# Sinon, si déjà nommées 'label' et 'text', garder tel quel

# Nettoyer et mélanger
spam_df = spam_df.dropna()
spam_df = spam_df.sample(frac=1, random_state=125).reset_index(drop=True)

print(f"Dataset loaded: {len(spam_df)} messages")
print(spam_df['label'].value_counts())

# 2. Préparer X et y
X = spam_df['text'].values
y = spam_df['label'].values

# 3. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=125
)

# 4. Créer le Pipeline
print("Training model...")
pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=3000, stop_words='english')),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=125))
])

# Entraîner
pipe.fit(X_train, y_train)

# 5. Prédictions
predictions = pipe.predict(X_test)

# 6. Calculer les métriques
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, pos_label=1, average='binary')

print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")

# 7. Sauvegarder les métriques
with open("Results/metrics.txt", "w") as outfile:
    outfile.write(f"\nAccuracy = {round(accuracy, 2)}, F1 Score = {round(f1, 2)}.")

# 8. Créer et sauvegarder la matrice de confusion
cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - Spam Classifier')
plt.savefig("Results/model_results.png", dpi=120, bbox_inches='tight')
print("Confusion matrix saved!")

# 9. Sauvegarder le modèle
sio.dump(pipe, "Model/spam_pipeline.skops")
print("Model saved successfully!")