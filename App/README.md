---
title: Spam Classifier
emoji: 📧
colorFrom: red
colorTo: orange
sdk: gradio
sdk_version: 4.16.0
app_file: spam_app.py
pinned: false
license: apache-2.0
---
```

✅ **Vérification** : Le fichier est sauvegardé

---

## **ÉTAPE 6 : Configurer App/requirements.txt**

### Actions à faire :

Ouvrir `App/requirements.txt` et **ajouter** :
```
scikit-learn
skops
```

✅ **Vérification** : Le fichier contient 2 lignes

---

## **ÉTAPE 7 : Configurer requirements.txt (racine)**

### Actions à faire :

Ouvrir `requirements.txt` (à la racine) et **ajouter** :
```
pandas
scikit-learn
skops
matplotlib
black