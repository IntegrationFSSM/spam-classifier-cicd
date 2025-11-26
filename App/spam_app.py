import gradio as gr
import skops.io as sio

# Charger le modèle
unknown_types = sio.get_untrusted_types(file="./Model/spam_pipeline.skops")
pipe = sio.load("./Model/spam_pipeline.skops", trusted=unknown_types)

def predict_spam(message):
    """Prédire si un message est spam ou non.
    
    Args:
        message (str): Le message à analyser
        
    Returns:
        str: Prédiction (SPAM ou NOT SPAM)
    """
    if not message or message.strip() == "":
        return "⚠️ Veuillez entrer un message"
    
    prediction = pipe.predict([message])[0]
    
    if prediction == 1:
        return "🚨 SPAM DÉTECTÉ"
    else:
        return "✅ NOT SPAM (Message légitime)"

# Interface Gradio
inputs = gr.Textbox(
    lines=5, 
    placeholder="Entrez votre message ici...",
    label="Message à analyser"
)

outputs = gr.Label(label="Résultat")

examples = [
    ["Congratulations! You won a $1000 gift card. Click here now!"],
    ["Hi, can we meet tomorrow at 3pm for coffee?"],
    ["URGENT: Your account will be closed. Verify now at bit.ly/123"],
    ["Hey, how are you doing? Long time no see!"],
    ["You have been selected for a FREE prize. Call 1-800-WIN-NOW"],
    ["Meeting rescheduled to next Monday at 10am"]
]

title = "📧 Spam Classifier"
description = "Détectez automatiquement si un message est un spam ou non. Entrez un message et l'IA vous dira s'il est suspect."
article = """
### À propos
Cette application utilise un modèle de Machine Learning entraîné avec un pipeline CI/CD automatisé.
- **Algorithme**: Random Forest avec TF-IDF
- **Déploiement**: Automatisé via GitHub Actions
- **Technologies**: scikit-learn, Gradio, Hugging Face
"""

# Lancer l'interface
gr.Interface(
    fn=predict_spam,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title=title,
    description=description,
    article=article,
).launch(theme=gr.themes.Soft())