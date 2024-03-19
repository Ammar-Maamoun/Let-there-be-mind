from flask import Flask, render_template, request
import spacy
import spacy.cli

app = Flask(__name__)

# Function to download and load the Spacy model
def load_spacy_model(model_name="en_core_web_sm"):
    try:
        return spacy.load(model_name)
    except OSError:
        print(f"Model {model_name} not found. Downloading...")
        spacy.cli.download(model_name)
        return spacy.load(model_name)

# Load the English NER and POS tagging model
nlp = load_spacy_model()

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/process', methods=['POST'])
def process():
    # Get user input from the form
    user_text = request.form['text']
    
    # Tokenization
    doc = nlp(user_text)
    tokens = [token.text for token in doc]
    # Adjusted tokenized_text to use " - " as separator
    tokenized_text = " - ".join(tokens)
    
    # POS tagging
    pos_tags = [(token.text, token.pos_) for token in doc]
    pos_tagged_text = " ".join([f"{token}/{tag}" for token, tag in pos_tags])
    
    # NER
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    ner_text = str(entities)
    
    # Pass the results to the template for display
    return render_template("home.html", user_text=user_text, 
                           tokenized_text=tokenized_text, 
                           pos_tagged_text=pos_tagged_text, 
                           ner_text=ner_text)

if __name__ == '__main__':
    app.run(debug=False, port=5012)