# app/ner.py

import spacy

nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    entities = {
        "PERSON": [],
        "DATE": [],
        "TIME": [],
        "ORG": [],
        "GPE": []
    }
    for ent in doc.ents:
        if ent.label_ in entities:
            if ent.text not in entities[ent.label_]:
                entities[ent.label_].append(ent.text)
    return entities
