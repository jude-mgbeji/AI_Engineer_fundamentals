import spacy

nlp = spacy.load("en_core_web_sm")

text = "Barack Obama was born in Hawaii on August 4, 1961."

doc = nlp(text)

for tokem in doc.ents:
    print(f"NAMED ENTITY:{tokem.text}, Labels:{tokem.label_}")