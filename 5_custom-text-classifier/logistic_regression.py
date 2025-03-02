import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Given a labelled data set, we can train a model to predict the sentiment of a given text.

labelled_data = pd.DataFrame([("i love spending time with my friends and family", "positive"),
                    ("that was the best meal i've ever had in my life", "positive"),
                    ("i feel so grateful for everything i have in my life", "positive"),
                    ("i received a promotion at work and i couldn't be happier", "positive"),
                    ("watching a beautiful sunset always fills me with joy", "positive"),
                    ("my partner surprised me with a thoughtful gift and it made my day", "positive"),
                    ("i am so proud of my daughter for graduating with honors", "positive"),
                    ("listening to my favorite music always puts me in a good mood", "positive"),
                    ("i love the feeling of accomplishment after completing a challenging task", "positive"),
                    ("i am excited to go on vacation next week", "positive"),
                    ("i feel so overwhelmed with work and responsibilities", "negative"),
                    ("the traffic during my commute is always so frustrating", "negative"),
                    ("i received a parking ticket and it ruined my day", "negative"),
                    ("i got into an argument with my partner and we're not speaking", "negative"),
                    ("i have a headache and i feel terrible", "negative"),
                    ("i received a rejection letter for the job i really wanted", "negative"),
                    ("my car broke down and it's going to be expensive to fix", "negative"),
                    ("i'm feeling sad because i miss my friends who live far away", "negative"),
                    ("i'm frustrated because i can't seem to make progress on my project", "negative"),
                    ("i'm disappointed because my team lost the game", "negative")], columns=["text", "sentiment"])

# shuffle and drop the index column
labelled_data = labelled_data.sample(frac=1)
labelled_data = labelled_data.reset_index(drop=True)

# Separate the model inputs (which is actual text) (X) and outputs (y) (which is the sentiment label)
X = labelled_data["text"]
y = labelled_data["sentiment"]

# split the data into training and testing sets. 30% of the data will be used for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

# Convert the train and text data into a matrix of token counts
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# Evaluate the model
text_accuracy = model.score(X_test_vectorized, y_test)
print(f"Model accuracy: {text_accuracy}")

# Predict the sentiment of a new text
y_pred_lr = model.predict(X_test_vectorized)
print(classification_report(y_test, y_pred_lr))