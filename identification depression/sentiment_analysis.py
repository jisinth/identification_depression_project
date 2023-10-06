# Import necessary libraries
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK resources (if not already downloaded)
nltk.download('vader_lexicon')

# Load CSV data
df = pd.read_csv('abc.csv')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()


# Function to classify tweets as 'Depressed' or 'Happy'
def classify_sentiment(label):
    if label == 0:
        return 'Happy'
    elif label == 1:
        return 'Depressed'
    else:
        return 'Neutral'


# Apply sentiment classification to the 'label' column
df['Sentiment'] = df['label'].apply(classify_sentiment)

# Split data into training and testing sets
X = df['text']  # Tweet text
y = df['label']  # Sentiment labels (0 or 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust max_features as needed
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Create and train Decision Tree classifier
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train_tfidf, y_train)
y_pred_tree = decision_tree.predict(X_test_tfidf)

# Create and train Linear SVM classifier
svm = LinearSVC(random_state=42)
svm.fit(X_train_tfidf, y_train)
y_pred_svm = svm.predict(X_test_tfidf)

# Create and train Logistic Regression classifier
logistic_regression = LogisticRegression(random_state=42)
logistic_regression.fit(X_train_tfidf, y_train)
y_pred_lr = logistic_regression.predict(X_test_tfidf)

# Create and train Naive Bayes (MultinomialNB) classifier
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_tfidf, y_train)
y_pred_nb = naive_bayes.predict(X_test_tfidf)


# Evaluate classifiers
def evaluate_classifier(y_true, y_pred, classifier_name):
    accuracy = accuracy_score(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)
    classification_rep = classification_report(y_true, y_pred)

    print(f'Classifier: {classifier_name}')
    print(f'Accuracy: {accuracy:.2f}\n')
    print('Confusion Matrix:')
    print(confusion)
    print('\nClassification Report:')
    print(classification_rep)

    return accuracy


# Create a dictionary to store accuracies and confusion matrices
results = {}

# Evaluate each classifier and store results
results['Decision Tree'] = {'accuracy': evaluate_classifier(y_test, y_pred_tree, 'Decision Tree')}
results['Linear SVM'] = {'accuracy': evaluate_classifier(y_test, y_pred_svm, 'Linear SVM')}
results['Logistic Regression'] = {'accuracy': evaluate_classifier(y_test, y_pred_lr, 'Logistic Regression')}
results['Naive Bayes'] = {'accuracy': evaluate_classifier(y_test, y_pred_nb, 'Naive Bayes')}

# Create a bar plot for accuracy comparison
plt.figure(figsize=(10, 6))
sns.set(style='whitegrid')
sns.barplot(x=list(results['accuracy'] for results in results.values()), y=list(results.keys()), palette='viridis')
plt.xlabel('Accuracy')
plt.title('Classifier Accuracy Comparison')
plt.xlim(0, 1.0)  # Set the x-axis limit
plt.show()
