import sys
import re
import sqlite3
import pandas as pd
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier

def load_data(database_filepath):
    '''

    Extract data from SQL database

    Arg: Path for database

    Returns: Training Data (DataFrame),
             Target (Series)
             Categories (Array)

    '''
    conn = sqlite3.connect(database_filepath)

    title = database_filepath.split('/')[-1].split('.db')[0]
    # load data from database
    df = pd.read_sql(f'SELECT * FROM {title}', conn)

    return df['message'], df.iloc[:,4:], df.iloc[:,4:].columns

def tokenize(text):

    '''
    Normalizing Text

    Splits up text into words
    Replace URLs with placeholder
    Remove Punctuation
    Lemmatizes words
    Removes Stop Words

    Args: text (str)

    Returns: Normalised text (list)

    '''

    stops = stopwords.words('english')

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_regex, 'url', text)

    text = re.sub('[^a-zA-Z0-9 ]',' ',text)

    lemmatizer = WordNetLemmatizer()

    #tokenize text, remove stops, Lemmatize, lower case and strip whitespace
    return [lemmatizer.lemmatize(tok).lower().strip() for tok in word_tokenize(text) if tok not in stops]

def build_model():

    '''
    NLP Pipeline and GridSearch to find optimium Transformation params

    Args: None

    Returns: NLP Pipeline (Pipeline)

    '''

    parameters = {'vect__max_df': (0.5,1.0),
                'tfidf__use_idf': (True, False)}

    pipe = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
        ('clf',MultiOutputClassifier(AdaBoostClassifier()))])

    return GridSearchCV(pipe, param_grid = parameters)

def evaluate_model(model, X_test, Y_test, category_names):

    '''

    Prints Precision, Recall, F1 score for all categories

    Args: model - Transformation, Classifcation (Pipeline)
          test data - Pandas series
          true values - DataFrame of True Classification
          category names - List


    Returns: Nothing

    '''
    y_predict = model.predict(X_test)

    results = {}
    cat_names = list(category_names)

    print(classification_report(Y_test, y_predict, target_names = cat_names))

    for i in range(y_predict.shape[1]):
        results[cat_names[i]] = classification_report(Y_test.iloc[:,i],y_predict[:,i],output_dict=True)['macro avg']

    precision = [results[k]['precision'] for k in results]
    recall = [results[k]['recall'] for k in results]
    f1_score = [results[k]['f1-score'] for k in results]

    results_table = [('precision',precision),('recall',recall),('f1_score',f1_score)]

    for r in results_table:
        print(f'Average {r[0]} of Model: {sum(r[1])/len(r[1])*100:.2f}%')


def save_model(model, model_filepath):

    '''
    Save model to pkl.file for future use

    Args: Calssification model, full file path to save model

    Returns: None

    '''

    with open(model_filepath,'wb') as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
