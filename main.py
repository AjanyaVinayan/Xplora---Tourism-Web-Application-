# from langdetect.lang_detect_exception import LangDetectException
from flask import Flask, render_template, request, jsonify  # type: ignore
from nltk.stem.wordnet import WordNetLemmatizer  # type: ignore
from nltk.tokenize import RegexpTokenizer  # type: ignore
from nltk import pos_tag  # type: ignore
# from langdetect import detect
import random
import nltk  # type: ignore
import os
import pickle
import csv

app = Flask(__name__)

# Load NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')

# Preprocessing functions
def preprocess(sentence):
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    tagged_tokens = pos_tag(tokens)
    # Exclude proper nouns from filtering
    filtered_words = [word for word, tag in tagged_tokens if tag not in ['NNP', 'NNPS']]
    return filtered_words

def extract_features(text):
    words = preprocess(text)
    return [WordNetLemmatizer().lemmatize(word) for word in words]

# Chatbot functions
tour = ['Munnar', 'Alleppey', 'Kochi', 'Wayanad', 'Varkala', 'Kumarakom',
        'Vagamon', 'Kovalam', 'Periyar National Park', 'Poovar', 'Kollam',
        'Idukki', 'Sabarimala', 'Kozhikode', 'Bekal', 'Thrissur',
        'Palakkad', 'Thalassery', 'Trivandrum', 'Nelliyampathy', 'Vythiri',
        'Nilambur', 'Ponmudi', 'Kalpetta', 'Malampuzha', 'Kannur',
        'Kasaragod', 'Kottayam', 'Thekkady']

def reply(input_sentence, classifier, input_lang='en'):
    category = classifier.classify({word: True for word in extract_features(input_sentence)})
    response = answers.get(category, "I'm sorry, I don't understand that.")
    if category == 'best-places':
        places = answers[category].split(',')
        return ', '.join(random.sample(places, 5))
    for place in tour:
        if category == f'{place}-package':
            items = answers[category].split("', '")
            selected_items = random.sample(items, min(4, len(items)))
            return ', '.join(selected_items)
    for place in tour:
        if category == f'{place}-restaurant':
            items = answers[category].split("', '")
            selected_items = random.sample(items, min(4, len(items)))
            return ', '.join(selected_items)
    for place in tour:
        if category == f'{place}-Hotels':
            items = answers[category].split("', '")
            itemss = []
            for i in items:
                rate = i.split('- rating ')
                rate[1] = rate[1].replace("'", "")
                num = float(rate[1])
                if num >= 8.5 and 'rating' in input_sentence:
                    rate[0] = rate[0].replace('onwards', '')
                    itemss.append(rate[0])
                else:
                    rate[0] = rate[0].replace('onwards', '')
                    itemss.append(rate[0])
            selected_items = random.sample(itemss, min(4, len(itemss)))
            return ', '.join(selected_items)
    return answers[category]

# Load or preprocess data
def get_content(filename):
    with open(filename, 'r') as content_file:
        lines = csv.reader(content_file, delimiter='|')
        return [x for x in lines if len(x) == 3]

if os.path.exists('preprocessed_data.pkl'):
    with open('preprocessed_data.pkl', 'rb') as f:
        data = pickle.load(f)
        answers = data['answers']
        features_data = data['features_data']
    print("Data loaded successfully from Pickle format.")
else:
    print('Data format not found. preprocessing data.txt. make sure data.txt is located in the root folder')
    data = get_content('data.txt')
    answers = {}
    features_data = []
    for text, category, answer in data:
        features = extract_features(text)
        features_data.append(({word: True for word in features}, category))
        answers[category] = answer
    with open('preprocessed_data.pkl', 'wb') as f:
        pickle.dump({'answers': answers, 'features_data': features_data}, f)
    print("\nData preprocessed and saved to Pickle format.")

# Load or train classifier
if os.path.exists('decision_tree_classifier.pkl'):
    with open('decision_tree_classifier.pkl', 'rb') as f:
        classifier = pickle.load(f)
    print("Classifier loaded successfully from Pickle format.")
else:
    random.shuffle(features_data)
    split_index = int(len(features_data) * 0.8)
    training_data = features_data[:split_index]  # type: ignore
    test_data = features_data[split_index:]  # type: ignore
    classifier = nltk.classify.DecisionTreeClassifier.train(training_data, entropy_cutoff=0.6, support_cutoff=6)
    with open('decision_tree_classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    print("Classifier trained and saved to Pickle format.")

# def detect_lang(text):
#     try:
#         return detect(text)
#     except LangDetectException:
#         print("Language detection failed.")
#         return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']
    print(f"Original Text: {user_input}")
    
    # Pass the input directly to the reply function without translation
    response = reply(user_input, classifier)
    
    # Process and shuffle the response
    response = response.replace('->', '-').replace("'", '').replace("â", '₹').replace("₹‚¹", '₹').replace("# x20b9;", '₹')
    response = response.split(', ')
    random.shuffle(response)
    response = response[:4]  # type: ignore  # Limit the number of responses to 4

    print('Responses:', response)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, port=5002)