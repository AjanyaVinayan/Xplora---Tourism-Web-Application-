from langdetect.lang_detect_exception import LangDetectException
from flask import Flask, render_template, request, jsonify
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk
from googletrans import Translator
from nltk.corpus import stopwords
from langdetect import detect
import random
import nltk
import csv
import os
import pickle

app = Flask(__name__)

# Load NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Preprocessing functions
def preprocess(sentence):
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    tagged_tokens = pos_tag(tokens)
    named_entities = ne_chunk(tagged_tokens)
    # Exclude proper nouns from filtering
    filtered_words = [word for word, tag in tagged_tokens if tag not in ['NNP', 'NNPS']]
    return filtered_words


def extract_features(text):
    words = preprocess(text)
    return [WordNetLemmatizer().lemmatize(word) for word in words]

# Chatbot functions
def train_and_classify(training_data, test_data):
    classifier = nltk.classify.DecisionTreeClassifier.train(training_data, entropy_cutoff=0.6, support_cutoff=6)
    training_set_accuracy = nltk.classify.accuracy(classifier, training_data)
    test_set_accuracy = nltk.classify.accuracy(classifier, test_data)
    return classifier, training_set_accuracy, test_set_accuracy


tour = ['Munnar', 'Alleppey', 'Kochi', 'Wayanad', 'Varkala', 'Kumarakom',
        'Vagamon', 'Kovalam', 'Periyar National Park', 'Poovar', 'Kollam',
        'Idukki', 'Sabarimala', 'Kozhikode', 'Bekal', 'Thrissur',
        'Palakkad', 'Thalassery', 'Trivandrum', 'Nelliyampathy', 'Vythiri',
        'Nilambur', 'Ponmudi', 'Kalpetta', 'Malampuzha', 'Kannur',
        'Kasaragod', 'Kottayam', 'Thekkady']

# Chatbot functions
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
            itemss=[]
            for i in items:
                rate=i.split('- rating ')
                # print('rate[1]',rate[1])
                rate[1]=rate[1].replace("'","")
                num=float(rate[1])
                if num>=8.5 and 'rating' in input_sentence:
                    rate[0]=rate[0].replace('onwards','')
                    itemss.append(rate[0])
                    # print('rate########################',rate)
                else:
                    rate[0]=rate[0].replace('onwards','')
                    itemss.append(rate[0])
                    # print('rate########################',rate)

            selected_items = random.sample(itemss, min(4, len(itemss)))
            return ', '.join(selected_items)
    return answers[category]

# # Load data
# def get_content(filename):
#     doc = os.path.join(filename)
#     with open(doc, 'r') as content_file:
#         lines = csv.reader(content_file, delimiter='|')
#         data = [x for x in lines if len(x) == 3]
#         return data

# data = get_content('data.txt')
print('pass 1')

# # Preprocess data
# answers = {}
# features_data = []
# for text, category, answer in data:
#     features = extract_features(text)
#     features_data.append(({word: True for word in features}, category))
#     answers[category] = answer
# Load the data

with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)
    answers = data['answers']
    features_data = data['features_data']

print("Data loaded successfully from Pickle format.")

# print('pass 2')
# # Split data into training and testing sets
# random.shuffle(features_data)
# split_index = int(len(features_data) * 0.8)
# training_data = features_data[:split_index]
# test_data = features_data[split_index:]

print('pass 3')

# # Train classifier
# classifier = nltk.classify.DecisionTreeClassifier.train(training_data, entropy_cutoff=0.6, support_cutoff=6)
# # print('train data',features_data)

# Load the saved classifier
with open('decision_tree_classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

print("Classifier loaded successfully.")

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')


def translate(text, target_language='en'):
    return Translator().translate(text, dest=target_language).text

def detect_lang(text):
    try:
        return detect(text)
    except:
        return None

preplace = []
@app.route('/get_response', methods=['POST'])

def get_response():
   
    user_input = request.form['user_input']
    lang = detect_lang(user_input)
    if lang =='ml':
        lang='ml'
    elif lang =='hi':
        lang='hi'
    elif lang =='tn':
        lang='tn'
    else:
        lang='en'
    
    translated_to_en = translate(user_input)
    search_string = translated_to_en
    newstring = []
    for place in tour:
        place_lower = place.lower()
        if place_lower in search_string.lower():
            if preplace!=[]:
                preplace.clear()
                preplace.append(place_lower)
                newstring.clear()
                newstring.append(search_string)                
            else:                
                preplace.append(place_lower)
                newstring.append(' '.join(preplace+ [search_string]))
        elif 'hotels' in search_string or 'restaurant' in search_string or 'package' in search_string:
            newstring.append(' '.join(preplace + [search_string])) 
        else:
            search_string=search_string
            newstring.append(search_string)  

    print(f"Original Text: {user_input}")
    print(f"Detected Language: {lang}")
    search_string = ''
    translated_to_en = newstring[-1]
    print(f"Translated to English: {translated_to_en}")

    # Get response in English
    response = reply(translated_to_en, classifier)

    response = response.replace('->', '-').replace("'", '').replace("â",'₹')
    response=response.replace("₹‚¹",'₹').replace("# x20b9;",'₹')
    response = response.split(', ')

    random.shuffle(response)
    response = response[:4]

    translated_responses = []
    for res in response:
        res=res.replace("# x20b9;",'₹')
        if lang=='en':
            
            translated_responses.append(res)
            
        elif lang=='ml':
            res=translate(res, target_language='ml')
            res=res.replace("# x20b9;",'₹')
           
            translated_responses.append(res)
        else:
            translated_responses.append(translate(res, target_language=lang))
    
    print('Translated Responses:', translated_responses)

    
    return jsonify({'response': translated_responses})

    
if __name__ == '__main__':
    app.run(debug=True,port=5002)