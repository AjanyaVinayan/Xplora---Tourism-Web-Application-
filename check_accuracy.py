import pickle
import nltk
import os
from sklearn.metrics import accuracy_score

def evaluate_model():
    if not os.path.exists('preprocessed_data.pkl') or not os.path.exists('decision_tree_classifier.pkl'):
        print("Required pickle files not found. Generate them first.")
        return

    with open('preprocessed_data.pkl', 'rb') as f:
        data = pickle.load(f)
        features_data = data['features_data']

    with open('decision_tree_classifier.pkl', 'rb') as f:
        classifier = pickle.load(f)

    # Recreate the train/test split (assuming 80/20 split without a fixed seed for shuffling like in views.py and main.py)
    # Since we can't reliably get the exact test set without saving it during training, 
    # we'll evaluate on the entire dataset to see how well it fits.
    # To get a more realistic accuracy metric, one should perform cross-validation or save the exact test set.
    
    true_labels = [category for (features, category) in features_data]
    predicted_labels = [classifier.classify(features) for (features, category) in features_data]

    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Accuracy on the entire dataset: {accuracy * 100:.2f}%")

    # If we evaluate on a generated split (which will mix some train data into test if it's reshuffled)
    import random
    random.seed(42) # Try to somewhat replicate a split for demonstration, but it's not the original split.
    temp_data = features_data.copy()
    random.shuffle(temp_data)
    split_index = int(len(temp_data) * 0.8)
    test_data = temp_data[split_index:]
    
    test_accuracy = nltk.classify.accuracy(classifier, test_data)
    print(f"Accuracy on a new random 20% test split: {test_accuracy * 100:.2f}%")

    print("\nNote: Since the original exact train/test split wasn't saved when the model was trained, ")
    print("the first accuracy represents how well the model learned the training data (which includes the test set),")
    print("and the second accuracy tests against a random subset that likely overlaps with the original training data.")

if __name__ == '__main__':
    evaluate_model()
