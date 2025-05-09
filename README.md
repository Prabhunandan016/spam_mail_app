Spam Email Classifier
Overview
The Spam Email Classifier is a machine learning project that classifies emails as spam or ham (non-spam) based on the content of the email. It uses a Naive Bayes classifier combined with a TF-IDF Vectorizer to preprocess the email text. The model is optimized using GridSearchCV for hyperparameter tuning, and it is evaluated using several metrics such as accuracy, precision, recall, and F1 score.

The project also allows real-time predictions through an interactive user input loop.

Features
Text Preprocessing: TF-IDF Vectorizer with unigrams and bigrams to represent the email text.

Model Training: Multinomial Naive Bayes classifier for text classification.

Hyperparameter Tuning: Model optimization using GridSearchCV to select the best alpha parameter for Naive Bayes.

Evaluation Metrics: Model performance evaluated using accuracy, precision, recall, and F1 score.

Confusion Matrix: Visualization of the confusion matrix using Seaborn.

Model Persistence: The trained model is saved using Joblib for future use.

Interactive Prediction: Users can input messages interactively to predict whether the email is spam or ham.

Installation
Requirements
Python 3.x

Required libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, joblib

Steps
Clone the repository:

bash
Copy
Edit
git clone https://github.com/Prabhunandan016/spammail-classifier.git
cd spammail-classifier
Install the necessary libraries:

bash
Copy
Edit
pip install -r requirements.txt
Download the dataset (e.g., from the SMS Spam Collection dataset) and save it as mail_data.csv in the project directory.

Run the script to train and evaluate the model:

bash
Copy
Edit
python train_model.py
Usage
Data Format
The dataset should have the following columns:

Category: A string (either "ham" or "spam") indicating the label of the email.

Message: The content of the email (string).

Example:

csv
Copy
Edit
Category,Message
ham,"Hey, how are you?"
spam,"Congratulations, you've won a free gift card!"
Training the Model
To train the model, run:

bash
Copy
Edit
python train_model.py
The model will use the Multinomial Naive Bayes classifier and GridSearchCV for hyperparameter optimization. Once training is completed, the best model will be saved as spam_model.pkl.

Model Evaluation
After training, the following evaluation metrics will be displayed:

Accuracy: The proportion of correct predictions (e.g., Accuracy: 0.99).

Precision: The ratio of true positives to all predicted positives.

Recall: The ratio of true positives to all actual positives.

F1-Score: The harmonic mean of precision and recall.

Example output:

bash
Copy
Edit
Evaluation Metrics:
Accuracy : 0.99
Precision: 0.99
Recall   : 0.91
F1-score : 0.95

Classification Report:
              precision    recall  f1-score   support

         Ham       0.99      1.00      0.99       966
        Spam       0.99      0.91      0.95       149

    accuracy                           0.99      1115
   macro avg       0.99      0.96      0.97      1115
weighted avg       0.99      0.99      0.99      1115
The Confusion Matrix will also be displayed to visualize the true positive, true negative, false positive, and false negative counts.

Interactive Prediction
You can also use the trained model for interactive predictions:

bash
Copy
Edit
python interactive_predict.py
You will be prompted to enter a message. The model will predict if the message is spam or ham. Type exit to quit.

Load the Saved Model
To load the saved model and use it for future predictions:

python
Copy
Edit
import joblib

# Load the trained model
model = joblib.load("spam_model.pkl")

# Predict a new message
message = "Congratulations! You've won a prize."
prediction = model.predict([message])
print("Prediction:", "Spam" if prediction == 1 else "Ham")
Contributing
Feel free to fork this repository, make changes, and submit pull requests. Contributions are always welcome!
