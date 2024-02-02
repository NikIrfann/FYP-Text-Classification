import gradio as gr
import joblib
import pickle
import pandas as pd
import os
import sklearn
from collections import Counter
from sklearn.calibration import LabelEncoder
from sklearn.metrics import accuracy_score
import spacy
import matplotlib
matplotlib.use("agg")  # Add this line to set the backend to 'agg'
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
from PIL import Image
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


# Load the saved TF-IDF vectorizer and Naive Bayes classifier
root_path = "FYP - TEXT CLASSIFICATION"
loaded_vectorizer = joblib.load(os.path.join(root_path, '../Modelling/tfidf_vectorizer.pkl'))
naive_bayes_classifier = joblib.load(os.path.join(root_path, '../Modelling/model.h5'))

csvFile =os.path.join(root_path,"../Data Cleaning/CleanedData.csv")
df = pd.read_csv(csvFile)

# Preprocess the 'Details' column
stop_words = set(nltk.corpus.stopwords.words('english'))
# Function for cleaning text data
def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

# Function to make text classification predictions
def class_det(Details, vectorizer, classifier, return_proba=False):
    input_data = [Details]
    vectorized_input_data = vectorizer.transform(input_data)
    
    if return_proba:
        predictions_proba = classifier.predict_proba(vectorized_input_data)
        return predictions_proba
    else:
        predictions = classifier.predict(vectorized_input_data)
        return predictions

# Function to calculate class probability and confidence
def class_prob_conf(query):
    cleaned_query = clean_text(query)
    predictions_proba = class_det(cleaned_query, loaded_vectorizer, naive_bayes_classifier, return_proba=True)
    top_class_idx = predictions_proba.argmax()
    top_class_prob = predictions_proba.max()
    top_class = naive_bayes_classifier.classes_[top_class_idx]
    return top_class, top_class_prob

def predict(query, loaded_vectorizer, naive_bayes_classifier, df):
    try:
        cleaned_text = clean_text(query)
        print("Cleaned Text:", cleaned_text)
        print("Tokenized Query:", [token.text for token in nlp(cleaned_text)])

        feature_names = loaded_vectorizer.get_feature_names_out()
        print("Feature Names:", feature_names)

        # Use the class_det function to get predictions from the Naive Bayes classifier
        predictions_proba = class_det(cleaned_text, loaded_vectorizer, naive_bayes_classifier, return_proba=True)

        # Get the predicted class and its probability
        top_class_idx = predictions_proba.argmax()
        top_class_prob = predictions_proba.max()
        top_class = naive_bayes_classifier.classes_[top_class_idx]

        # Get the rows from the DataFrame associated with the predicted class
        predicted_class_rows = df[df['Class'] == top_class][['Country','Place', 'Details', 'Class']]

        # Calculate TF-IDF vectors for places
        place_tfidf_matrix = loaded_vectorizer.transform(predicted_class_rows['Details'])
        # Calculate cosine similarity between query and places
        query_tfidf_vector = loaded_vectorizer.transform([cleaned_text])
        print("Query TF-IDF Vector:", query_tfidf_vector)

        similarity_scores = cosine_similarity(query_tfidf_vector, place_tfidf_matrix).flatten()

        # Filter places based on a similarity threshold (you can adjust this threshold)
        similarity_threshold = 0.2
        matched_rows = predicted_class_rows[similarity_scores >= similarity_threshold]

        if matched_rows.empty:
            return "No places found for the given query and predicted class.", "N/A", "N/A"

        matched_rows = matched_rows.copy()  # create a copy to avoid the SettingWithCopyWarning
        matched_rows.loc[:, 'Similarity'] = similarity_scores[similarity_scores >= similarity_threshold]

        # Sort by similarity in descending order
        matched_rows = matched_rows.sort_values(by='Similarity', ascending=False)

        for index, row in matched_rows.iterrows():
            print(f"\nPlace: {row['Place']}\nCountry: {row['Country']}\nDetails: {row['Details']}\nClass: {row['Class']}\nSimilarity: {row['Similarity']:.4f}\n{'-'*50}")

        # Format the list of places for the interface output with indices
        places_list = [f"{i+1} - {place}, {country}" for i, (place, country) in enumerate(zip(matched_rows['Place'], matched_rows['Country']))]
        places_text = "\n".join(places_list)

        # Display the top predicted class and its confidence score
        result_text_class = f"{top_class}"
        result_top_class = f"{top_class_prob:.2%}"

        return places_text, result_text_class, result_top_class

    except Exception as e:
        print("Error in predict function:", e)
        raise e
    
##
# Relative path to image files
MostWords_Path_Aero = "Data Cleaning/most_common_words_class_Aerospace.png"
MostWords_Path_Bio = "Data Cleaning/most_common_words_class_Biotechnology.png"
MostWords_Path_Robotic = "Data Cleaning/most_common_words_class_Robotic.png"
MostWords_Path_NoneRelated = "Data Cleaning/most_common_words_class_None Related.png"

# Open Image 
Aero_Image = Image.open(MostWords_Path_Aero)
Bio_Image = Image.open(MostWords_Path_Bio)
Robotic_Image = Image.open(MostWords_Path_Robotic)
NoneRelated_Image = Image.open(MostWords_Path_NoneRelated)
#
##

# Gradio interface for a single query
with gr.Blocks() as single_query:
    with gr.Row():
        with gr.Column(scale=9):
            inp = gr.Textbox(label='Input Query')
        with gr.Column(scale=1):
            with gr.Row():
                run_btn = gr.Button("Run", size="lg")
    with gr.Row():
        with gr.Column(scale=4):
            predicted_class_places_label = gr.Textbox(label='Places Related to Query')
       
        with gr.Column(scale=1):
            predictclass = gr.Textbox(label='Category of Places')
            result_prob = gr.Textbox(label='Probability')
    with gr.Row():
        with gr.Accordion("Click HERE to see list of suggestion words for input query",open=False):
            with gr.Row():
                with gr.Column():
                    gr.Image(Aero_Image, label="Most Common Words for Aerospace Category", show_download_button=False)
            with gr.Row():
                with gr.Column():
                    gr.Image(Bio_Image, label="Most Common Words for Biotechnology Category",show_download_button=False)
            with gr.Row():
                with gr.Column():
                        gr.Image(Robotic_Image, label="Most Common Words for Robotic Category",show_download_button=False)
            with gr.Row():
                with gr.Column():
                    gr.Image(NoneRelated_Image, label="Most Common Words for None Related Category",show_download_button=False)


    run_btn.click(lambda query: predict(query, loaded_vectorizer, naive_bayes_classifier,df), inputs=inp, outputs=[predicted_class_places_label,predictclass,result_prob])

###
   ### End of Single Query Part 
###

# Function to make predictions for multiple queries and calculate max, min, and average word counts
def predict_mult(df: pd.DataFrame):
    vectorized_input_data = loaded_vectorizer.transform(df['Details'])
    predictions = naive_bayes_classifier.predict(vectorized_input_data)
    new_df = df.copy()
    new_df['Predictions'] = predictions.tolist()

    # Calculate word count for each query
    new_df['WordCountAfterCleaning'] = new_df['Details'].apply(get_word_count)

    # Assuming you have a column named 'Actual_Labels' in your DataFrame
    actual_labels = df['Class']  # Change 'Actual_Labels' to the actual column name
    accuracy_ = calculate_accuracy(predictions, actual_labels)
    f1_score_ = calculate_f1_score(predictions, actual_labels)
    accuracy = f"{accuracy_:.2%}"
    f1_score_str = f"{f1_score_:.2f}"
    
    # Calculate max, min, and average word counts
    max_word_count = new_df['WordCountAfterCleaning'].max()
    min_word_count = new_df['WordCountAfterCleaning'].min()
    avg_word_count = int(new_df['WordCountAfterCleaning'].mean())

    return new_df, accuracy, f1_score_str, max_word_count, min_word_count, avg_word_count

# Function to calculate F1 score
def calculate_f1_score(predictions, actual_labels):
    f1_score_ = f1_score(actual_labels, predictions, average='weighted')
    return f1_score_

# Function to calculate word count for a single query
def get_word_count(text):
    tokens = word_tokenize(text)
    return len(tokens)

# Function to download predictions to a CSV file
def download_df(file: pd.DataFrame):
    download_path = os.path.join("predicted.csv")
    file.to_csv(download_path)
    print(f"Predictions Downloaded to: {download_path}")

# Function to calculate accuracy
def calculate_accuracy(predictions, actual_labels):
    accuracy = accuracy_score(actual_labels, predictions)
    return accuracy

# Gradio interface for the input file with additional outputs for max, min, and avg word counts
with gr.Blocks() as input_file:
    gr.Markdown("Upload a CSV with a column of queries titled 'Details'. Click Run to see and download predictions.")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                with gr.Column(scale=4):
                    file_inp = gr.DataFrame()
                with gr.Column(scale=1):
                    with gr.Row():
                        upload_button = gr.UploadButton("Upload File", file_types=["file"])
                    with gr.Row():
                        run_button = gr.Button("Predict")
    with gr.Row():
        with gr.Column(scale=4):      
            with gr.Accordion("Show data prediction", open=False):
                with gr.Row():
                    with gr.Column():
                        file_out = gr.DataFrame()
        with gr.Column(scale=1):    
            download_button = gr.Button("Download")
    with gr.Row():
        with gr.Accordion("Show results", open=False):
            with gr.Row():
                with gr.Column():
                    accuracy_text = gr.Textbox(label='Accuracy')
                with gr.Column():
                    f1_score_text = gr.Textbox(label='F1 Score')
                with gr.Column():
                    text = gr.Textbox(visible=False)
            with gr.Row():
                with gr.Column():
                    max_word_count_text = gr.Textbox(label='Max Word Count')
                with gr.Column():
                    min_word_count_text = gr.Textbox(label='Min Word Count')
                with gr.Column():
                    avg_word_count_text = gr.Textbox(label='Avg Word Count')


    upload_button.upload(lambda file_path: file_path, inputs=upload_button, outputs=file_inp)
    run_button.click(predict_mult, inputs=file_inp, outputs=[file_out, accuracy_text,f1_score_text, max_word_count_text, min_word_count_text, avg_word_count_text])
    download_button.click(download_df, inputs=file_out)