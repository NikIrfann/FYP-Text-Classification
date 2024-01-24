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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from PIL import Image

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")



# Load the saved TF-IDF vectorizer and Naive Bayes classifier
root_path = "FYP - TEXT CLASSIFICATION"
loaded_vectorizer = joblib.load(os.path.join(root_path, '../Modelling/tfidf_vectorizer.pkl'))
naive_bayes_classifier = joblib.load(os.path.join(root_path, '../Modelling/model.h5'))

# Function for cleaning text data
def clean_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
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
    predictions_proba = class_det(query, loaded_vectorizer, naive_bayes_classifier, return_proba=True)
    top_class_idx = predictions_proba.argmax()
    top_class_prob = predictions_proba.max()
    top_class = naive_bayes_classifier.classes_[top_class_idx]
    return top_class, top_class_prob

# def visualize_data(Details):
#     img_dir = os.path.expanduser('~/visualization_images')
#     os.makedirs(img_dir, exist_ok=True)

#     wordcloud = WordCloud(width=800, height=400, background_color='white').generate(Details)
#     plt.figure(figsize=(10, 5))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis('off')
#     plt.title('Word Cloud for Most Frequent Words')
#     plt.savefig(os.path.join(img_dir, 'word_cloud.png'))

#     word_cloud_img_path = os.path.join(img_dir, 'word_cloud.png')
#     return word_cloud_img_path

# Function to predict and display results for a single query


#######
#####
####
###

###
####
#####
######
def predict(query, loaded_vectorizer, naive_bayes_classifier):
    try:
        word_count = len(query.split())

        top_class, top_class_prob = class_prob_conf(query)


        cleaned_text = clean_text(query)


        final_words = word_tokenize(cleaned_text)
        counts = Counter(final_words)

        # df = pd.DataFrame(counts.most_common(10), columns=['Words', 'Counts'])
        # bar_plot = gr.BarPlot(df, x="Words", y="Counts", width=500, title="Top 10 Most Common Words")       

        # Display the top predicted class and its confidence score
        result_text_class = f"{top_class}"
        result_top_class = f"{top_class_prob:.2%}"

        # wordcloud_img_path = visualize_data(cleaned_text)

        # word_cloud = gr.Image(wordcloud_img_path, label="Word Cloud")

        return (
            word_count,
            result_text_class, 
            result_top_class
        )

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
        with gr.Column():
            word_count = gr.Textbox(label="Word Count")
        with gr.Column():
            predictclass = gr.Textbox(label='Category of Places')
        with gr.Column():
            result_prob = gr.Textbox(label='Probability')
    with gr.Row():
        with gr.Accordion("Click HERE to see list of words to be use for each category",open=False):
            with gr.Row():
                with gr.Column():
                    gr.Image(Aero_Image, label="Most Common Words for Aerospace", show_download_button=False)
                with gr.Column():
                    gr.Image(Bio_Image, label="Most Common Words for Biotechnology",show_download_button=False)
            with gr.Row():
                with gr.Column():
                        gr.Image(Robotic_Image, label="Most Common Words for Robotic",show_download_button=False)
                with gr.Column():
                    gr.Image(NoneRelated_Image, label="Most Common Words for None Related",show_download_button=False)



    run_btn.click(lambda query: predict(query, loaded_vectorizer, naive_bayes_classifier), inputs=inp, outputs=[word_count,predictclass,result_prob])

#############################################################################################################################################################################
    #############################################################################################################################################################################
        #############################################################################################################################################################################
    #############################################################################################################################################################################
#############################################################################################################################################################################
# Function to make predictions for multiple queries and calculate max, min, and average word counts
def predict_mult(df: pd.DataFrame):
    vectorized_input_data = loaded_vectorizer.transform(df['Details'])
    predictions = naive_bayes_classifier.predict(vectorized_input_data)
    new_df = df.copy()
    new_df['Predictions'] = predictions.tolist()

    # Calculate word count for each query
    new_df['WordCount'] = new_df['Details'].apply(get_word_count)

    # Assuming you have a column named 'Actual_Labels' in your DataFrame
    actual_labels = df['Class']  # Change 'Actual_Labels' to the actual column name
    accuracy_ = calculate_accuracy(predictions, actual_labels)
    accuracy = f"{accuracy_:.2%}"
    
    # Calculate max, min, and average word counts
    max_word_count = new_df['WordCount'].max()
    min_word_count = new_df['WordCount'].min()
    avg_word_count = int(new_df['WordCount'].mean())

    return new_df, accuracy, max_word_count, min_word_count, avg_word_count



# Path to image file
confusionmatrixpath = "Modelling/confusion_matrix.png"
# Open the images using PIL
confusionmatrix_image = Image.open(confusionmatrixpath)



#############################################################################################################################################################################
    #############################################################################################################################################################################
        #############################################################################################################################################################################
    #############################################################################################################################################################################
#############################################################################################################################################################################

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

# # Function to make predictions for multiple queries
# def predict_mult(df: pd.DataFrame):
#     vectorized_input_data = loaded_vectorizer.transform(df['Details'])
#     predictions = naive_bayes_classifier.predict(vectorized_input_data)
#     new_df = df.copy()
#     new_df['Predictions'] = predictions.tolist()

#     # Calculate word count for each query
#     new_df['Word Count'] = new_df['Details'].apply(get_word_count)

#     # Assuming you have a column named 'Actual_Labels' in your DataFrame
#     actual_labels = df['Class']  # Change 'Actual_Labels' to the actual column name
#     accuracy = calculate_accuracy(predictions, actual_labels)
#     return new_df, accuracy

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
                        run_button = gr.Button("Run")
    with gr.Row():
        with gr.Column(scale=4):      
            with gr.Accordion("Click Here to see the updated Dataframe or can Download and see it in CSV", open=False):
                with gr.Row():
                    with gr.Column():
                        file_out = gr.DataFrame()
        with gr.Column(scale=1):    
            download_button = gr.Button("Download")
    with gr.Row():
        gr.Markdown("Results")
    with gr.Row():
        with gr.Column():
            accuracy_text = gr.Textbox(label='Accuracy')
        with gr.Column():
            max_word_count_text = gr.Textbox(label='Max Word Count')
        with gr.Column():
            min_word_count_text = gr.Textbox(label='Min Word Count')
        with gr.Column():
            avg_word_count_text = gr.Textbox(label='Avg Word Count')
    with gr.Row():
        gr.Image(confusionmatrix_image, label="Confusion Matrix", show_download_button=False)

    upload_button.upload(lambda file_path: file_path, inputs=upload_button, outputs=file_inp)
    run_button.click(predict_mult, inputs=file_inp, outputs=[file_out, accuracy_text,max_word_count_text, min_word_count_text, avg_word_count_text])
    download_button.click(download_df, inputs=file_out)


