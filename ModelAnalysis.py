import gradio as gr
import pickle
import joblib
import pandas as pd
import os
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from PIL import Image

# Load the saved TF-IDF vectorizer and Naive Bayes classifier
root_path = "FYP - TEXT CLASSIFICATION"
loaded_vectorizer = joblib.load(os.path.join(root_path, '../Modelling/tfidf_vectorizer.pkl'))
naive_bayes_classifier = joblib.load(os.path.join(root_path, '../Modelling/model.h5'))

# Function to construct full data of ranked locations
def construct_full_data_ranked_locations():
    with open(os.path.join(root_path, "../Data Cleaning/ranked_locations.pkl"), "rb") as f:
        ranked_locations = pickle.load(f)

    df = pd.DataFrame(ranked_locations, columns=["Entity Location", "Frequency"])
    return df

# Use relative paths for the image files
wordcountpath = "Data Cleaning/word_count_distribution.png"
locationspath = "Data Cleaning/most_common_locations.png"
organizationspath = "Data Cleaning/most_common_organizations.png"
classdistributionpath = "Dataset/class_distribution.png"
raw_wordcountdistributionpath = "Dataset/raw_word_count_distribution.png"
wordcloudpath = "Modelling/Word_Cloud_Dataset.png"

# Open the images using PIL
wordcount_image = Image.open(wordcountpath)
locations_image = Image.open(locationspath)
organizations_image = Image.open(organizationspath)
class_distribution_image = Image.open(classdistributionpath)
raw_wordcount_image = Image.open(raw_wordcountdistributionpath)
wordcloud_image = Image.open(wordcloudpath)

# Display the images using Gradio
with gr.Blocks() as EDA:
    with gr.Row():
        with gr.Column():
            with gr.Accordion("Raw Data", open=True):
                with gr.Row():
                    with gr.Column():
                        gr.Image(class_distribution_image, label="Class Distribution",show_download_button=False)
                with gr.Row():
                    with gr.Column():
                        gr.Image(raw_wordcount_image, label = "Word Count", show_download_button=False)

        with gr.Column():
            with gr.Accordion("Clean Data",open=True):
                with gr.Row():
                    with gr.Column():
                        gr.Image(wordcloud_image, label="Word Cloud", show_download_button=False)
                with gr.Row():
                    with gr.Column():
                        gr.Image(wordcount_image, label="Word Count",show_download_button=False)

    with gr.Row():
        with gr.Column():
            with gr.Accordion("Name Entity Recognition (NER)", open=False):
                with gr.Row():
                    with gr.Column():
                        gr.Image(organizations_image, label="Name Entity Recognition (Organizations)",show_download_button=False, scale=1)
                with gr.Row():
                    with gr.Column():
                        gr.Image(locations_image, label="Name Entity Recognition (Locations)",show_download_button=False, scale=1)