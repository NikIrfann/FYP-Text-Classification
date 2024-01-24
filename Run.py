import gradio as gr
import joblib
import pickle
import pandas as pd
import os
import sklearn
from collections import Counter
from sklearn.metrics import accuracy_score
import spacy
import matplotlib
matplotlib.use("agg")  # Add this line to set the backend to 'agg'
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Define the theme for Gradio
theme = gr.themes.Soft()

print("CURRENT WORKING DIRECTORY:", os.path.dirname(os.path.abspath(__file__)))
print("FILES IN THIS DIRECTORY:", os.listdir())


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



# Function to predict and display results for a single query
def predict(query, loaded_vectorizer, naive_bayes_classifier):
    try:
        word_count = len(query.split())

        top_class, top_class_prob = class_prob_conf(query)


        cleaned_text = clean_text(query)


        final_words = word_tokenize(cleaned_text)
        counts = Counter(final_words)

        df = pd.DataFrame(counts.most_common(10), columns=['Words', 'Counts'])
        bar_plot = gr.BarPlot(df, x="Words", y="Counts", width=500, title="Top 10 Most Common Words")       

        # Display the top predicted class and its confidence score
        result_text_class = f"{top_class}"
        result_top_class = f"{top_class_prob:.2%}"

        wordcloud_img_path = visualize_data(cleaned_text)

        word_cloud = gr.Image(wordcloud_img_path, label="Word Cloud")

        return (
            word_count,
            result_text_class, 
            result_top_class,
            bar_plot,
            word_cloud
        )

    except Exception as e:
        print("Error in predict function:", e)
        raise e


 #Function to visualize data and save word cloud image
def visualize_data(Details):
    img_dir = os.path.expanduser('~/visualization_images')
    os.makedirs(img_dir, exist_ok=True)

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(Details)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud for Most Frequent Words')
    plt.savefig(os.path.join(img_dir, 'word_cloud.png'))

    word_cloud_img_path = os.path.join(img_dir, 'word_cloud.png')
    return word_cloud_img_path
    

# Function to calculate accuracy
def calculate_accuracy(predictions, actual_labels):
    accuracy = accuracy_score(actual_labels, predictions)
    return accuracy

# Function to make predictions for multiple queries
def predict_mult(df: pd.DataFrame):
    vectorized_input_data = loaded_vectorizer.transform(df['Details'])
    predictions = naive_bayes_classifier.predict(vectorized_input_data)
    new_df = df.copy()
    new_df['Predictions'] = predictions.tolist()

    # Assuming you have a column named 'Actual_Labels' in your DataFrame
    actual_labels = df['Class']  # Change 'Actual_Labels' to the actual column name
    accuracy = calculate_accuracy(predictions, actual_labels)
    return new_df, predictions.tolist(), accuracy


# Load the saved TF-IDF vectorizer and Naive Bayes classifier
root_path = "FYP - TEXT CLASSIFICATION"
loaded_vectorizer = joblib.load(os.path.join(root_path, '../Modelling/tfidf_vectorizer.pkl'))
naive_bayes_classifier = joblib.load(os.path.join(root_path, '../Modelling/model.h5'))


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
            predictclass = gr.Textbox(label='Text Class')
        with gr.Column():
            result_prob = gr.Textbox(label='Probability')
    with gr.Row():
        similar_words = gr.Textbox(label="Similar words to the class")
    with gr.Row():
        with gr.Column():
            graph = gr.ScatterPlot(label='Scatter Plot')
        with gr.Column():
            word_cloud = gr.Image(type="pil", label="Word Cloud")

    run_btn.click(lambda query: predict(query, loaded_vectorizer, naive_bayes_classifier), inputs=inp, outputs=[word_count,predictclass,result_prob,graph,word_cloud])


# Function to construct full data of ranked locations
def construct_full_data_ranked_locations():
    with open(os.path.join(root_path, "../Data Cleaning/ranked_locations.pkl"), "rb") as f:
        ranked_locations = pickle.load(f)

    df = pd.DataFrame(ranked_locations, columns=["Entity Location", "Frequency"])
    return df

# Function to create a custom bar plot
def custom_bar_plot(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    locations = df["Entity Location"]
    frequencies = df["Frequency"]

    bars = ax.bar(locations, frequencies, color='green')

    for bar, frequency in zip(bars, frequencies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width(), height, str(frequency), ha='left', va='center')

    ax.set_title('Ranked Entity Locations')
    ax.set_xlabel('Entity Location')
    ax.set_ylabel('Frequency')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("bar_plot.png")

# Create the DataFrame and custom bar plot
df_ranked_locations = construct_full_data_ranked_locations()
custom_bar_plot(df_ranked_locations)

with gr.Blocks()as iface:
    gr.Image("bar_plot.png", label="")


# Function to download predictions to a CSV file
def download_df(file: pd.DataFrame):
    download_path = os.path.join("predicted.csv")
    file.to_csv(download_path)
    print(f"Predictions Downloaded to: {download_path}")

# Gradio interface for the 2nd tab
with gr.Blocks() as input_file:
    gr.Markdown("Upload a CSV with a column of queries titled 'Details'. Click Run to see and download predictions.")
    with gr.Row():
        with gr.Column():
            file_inp = gr.DataFrame()
            with gr.Row():
                upload_button = gr.UploadButton("Click to Upload a File", file_types=["file"])
                run_button = gr.Button("Run")
            with gr.Row():
                file_out = gr.DataFrame()
            with gr.Row():
                out_text = gr.Textbox()  
            with gr.Row():
                download_button = gr.Button("Download")
            with gr.Row():
                accuracy_text = gr.Textbox(label='Accuracy')

    upload_button.upload(lambda file_path: file_path, inputs=upload_button, outputs=file_inp)
    run_button.click(predict_mult, inputs=file_inp, outputs=[file_out, out_text, accuracy_text])
    download_button.click(download_df, inputs=file_out)


demo_tabbed = gr.TabbedInterface(
    [single_query, input_file],
    ["Query", "File"],
    title="Text Classification"
)

# Create a Gradio Tabbed Interface
page = gr.TabbedInterface([iface,demo_tabbed], ["Data Analysis","Classification Demo"], 
                          theme=theme)

# Launch the Gradio interface
page.launch()