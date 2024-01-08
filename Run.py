import gradio as gr
import joblib
from IPython import embed
import pickle
import pandas as pd
import os
import sklearn
from collections import Counter
import spacy
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
nltk.download('stopwords')
nltk.download('punkt')

theme = gr.themes.Soft()
nlp = spacy.load("en_core_web_sm")

print("CURRENT WORKING DIRECTORY:", os.path.dirname(os.path.abspath(__file__)))
print("FILES IN THIS DIRECTORY:", os.listdir())

# Function for cleaning text data
def clean_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

    # Causing lots of word missing alphabet
    # Stemming (optional)
    # porter = PorterStemmer()
    # tokens = [porter.stem(word) for word in tokens]

    # Join the tokens back into a string
    cleaned_text = ' '.join(tokens)
    return cleaned_text

def class_det(Details, vectorizer, classifier):
    input_data = [Details]
    vectorized_input_data = vectorizer.transform(input_data)
    predictions = classifier.predict(vectorized_input_data)
    return predictions

def predict(query):
    preds = class_det(query, loaded_vectorizer, naive_bayes_classifier)
    cleaned_text = clean_text(query)
    final_words = word_tokenize(cleaned_text)
    counts=Counter(final_words)
    
    df = pd.DataFrame(counts.most_common(10), columns=['Words', 'Counts'])
    # Include the Word Cloud image in the output
    wordcloud_img_path = visualize_data(cleaned_text)
    return preds, gr.BarPlot(df, x="Words", y="Counts", width=500, title="Top 10 Most Common Words"),wordcloud_img_path

def visualize_data(Details):
    img_dir = os.path.expanduser('~/visualization_images')
    os.makedirs(img_dir, exist_ok=True)

    # Plot word cloud for most frequent words
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(Details)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud for Most Frequent Words')
    plt.savefig(os.path.join(img_dir, 'word_cloud.png'))

    # Save the generated images to the directory
    word_cloud_img_path = os.path.join(img_dir, 'word_cloud.png')

    return word_cloud_img_path
    



def predict_mult(df:pd.DataFrame):
    # embed()
    # raise Exception
    vectorized_input_data = loaded_vectorizer.transform(df['Details'])
    predictions = naive_bayes_classifier.predict(vectorized_input_data)
    new_df = df.copy()
    new_df['Predictions'] = predictions.tolist()
    return new_df, predictions.tolist()

root_path = "Data Cleaning"
# Load the saved TF-IDF vectorizer
loaded_vectorizer = joblib.load(os.path.join(root_path, 'tfidf_vectorizer.pkl'))
# Load the saved Naive Bayes classifier
naive_bayes_classifier = joblib.load(os.path.join(root_path, 'model.h5'))


with gr.Blocks() as single_query:
    gr.Markdown("Start typing below and then click **Run** to see the 1.")
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(label='Input Query')
            #btn = gr.Button("Run")
        with gr.Column():
            out = gr.Textbox(label='Prediction')
            graph = gr.ScatterPlot(render=False, label='Scatter Plot')
    
    # btn.click(fn=predict, inputs=inp, outputs=[out, graph])

# 1st tab 
demo = gr.Interface(
    fn=predict, 
    inputs=inp, 
    outputs=[out, graph, gr.Image(type="pil", label="Word Cloud") ],
    theme=theme, 
    allow_flagging="never")


def construct_full_data_ranked_locations():
    with open(os.path.join(root_path, "ranked_locations.pkl"), "rb") as f:
        ranked_locations = pickle.load(f)

    return {
        "value": pd.DataFrame(ranked_locations, columns=["Entity Location", "Frequency"]),
        "x": "Entity Location",
        "y": "Frequency",
        "vertical": False,
        "title": "Ranked Entity Locations",
        "width":1200,
        "height":640
    }

# 2nd Tab
#
bar = gr.BarPlot(**construct_full_data_ranked_locations())



# def upload_file(file):
#     return file

def download_df(file:pd.DataFrame):
    download_path = os.path.join("predicted.csv")
    file.to_csv(download_path)
    print(f"Predictions Downloaded to: {download_path}")

with gr.Blocks() as third_page_layout:
    gr.Markdown("Upload a CSV with a column of queries titled 'Details'. Click Run to see and download predictions.")
    with gr.Row():
        with gr.Column():
            file_inp = gr.DataFrame()
            with gr.Row():
                upload_button = gr.UploadButton("Click to Upload a File", file_types=["file"])
                run_button = gr.Button("Run")
        with gr.Column():
            file_out = gr.DataFrame(visible=False)
            out_text = gr.Textbox()
            download_button = gr.Button("Download")
    
    upload_button.upload(lambda file_path: file_path, inputs=upload_button, outputs=file_inp)
    run_button.click(predict_mult, inputs=file_inp, outputs=[file_out, out_text])
    download_button.click(download_df, inputs=file_out)

page = gr.TabbedInterface([demo, bar, third_page_layout], ["Text Classification Demo", "Full Data Plot", "Input File Classification"], theme=theme)

page.launch()