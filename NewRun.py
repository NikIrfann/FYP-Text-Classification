import gradio as gr
from Classification import single_query, input_file
from ModelAnalysis import EDA

# Define the theme for Gradio
theme = gr.themes.Soft()

ModelAnalysis = gr.TabbedInterface([EDA], 
                          ["Exploratory Data Analysis"])

DemoClassification = gr.TabbedInterface([single_query, input_file], 
                          ["Single Query","Input File Demo"])


MainPage= gr.TabbedInterface([DemoClassification,ModelAnalysis], ["Model Demonstration","Model Analysis"],theme=theme)

MainPage.launch()

