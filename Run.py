import gradio as gr
from Classification import single_query, input_file
from ModelAnalysis import EDA

# Define the theme for Gradio
theme = gr.themes.Soft()

ExploratoryDataAnalysis = gr.TabbedInterface([EDA], 
                          [""])

ModelDemonstration = gr.TabbedInterface([single_query, input_file], 
                          ["Single Query","Input File Demo"])

MainPage= gr.TabbedInterface([ModelDemonstration,ExploratoryDataAnalysis], ["Model Demonstration","Exploratory Data Analysis"],theme=theme)

MainPage.launch()