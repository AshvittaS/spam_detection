import gradio as gr
from src.predict import predict_spam

demo = gr.Interface(
    fn=predict_spam,
    inputs=gr.Textbox(lines=5, placeholder="Enter a message"),
    outputs=gr.Textbox(label="Prediction"),
    title="Spam Detection"
)