import gradio as gr
from transformers import BartTokenizer, BartForConditionalGeneration

model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

def bart_summarizer(input_text):
    input_text = tokenizer.batch_encode_plus([input_text], max_length=1024, return_tensors='pt')
    summary_ids = model.generate(input_text['input_ids'], num_beams=4, max_length=100, early_stopping=True)
    output = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    return output[0]

interface = gr.Interface(
    fn=bart_summarizer,
    inputs=gr.Textbox(lines=7, placeholder="Enter some long text here"),
    outputs="textbox",
    live=True
)


interface.launch(
	share=True
)