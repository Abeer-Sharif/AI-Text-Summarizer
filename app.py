from fastapi import FastAPI, Request
from pydantic import BaseModel

from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


# Initialize FastAPI application
app = FastAPI(
    title="Text-Summarizer",
    description="Text summarization using T5",
    version="1.0"
)


# Load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained(
    "./saved_summary_model"
)

tokenizer = T5Tokenizer.from_pretrained(
    "./saved_summary_model"
)


# Define device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# Move model to selected device
model.to(device)


# Templating
templates = Jinja2Templates(directory=".")


# Input schema
class DialogueInput(BaseModel):
    dialogue: str

#define the clean data function
def clean_data(text):
    text = re.sub(r"\r\n"," ",text) #remove lines
    text = re.sub(r"\s+", " ", text) # remove spaces
    text = re.sub(r"<.*?>"," ", text) #remove html tags
    text = text.strip().lower()
    return text
#summariztion func
def summarize_dialogue(dialogue : str) -> str:
    #clean the data
    dialogue = clean_data(dialogue)
    #tokenize
    inputs = tokenizer(
        dialogue,
        padding = "max_length",
        max_length = 512,
        truncation = True,
        return_tensors = "pt"#return as pytorch tensors
    )
    inputs = {
    key: value.to(device)
    for key, value in inputs.items()
    }
    #generate the summary => token ids
    targets = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask = inputs["attention_mask"],
        max_length = 150,
        num_beams = 4,
        early_stopping = True
    )
    #decode the token ids : convert to summary
    summary = tokenizer.decode(targets[0], skip_special_tokens=True)
    return summary

#API Endpoints
@app.post("/summarize/")
async def summarize(dialogue_input : DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )
    
