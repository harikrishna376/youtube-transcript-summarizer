import streamlit as st
import streamlit.components.v1 as components
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import T5ForConditionalGeneration, T5Tokenizer
from urllib.parse import urlparse, parse_qs
from fpdf import FPDF
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="YouTube Summarizer Pro", layout="centered")

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    return model, tokenizer

model, tokenizer = load_model()

def get_video_id(url):
    parsed_url = urlparse(url)
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query)['v'][0]
    return None

def generate_summary(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- UI COMPONENT ---
with open("templates/popup.html", "r") as f:
    html_code = f.read()
components.html(html_code, height=400)

# Streamlit-native input for logic handling
url = st.text_input("Confirm URL for Processing:", placeholder="Paste the same link here to trigger AI")

if st.button("Generate & Download PDF"):
    video_id = get_video_id(url)
    if not video_id:
        st.error("Please enter a valid YouTube URL.")
    else:
        with st.spinner("Fetching transcript and summarizing..."):
            try:
                # 1. Fetch Transcript [cite: 681]
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                full_text = " ".join([t['text'] for t in transcript_list])
                
                # 2. Summarize [cite: 690]
                summary = generate_summary(full_text)
                
                st.subheader("Summary Result:")
                st.write(summary)
                
                # 3. Create PDF [cite: 691]
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(200, 10, txt="YouTube Video Summary", ln=True, align='C')
                pdf.ln(10)
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 10, txt=summary)
                
                pdf_output = pdf.output(dest='S').encode('latin-1')
                
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_output,
                    file_name=f"Summary_{video_id}.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"Error: {str(e)}. (Ensure the video has captions enabled)")
