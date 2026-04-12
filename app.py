import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from urllib.parse import urlparse, parse_qs
from fpdf import FPDF

# --- PAGE CONFIG ---
st.set_page_config(page_title="YouTube AI Summarizer", layout="centered")

# --- MODEL LOADING (Manual Architecture) ---
@st.cache_resource
def load_model_and_tokenizer():
    # Loading BART model architecture specifically as mentioned in report
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

def extract_video_id(url):
    query = urlparse(url)
    if query.hostname == 'youtu.be': return query.path[1:]
    if query.hostname in ('www.youtube.com', 'youtube.com'):
        if query.path == '/watch': return parse_qs(query.query).get('v', [None])[0]
    return None

# --- UI INTERFACE ---
st.title("📺 YouTube Transcript AI Summarizer")
st.markdown("### Generate Abstractive Summaries & PDF Reports")

video_url = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")

if st.button("🚀 Process Video & Generate PDF"):
    vid_id = extract_video_id(video_url)
    
    if not vid_id:
        st.error("❌ Invalid URL.")
    else:
        try:
            with st.spinner("Extracting Transcript..."):
                transcript_data = YouTubeTranscriptApi.get_transcript(vid_id)
                full_text = " ".join([t['text'] for t in transcript_data])
            
            with st.spinner("AI Generating Summary..."):
                # Manual Inference Logic (Avoids pipeline KeyError)
                inputs = tokenizer([full_text[:3000]], max_length=1024, return_tensors="pt", truncation=True)
                summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=150, early_stopping=True)
                final_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            st.success("✅ Summarization Successful!")
            st.write(final_summary)

            # --- PDF GENERATION [cite: 691-692] ---
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, txt=f"Summary for Video: {video_url}\n\n{final_summary}")
            
            pdf_bytes = pdf.output(dest='S').encode('latin-1', 'ignore')
            st.download_button(label="📥 Download PDF", data=pdf_bytes, file_name="summary.pdf", mime="application/pdf")

        except Exception as e:
            st.error(f"Execution Error: {str(e)}")
