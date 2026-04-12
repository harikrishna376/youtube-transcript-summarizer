import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
from urllib.parse import urlparse, parse_qs
from fpdf import FPDF
import re

# --- PAGE CONFIG ---
st.set_page_config(page_title="YouTube AI Summarizer", layout="centered")

# --- CUSTOM THEME ---
st.markdown("""
    <style>
    .main { background-color: #f0f4f8; }
    .stButton>button { background-color: #2e7d32; color: white; border-radius: 8px; width: 100%; }
    .stTextInput>div>div>input { border: 2px solid #2e7d32; }
    </style>
    """, unsafe_allow_html=True)

# --- CORE LOGIC ---
@st.cache_resource
def load_summarizer():
    # Using a fast, stable pipeline for abstractive summarization
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_summarizer()

def extract_video_id(url):
    """Robust extraction for all YouTube link types."""
    query = urlparse(url)
    if query.hostname == 'youtu.be': return query.path[1:]
    if query.hostname in ('www.youtube.com', 'youtube.com'):
        if query.path == '/watch': return parse_qs(query.query).get('v', [None])[0]
        if query.path[:7] == '/embed/': return query.path[7:]
        if query.path[:3] == '/v/': return query.path[3:]
    return None

# --- UI INTERFACE ---
st.title("📺 YouTube Transcript AI Summarizer")
st.info("Paste a YouTube URL below to generate an AI-powered summary and PDF report.")

video_url = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")

if st.button("🚀 Process Video & Generate PDF"):
    vid_id = extract_video_id(video_url)
    
    if not vid_id:
        st.error("❌ Invalid YouTube URL. Please check the link.")
    else:
        try:
            with st.spinner("Step 1: Extracting Transcript..."):
                # Fetching transcript [cite: 689]
                transcript_data = YouTubeTranscriptApi.get_transcript(vid_id)
                full_text = " ".join([t['text'] for t in transcript_data])
            
            with st.spinner("Step 2: AI Summarizing (Abstractive)..."):
                # Handling long transcripts by chunking (essential for execution)
                # Limits to first 1000 tokens for speed/stability
                summary_input = full_text[:3000] 
                summary_result = summarizer(summary_input, max_length=150, min_length=50, do_sample=False)
                final_summary = summary_result[0]['summary_text']
            
            # --- SUCCESS OUTPUT ---
            st.success("✅ Summarization Successful!")
            st.subheader("Abstractive Summary:")
            st.write(final_summary)

            # --- PDF GENERATION [cite: 691] ---
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, txt="Project: YouTube Transcript Summary", ln=True, align='C')
            pdf.ln(10)
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, txt=f"Video URL: {video_url}\n\nSummary:\n{final_summary}")
            
            pdf_bytes = pdf.output(dest='S').encode('latin-1', 'ignore')
            
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name=f"Summary_{vid_id}.pdf",
                mime="application/pdf"
            )

        except Exception as e:
            if "TranscriptsDisabled" in str(e):
                st.error("❌ Subtitles are disabled for this video. AI cannot extract text.")
            else:
                st.error(f"❌ Execution Error: {str(e)}")

# --- FOOTER ---
st.markdown("---")
st.caption("Developed based on: Voice Activity Detection and PDF Conversion Mini-Project [cite: 4]")
