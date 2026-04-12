import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
from urllib.parse import urlparse, parse_qs
from fpdf import FPDF

# --- PAGE CONFIG ---
st.set_page_config(page_title="YouTube AI Summarizer", layout="centered")

# --- CUSTOM THEME ---
st.markdown("""
    <style>
    .main { background-color: #f0f4f8; }
    .stButton>button { background-color: #2e7d32; color: white; border-radius: 8px; width: 100%; }
    </style>
    """, unsafe_allow_html=True)

# --- CORE LOGIC ---
@st.cache_resource
def load_summarizer():
    try:
        # Using a reliable model mentioned in the methodology [cite: 304, 341]
        return pipeline("summarization", model="facebook/bart-large-cnn")
    except Exception as e:
        st.error(f"Failed to load AI model: {e}")
        return None

summarizer = load_summarizer()

def extract_video_id(url):
    query = urlparse(url)
    if query.hostname == 'youtu.be': return query.path[1:]
    if query.hostname in ('www.youtube.com', 'youtube.com'):
        if query.path == '/watch': return parse_qs(query.query).get('v', [None])[0]
    return None

# --- UI INTERFACE [cite: 457-459] ---
st.title("📺 YouTube Transcript AI Summarizer")
video_url = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")

if st.button("🚀 Process Video & Generate PDF"):
    vid_id = extract_video_id(video_url)
    
    if not vid_id:
        st.error("❌ Invalid YouTube URL.")
    elif summarizer is None:
        st.error("❌ Summarization engine is not available.")
    else:
        try:
            with st.spinner("Extracting Transcript..."):
                # Fetching transcript logic [cite: 681, 689]
                transcript_data = YouTubeTranscriptApi.get_transcript(vid_id)
                full_text = " ".join([t['text'] for t in transcript_data])
            
            with st.spinner("AI Summarizing..."):
                # Abstractive summarization logic [cite: 162-164, 498-499]
                summary_input = full_text[:2000] # Input limit for stability
                summary_result = summarizer(summary_input, max_length=150, min_length=50, do_sample=False)
                final_summary = summary_result[0]['summary_text']
            
            st.success("✅ Successful!")
            st.subheader("Summary:")
            st.write(final_summary)

            # --- PDF GENERATION [cite: 294-295, 691] ---
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, txt=f"Summary for: {video_url}\n\n{final_summary}")
            
            pdf_bytes = pdf.output(dest='S').encode('latin-1', 'ignore')
            st.download_button(label="📥 Download PDF", data=pdf_bytes, file_name="summary.pdf", mime="application/pdf")

        except Exception as e:
            st.error(f"Execution Error: {str(e)}")
