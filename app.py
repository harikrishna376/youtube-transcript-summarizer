import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
from urllib.parse import urlparse, parse_qs
from fpdf import FPDF

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Video Summarizer", layout="centered")

# --- STYLE ---
st.markdown("""
    <style>
    .stButton>button { background-color: #2e7d32; color: white; width: 100%; border-radius: 10px; }
    .stTextInput>div>div>input { border: 2px solid #2e7d32; }
    </style>
    """, unsafe_allow_html=True)

# --- CACHED AI MODEL ---
@st.cache_resource
def load_ai():
    # Loading the Abstractive Summarizer [cite: 162, 498]
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_ai()

def get_id(url):
    """Extracts the unique video ID for the API call."""
    u_pars = urlparse(url)
    if u_pars.hostname == 'youtu.be': return u_pars.path[1:]
    if u_pars.hostname in ('www.youtube.com', 'youtube.com'):
        if u_pars.path == '/watch': return parse_qs(u_pars.query).get('v', [None])[0]
    return None

# --- UI CONTENT [cite: 457-459] ---
st.title("📺 YouTube Transcript AI Summarizer")
st.subheader("Generate Abstractive Summaries & PDF Reports")

link = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")

if st.button("🚀 Process Video & Generate PDF"):
    v_id = get_id(link)
    
    if not v_id:
        st.error("❌ Link format not recognized.")
    else:
        try:
            with st.spinner("Step 1: Extracting Transcript..."):
                # FIXED CALL: Calling the method on the class directly
                data = YouTubeTranscriptApi.get_transcript(v_id)
                text = " ".join([i['text'] for i in data])
            
            with st.spinner("Step 2: AI Generating Summary..."):
                # Limit input to avoid RAM overflow on Streamlit Cloud
                clean_text = text[:3000] 
                res = summarizer(clean_text, max_length=130, min_length=30, do_sample=False)
                summary = res[0]['summary_text']
            
            st.success("✅ Content Summarized!")
            st.info(summary)

            # --- PDF CREATION [cite: 294, 691] ---
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, txt="Mini Project: Video Summary Report", ln=True, align='C')
            pdf.ln(10)
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, txt=f"Source: {link}\n\nSummary:\n{summary}")
            
            pdf_out = pdf.output(dest='S').encode('latin-1', 'ignore')
            st.download_button(label="📥 Download PDF Summary", data=pdf_out, file_name="YouTube_Summary.pdf", mime="application/pdf")

        except Exception as e:
            st.error(f"Execution Error: {str(e)}")
            st.warning("Note: Ensure the video has closed captions/subtitles enabled.")
