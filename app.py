"""
YouTube Video Summarizer (Groq Edition — 100% Free)
====================================================
Uses Groq API with LLaMA 3 (FREE — 14,400 requests/day)
No credit card required.

Requirements:
    pip install yt-dlp openai-whisper groq \
                youtube-transcript-api python-dotenv streamlit

Also install ffmpeg binary:
    Windows : choco install ffmpeg -y
    macOS   : brew install ffmpeg
    Ubuntu  : sudo apt install ffmpeg

Setup:
    1. Get free API key at https://console.groq.com
    2. Add to .env file: GROQ_API_KEY=your_key_here

Usage:
    streamlit run app.py
"""

import os
import re
import smtplib
import tempfile
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import groq
import streamlit as st
import whisper
import yt_dlp
from dotenv import load_dotenv
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"   # free, fast, reliable
CHUNK_SIZE   = 3000                # words per chunk
WHISPER_MODEL = "base"             # tiny | base | small | medium


# ─────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────

def extract_video_id(url: str) -> str:
    """Parse video ID from any common YouTube URL format."""
    patterns = [
        r"(?:v=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:embed/)([A-Za-z0-9_-]{11})",
        r"(?:shorts/)([A-Za-z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    raise ValueError("Could not extract a valid YouTube video ID from the URL.")


def get_video_title(url: str) -> str:
    """Fetch the video title using yt-dlp (no download)."""
    opts = {"quiet": True, "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get("title", "Unknown Title")
    except Exception:
        return "Unknown Title"


# ─────────────────────────────────────────────
# Phase 2 — Transcript extraction
# ─────────────────────────────────────────────

def fetch_youtube_transcript(video_id: str) -> str:
    """Try YouTube built-in captions first (fastest, free)."""
    transcript_list = YouTubeTranscriptApi.get_transcript(
        video_id, languages=["en", "en-US", "en-GB"]
    )
    return " ".join(seg["text"] for seg in transcript_list)


def download_audio(url: str, output_path: str) -> str:
    """Download best audio stream with yt-dlp."""
    opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "quiet": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    return output_path if os.path.exists(output_path) else output_path + ".mp3"


def transcribe_audio(audio_path: str, language: str = "en") -> str:
    """Transcribe audio locally using OpenAI Whisper (no API key needed)."""
    model = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(audio_path, language=language)
    return result["text"]


def extract_transcript(url: str, language: str = "en", status_fn=None) -> str:
    """
    Smart extractor:
      1. Tries YouTube caption API (instant, free)
      2. Falls back to yt-dlp + Whisper (handles any video)
    """
    video_id = extract_video_id(url)

    # Strategy A — YouTube captions
    try:
        if status_fn:
            status_fn("Trying YouTube captions API…")
        transcript = fetch_youtube_transcript(video_id)
        if status_fn:
            status_fn("Captions found via YouTube API.")
        return transcript
    except (TranscriptsDisabled, NoTranscriptFound, Exception):
        pass

    # Strategy B — Whisper fallback
    if status_fn:
        status_fn("No captions found — downloading audio for Whisper transcription…")

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_file = os.path.join(tmpdir, "audio")
        audio_path = download_audio(url, audio_file)
        if status_fn:
            status_fn(f"Transcribing with Whisper '{WHISPER_MODEL}' model…")
        transcript = transcribe_audio(audio_path, language=language)

    return transcript


# ─────────────────────────────────────────────
# Phase 3 — AI summarization (Groq, map-reduce)
# ─────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Split transcript into word-count chunks."""
    words = text.split()
    return [
        " ".join(words[i : i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]


def summarize_chunk(client: groq.Groq, chunk: str, index: int, total: int) -> str:
    """Summarize one transcript chunk (map phase)."""
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=800,
        messages=[{
            "role": "user",
            "content": (
                f"This is part {index + 1} of {total} from a video transcript. "
                "Summarize it concisely, preserving key points, examples, "
                "and any mentioned timestamps or chapter headings.\n\n"
                f"Transcript chunk:\n{chunk}"
            ),
        }],
    )
    return response.choices[0].message.content


def build_final_summary(client: groq.Groq, partial_summaries: list[str], title: str) -> str:
    """Merge partial summaries into one structured report (reduce phase)."""
    combined = "\n\n---\n\n".join(partial_summaries)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": (
                f"You have received partial summaries of the YouTube video: \"{title}\".\n\n"
                "Produce a comprehensive, well-structured summary in markdown with these sections:\n\n"
                "## Overview\n"
                "(2-3 sentence high-level description)\n\n"
                "## Key Chapters\n"
                "(Bullet list of major topics with approximate timestamps if detectable)\n\n"
                "## Top Takeaways\n"
                "(5 most important insights or lessons)\n\n"
                "## Action Items\n"
                "(Concrete things the viewer can do based on this video)\n\n"
                "## Notable Quotes\n"
                "(2-3 memorable statements from the video)\n\n"
                f"Partial summaries:\n\n{combined}"
            ),
        }],
    )
    return response.choices[0].message.content


def answer_question(client: groq.Groq, transcript: str, question: str) -> str:
    """Answer a follow-up question grounded in the transcript."""
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=800,
        messages=[{
            "role": "user",
            "content": (
                f"Based on the following video transcript, answer this question:\n"
                f"Question: {question}\n\n"
                f"Transcript (excerpt):\n{transcript[:8000]}"
            ),
        }],
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────
# Phase 4 — Export
# ─────────────────────────────────────────────

def build_markdown_report(url: str, title: str, summary: str) -> str:
    """Compose the full markdown report."""
    date = datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        f"# {title}\n\n"
        f"> **Source:** {url}  \n"
        f"> **Generated:** {date}\n\n"
        "---\n\n"
        f"{summary}\n"
    )


def send_email_report(to: str, subject: str, body: str, smtp_user: str, smtp_pass: str) -> None:
    """Send the report via Gmail SMTP (requires App Password)."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = smtp_user
    msg["To"]      = to
    msg.attach(MIMEText(body, "plain"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, to, msg.as_string())


# ─────────────────────────────────────────────
# Phase 5 — Streamlit UI
# ─────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="YouTube Summarizer",
        page_icon="🎬",
        layout="wide",
    )

    # ── Sidebar ───────────────────────────────
    with st.sidebar:
        st.header("Settings")

        api_key = st.text_input(
            "Groq API Key (Free)",
            value=GROQ_API_KEY,
            type="password",
            help="Get your free key at console.groq.com",
        )
        st.caption("Free tier: 14,400 requests/day — no credit card needed.")

        language = st.selectbox(
            "Transcript language (Whisper fallback)",
            ["en", "hi", "te", "ta", "fr", "de", "es", "ja", "zh"],
            index=0,
        )

        st.selectbox(
            "Whisper model size",
            ["tiny", "base", "small", "medium"],
            index=1,
            help="Larger = more accurate but slower",
        )

        st.divider()
        st.subheader("Email delivery (optional)")
        email_to  = st.text_input("Send report to",     placeholder="you@example.com")
        smtp_user = st.text_input("Gmail address",      placeholder="sender@gmail.com")
        smtp_pass = st.text_input("Gmail App Password", type="password")

    # ── Main area ─────────────────────────────
    st.title("YouTube Video Summarizer")
    st.caption("Paste any YouTube URL — get a free AI-powered structured summary.")

    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    run = st.button("Summarize", type="primary")

    if not run or not url:
        st.info("Enter a YouTube URL above and click Summarize.")
        return

    if not api_key:
        st.error("Please enter your Groq API key in the sidebar. Get one free at console.groq.com")
        return

    # Initialize Groq client
    client = groq.Groq(api_key=api_key)

    # Session state
    for key in ["summary", "transcript", "title", "report_md"]:
        if key not in st.session_state:
            st.session_state[key] = None

    status = st.empty()

    # ── Step 1: Video title ────────────────────
    with st.spinner("Fetching video info…"):
        try:
            title = get_video_title(url)
            st.session_state.title = title
        except Exception as e:
            st.error(f"Could not fetch video info: {e}")
            return

    st.subheader(f"📺 {title}")

    # ── Step 2: Extract transcript ─────────────
    with st.spinner("Extracting transcript…"):
        try:
            transcript = extract_transcript(
                url,
                language=language,
                status_fn=lambda msg: status.caption(msg),
            )
            st.session_state.transcript = transcript
            status.empty()
        except Exception as e:
            st.error(f"Transcript extraction failed: {e}")
            return

    word_count = len(transcript.split())
    chunks     = chunk_text(transcript, CHUNK_SIZE)
    st.caption(f"Transcript: {word_count:,} words → {len(chunks)} chunk(s)")

    # ── Step 3: Summarize chunks ───────────────
    partial_summaries = []
    prog = st.progress(0, text="Summarizing…")

    for i, chunk in enumerate(chunks):
        try:
            partial = summarize_chunk(client, chunk, i, len(chunks))
            partial_summaries.append(partial)
        except Exception as e:
            st.warning(f"Chunk {i+1} failed: {e}")
        prog.progress(
            (i + 1) / len(chunks),
            text=f"Summarized chunk {i+1} of {len(chunks)}"
        )

    prog.empty()

    # ── Step 4: Final summary ──────────────────
    with st.spinner("Building final structured report…"):
        try:
            summary = build_final_summary(client, partial_summaries, title)
            st.session_state.summary = summary
        except Exception as e:
            st.error(f"Final summary failed: {e}")
            return

    # ── Step 5: Display ────────────────────────
    st.divider()
    st.markdown(summary)

    report_md = build_markdown_report(url, title, summary)
    st.session_state.report_md = report_md

    # ── Step 6: Download & email ───────────────
    st.divider()
    dl_col, email_col = st.columns(2)

    with dl_col:
        st.download_button(
            "Download Report (.md)",
            data=report_md,
            file_name=f"{re.sub(r'[^a-zA-Z0-9]', '_', title)[:50]}_summary.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with email_col:
        if st.button("Send via Email", use_container_width=True):
            if email_to and smtp_user and smtp_pass:
                try:
                    send_email_report(
                        to=email_to,
                        subject=f"Summary: {title}",
                        body=report_md,
                        smtp_user=smtp_user,
                        smtp_pass=smtp_pass,
                    )
                    st.success(f"Report sent to {email_to}")
                except Exception as e:
                    st.error(f"Email failed: {e}")
            else:
                st.warning("Fill in email settings in the sidebar first.")

    # ── Step 7: Q&A ────────────────────────────
    st.divider()
    st.subheader("Ask a question about this video")
    question = st.text_input(
        "Your question",
        placeholder='e.g. "What did they say about machine learning?"'
    )
    if st.button("Ask") and question:
        with st.spinner("Thinking…"):
            try:
                answer = answer_question(client, st.session_state.transcript, question)
                st.markdown(f"**Answer:** {answer}")
            except Exception as e:
                st.error(f"Q&A failed: {e}")


if __name__ == "__main__":
    main()
