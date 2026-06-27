"""
Voice Assistant Module
======================
Multilingual voice input and text-to-speech for the Smart Meal Planner.

Supports:
  - English (en-IN, en-US)
  - Hindi (hi-IN)
  - Marathi (mr-IN)
  - Tamil (ta-IN)
  - Telugu (te-IN)
  - Bengali (bn-IN)
  - Gujarati (gu-IN)
  - Kannada (kn-IN)
  - Punjabi (pa-IN)
  - Malayalam (ml-IN)

Uses:
  1. Browser Web Speech API (real-time, no API key)
  2. Python SpeechRecognition + Google STT (audio file fallback)
"""

import io
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ── Supported languages ────────────────────────────────────────────────────────
LANGUAGES = {
    "🇬🇧 English (India)":   "en-IN",
    "🇺🇸 English (US)":       "en-US",
    "🇮🇳 Hindi":              "hi-IN",
    "🏔️ Marathi":             "mr-IN",
    "🌊 Tamil":               "ta-IN",
    "🌿 Telugu":              "te-IN",
    "🌸 Bengali":             "bn-IN",
    "🦁 Gujarati":            "gu-IN",
    "⚔️ Kannada":             "kn-IN",
    "🌴 Malayalam":           "ml-IN",
    "🥁 Punjabi":             "pa-IN",
}


def transcribe_audio(audio_bytes: bytes, language_code: str = "en-IN") -> Tuple[str, Optional[str]]:
    """
    Transcribe audio bytes using Google Speech Recognition.

    Parameters
    ----------
    audio_bytes   : raw WAV/WebM bytes from st.audio_input()
    language_code : BCP-47 language code e.g. 'hi-IN'

    Returns
    -------
    (transcript, error_message)
    """
    try:
        import speech_recognition as sr
    except ImportError:
        return "", "SpeechRecognition package not installed."

    try:
        from pydub import AudioSegment
        # Convert WebM/OGG → WAV for SpeechRecognition
        try:
            audio_seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
            wav_io = io.BytesIO()
            audio_seg.export(wav_io, format="wav")
            wav_io.seek(0)
            audio_bytes = wav_io.read()
        except Exception:
            pass  # Try raw bytes anyway

        r = sr.Recognizer()
        r.energy_threshold = 300
        r.dynamic_energy_threshold = True

        audio_data = sr.AudioData(audio_bytes, sample_rate=16000, sample_width=2)
        text = r.recognize_google(audio_data, language=language_code)
        return text, None
    except Exception as e:
        err = str(e)
        if "recognition connection failed" in err.lower():
            return "", "Network error — check your internet connection."
        if "could not understand" in err.lower() or "unintelligible" in err.lower():
            return "", "Could not understand — please speak more clearly or try again."
        return "", f"Speech recognition error: {err}"


def get_voice_widget_html(lang_code: str = "en-IN", widget_id: str = "voice_1") -> str:
    """
    Generate a self-contained HTML/JS voice input widget using the
    browser's native Web Speech API.

    The widget posts recognized text to the parent Streamlit window
    via window.parent.postMessage().
    """
    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; font-family: 'Inter', sans-serif; }}
  body {{ background: transparent; }}

  .voice-widget {{
    background: linear-gradient(135deg, rgba(26,32,53,0.95), rgba(20,27,45,0.98));
    border: 1px solid rgba(139,92,246,0.3);
    border-radius: 16px;
    padding: 16px;
    color: #f1f5f9;
  }}

  .mic-area {{
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 12px;
  }}

  .mic-btn {{
    width: 56px; height: 56px;
    border-radius: 50%;
    background: linear-gradient(135deg, #7c3aed, #8b5cf6);
    border: none;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    box-shadow: 0 4px 20px rgba(124,58,237,0.45);
    transition: all 0.25s ease;
    flex-shrink: 0;
    position: relative;
  }}
  .mic-btn:hover {{ transform: scale(1.07); box-shadow: 0 6px 28px rgba(124,58,237,0.65); }}
  .mic-btn.listening {{
    background: linear-gradient(135deg, #dc2626, #ef4444);
    box-shadow: 0 0 0 0 rgba(220,38,38,0.4);
    animation: pulse 1.2s infinite;
  }}
  .mic-btn.listening::after {{
    content: '';
    position: absolute;
    width: 100%; height: 100%;
    border-radius: 50%;
    border: 2px solid rgba(239,68,68,0.5);
    animation: ripple 1.2s infinite;
  }}
  @keyframes pulse {{
    0%   {{ box-shadow: 0 0 0 0 rgba(220,38,38,0.5); }}
    70%  {{ box-shadow: 0 0 0 14px rgba(220,38,38,0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(220,38,38,0); }}
  }}
  @keyframes ripple {{
    0%   {{ transform: scale(1); opacity: 1; }}
    100% {{ transform: scale(1.5); opacity: 0; }}
  }}

  .mic-info h4 {{
    font-size: 0.88rem;
    font-weight: 700;
    color: #c4b5fd;
    margin-bottom: 3px;
  }}
  .mic-info .status {{
    font-size: 0.75rem;
    color: #64748b;
    transition: color 0.2s;
  }}
  .mic-info .status.active {{ color: #34d399; }}
  .mic-info .status.error  {{ color: #f87171; }}

  .result-box {{
    background: rgba(13,17,23,0.7);
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 10px;
    padding: 10px 14px;
    min-height: 46px;
    font-size: 0.86rem;
    color: #e2e8f0;
    margin-bottom: 10px;
    line-height: 1.5;
    word-break: break-word;
  }}
  .result-box.empty {{ color: #64748b; font-style: italic; }}

  .actions {{
    display: flex;
    gap: 8px;
  }}
  .copy-btn, .clear-btn {{
    flex: 1;
    padding: 7px 10px;
    border-radius: 8px;
    border: none;
    font-size: 0.78rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
  }}
  .copy-btn {{
    background: linear-gradient(135deg, #7c3aed, #8b5cf6);
    color: white;
    box-shadow: 0 2px 10px rgba(124,58,237,0.3);
  }}
  .copy-btn:hover {{ transform: translateY(-1px); box-shadow: 0 4px 14px rgba(124,58,237,0.5); }}
  .copy-btn:disabled {{ opacity: 0.4; cursor: not-allowed; transform: none; }}
  .clear-btn {{
    background: rgba(30,45,69,0.8);
    color: #94a3b8;
    border: 1px solid rgba(139,92,246,0.2);
  }}
  .clear-btn:hover {{ background: rgba(30,45,69,1); color: #f1f5f9; }}

  .support-note {{
    font-size: 0.7rem;
    color: #475569;
    margin-top: 10px;
    text-align: center;
  }}
  .support-note.ok {{ color: #34d399; }}
  .support-note.no {{ color: #f87171; }}
</style>
</head>
<body>
<div class="voice-widget">
  <div class="mic-area">
    <button class="mic-btn" id="micBtn_{widget_id}" onclick="toggleListening_{widget_id}()">🎤</button>
    <div class="mic-info">
      <h4>🗣️ Speak your ingredients</h4>
      <div class="status" id="status_{widget_id}">Click the mic to start</div>
    </div>
  </div>

  <div class="result-box empty" id="result_{widget_id}">
    Your speech will appear here…
  </div>

  <div class="actions">
    <button class="copy-btn" id="copyBtn_{widget_id}" onclick="copyAndSend_{widget_id}()" disabled>
      📋 Copy to clipboard
    </button>
    <button class="clear-btn" onclick="clearResult_{widget_id}()">✕ Clear</button>
  </div>

  <div class="support-note" id="supportNote_{widget_id}">Checking browser support…</div>
</div>

<script>
(function() {{
  const LANG = "{lang_code}";
  const ID   = "{widget_id}";

  let recognition = null;
  let isListening = false;
  let fullTranscript = "";

  const micBtn    = document.getElementById("micBtn_"    + ID);
  const statusEl  = document.getElementById("status_"   + ID);
  const resultEl  = document.getElementById("result_"   + ID);
  const copyBtn   = document.getElementById("copyBtn_"  + ID);
  const noteEl    = document.getElementById("supportNote_" + ID);

  // Check browser support
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRec) {{
    noteEl.textContent = "❌ Voice input not supported in this browser. Use Chrome or Edge.";
    noteEl.className = "support-note no";
    micBtn.disabled = true;
    micBtn.style.opacity = "0.4";
    return;
  }}
  noteEl.textContent = "✅ Voice ready — works best in Chrome/Edge";
  noteEl.className = "support-note ok";

  recognition = new SpeechRec();
  recognition.lang = LANG;
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.maxAlternatives = 1;

  recognition.onstart = function() {{
    isListening = true;
    micBtn.classList.add("listening");
    micBtn.innerHTML = "⏹";
    statusEl.textContent = "Listening… speak now";
    statusEl.className = "status active";
  }};

  recognition.onresult = function(event) {{
    let interim = "";
    let final   = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {{
      if (event.results[i].isFinal) {{
        final += event.results[i][0].transcript;
      }} else {{
        interim += event.results[i][0].transcript;
      }}
    }}
    if (final) fullTranscript += (fullTranscript ? ", " : "") + final.trim();
    const display = fullTranscript + (interim ? " <em style='color:#64748b'>" + interim + "</em>" : "");
    resultEl.innerHTML = display || "<em style='color:#64748b'>Listening…</em>";
    resultEl.className = "result-box";
    copyBtn.disabled = !fullTranscript;

    // Post to parent Streamlit frame
    if (fullTranscript) {{
      try {{
        window.parent.postMessage({{
          type: "VOICE_RESULT",
          text: fullTranscript,
          lang: LANG
        }}, "*");
      }} catch(e) {{}}
    }}
  }};

  recognition.onerror = function(event) {{
    const msgs = {{
      "no-speech":         "No speech detected — try again",
      "audio-capture":     "Microphone not accessible — check browser permissions",
      "not-allowed":       "Microphone permission denied — allow mic in browser settings",
      "network":           "Network error — check your internet connection",
      "aborted":           "Stopped",
    }};
    statusEl.textContent = msgs[event.error] || "Error: " + event.error;
    statusEl.className = "status error";
    stopListening();
  }};

  recognition.onend = function() {{
    if (isListening) {{
      // Auto-restart for continuous listening
      try {{ recognition.start(); }} catch(e) {{ stopListening(); }}
    }}
  }};

  window["toggleListening_" + ID] = function() {{
    if (isListening) stopListening();
    else startListening();
  }};

  function startListening() {{
    fullTranscript = "";
    resultEl.innerHTML = "<em style='color:#64748b'>Listening…</em>";
    resultEl.className = "result-box";
    copyBtn.disabled = true;
    recognition.lang = LANG;
    try {{ recognition.start(); }}
    catch(e) {{ statusEl.textContent = "Could not start mic: " + e.message; statusEl.className = "status error"; }}
  }}

  function stopListening() {{
    isListening = false;
    micBtn.classList.remove("listening");
    micBtn.innerHTML = "🎤";
    statusEl.textContent = fullTranscript ? "Done! Copy the text below." : "Click mic to start";
    statusEl.className = "status";
    try {{ recognition.stop(); }} catch(e) {{}}
  }}

  window["copyAndSend_" + ID] = function() {{
    if (!fullTranscript) return;
    navigator.clipboard.writeText(fullTranscript).then(() => {{
      copyBtn.textContent = "✅ Copied!";
      setTimeout(() => {{ copyBtn.innerHTML = "📋 Copy to clipboard"; }}, 2000);
    }}).catch(() => {{
      // Fallback
      const ta = document.createElement("textarea");
      ta.value = fullTranscript;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      copyBtn.textContent = "✅ Copied!";
      setTimeout(() => {{ copyBtn.innerHTML = "📋 Copy to clipboard"; }}, 2000);
    }});
  }};

  window["clearResult_" + ID] = function() {{
    fullTranscript = "";
    resultEl.innerHTML = "Your speech will appear here…";
    resultEl.className = "result-box empty";
    copyBtn.disabled = true;
    if (isListening) stopListening();
    statusEl.textContent = "Click the mic to start";
    statusEl.className = "status";
  }};
}})();
</script>
</body>
</html>
"""


def speak_text_html(text: str, lang_code: str = "en-IN", rate: float = 0.95) -> str:
    """
    Generate an HTML snippet that speaks `text` using the browser's
    Web Speech Synthesis API (TTS). Auto-plays when rendered.
    """
    # Escape for JS string
    safe_text = text.replace("'", "\\'").replace("\n", " ")
    return f"""
<script>
(function() {{
  if (!window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance('{safe_text}');
  u.lang  = '{lang_code}';
  u.rate  = {rate};
  u.pitch = 1.05;
  // Wait for voices to load
  function speak() {{ window.speechSynthesis.speak(u); }}
  if (window.speechSynthesis.getVoices().length) speak();
  else window.speechSynthesis.onvoiceschanged = speak;
}})();
</script>
"""


def get_read_aloud_button_html(text: str, lang_code: str = "en-IN", rate: float = 0.95) -> str:
    """
    Generate an HTML snippet rendering a styled 'Read Aloud' button that is
    clicked by the user inside the iframe context to bypass browser TTS autoplay restrictions.
    """
    # Escape single quotes and backslashes for JS safety
    safe_text = text.replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ")
    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{
    margin: 0;
    padding: 0;
    background: transparent;
    overflow: hidden;
  }}
  .read-aloud-btn {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    background: rgba(212, 168, 83, 0.08);
    border: 1px solid rgba(212, 168, 83, 0.20);
    color: #F5F1E8;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 0.85rem;
    font-weight: 500;
    padding: 8px 16px;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.25s ease;
    height: 38px;
    box-sizing: border-box;
    width: 100%;
    max-width: 150px;
    outline: none;
  }}
  .read-aloud-btn:hover {{
    background: rgba(212, 168, 83, 0.16);
    border-color: rgba(212, 168, 83, 0.35);
    transform: translateY(-1px);
  }}
  .read-aloud-btn:active {{
    transform: translateY(0);
  }}
  .read-aloud-btn.speaking {{
    background: rgba(239, 68, 68, 0.15);
    border-color: rgba(239, 68, 68, 0.4);
    color: #ef4444;
  }}
</style>
</head>
<body>
<button class="read-aloud-btn" id="speakBtn" onclick="toggleSpeak()">🔊 Read Aloud</button>

<script>
  const textToSpeak = '{safe_text}';
  const langCode = '{lang_code}';
  const rateVal = {rate};
  
  let utterance = null;
  let isSpeaking = false;
  
  function toggleSpeak() {{
    const btn = document.getElementById("speakBtn");
    if (!window.speechSynthesis) {{
      alert("Speech synthesis is not supported in this browser.");
      return;
    }}
    
    if (isSpeaking) {{
      window.speechSynthesis.cancel();
      isSpeaking = false;
      btn.innerHTML = "🔊 Read Aloud";
      btn.classList.remove("speaking");
    }} else {{
      window.speechSynthesis.cancel();
      utterance = new SpeechSynthesisUtterance(textToSpeak);
      utterance.lang = langCode;
      utterance.rate = rateVal;
      utterance.pitch = 1.05;
      
      utterance.onend = function() {{
        isSpeaking = false;
        btn.innerHTML = "🔊 Read Aloud";
        btn.classList.remove("speaking");
      }};
      
      utterance.onerror = function(e) {{
        console.error("Speech error", e);
        isSpeaking = false;
        btn.innerHTML = "🔊 Read Aloud";
        btn.classList.remove("speaking");
      }};
      
      isSpeaking = true;
      btn.innerHTML = "⏹ Stop Aloud";
      btn.classList.add("speaking");
      window.speechSynthesis.speak(utterance);
    }}
  }}
</script>
</body>
</html>
"""

