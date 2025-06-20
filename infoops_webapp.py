import streamlit as st
from transformers import pipeline
import re

# --- Setup ---
@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = load_classifier()

INFO_OPS = {
    "Nationalist Mobilization": ['century of humiliation', 'sovereignty', 'united', 'foreign interference', 'national pride', 'patriot', 'homeland'],
    "Demoralization": ['failed', 'nothing will ever change', 'system is broken', 'hope is lost', 'decline', 'collapse', 'weak', 'incompetence'],
    "Rage Farming & Polarization": ['they', 'us', 'them', 'angry', "can't trust", 'division', 'protest', 'outrage', 'tribalism'],
    "False Reassurance / Legitimacy Campaigns": ['win-win', 'generosity', 'vision', 'benevolent', 'cooperation', 'trust', 'legitimacy', 'whitewashing'],
    "Distraction / Obfuscation": ['look over there', 'while everyone is talking', 'celebrity scandal', 'smoke screen', 'red herring', 'contradictory theories'],
}
labels = list(INFO_OPS.keys())
THRESHOLD = 0.4

def highlight_keywords(text, keywords):
    text_lower = text.lower()
    found = []
    for kw in keywords:
        if kw in text_lower:
            found.append(kw)
    return found

# --- Streamlit UI ---
st.title("Information Operations Detector")
st.write("Type or paste a tweet, post, or any text below. The app will detect the most likely information operation tactics.")

user_text = st.text_area("Enter text to analyze", height=150)

if user_text.strip() and st.button("Analyze"):
    with st.spinner("Analyzing..."):
        result = classifier(user_text, candidate_labels=labels, multi_label=True)
        # Filter results above threshold
        filtered = [
            (label, score) for label, score in zip(result['labels'], result['scores']) if score >= THRESHOLD
        ]
        # Sort by score descending and take up to 3
        filtered = sorted(filtered, key=lambda x: x[1], reverse=True)[:3]
        if filtered:
            for label, score in filtered:
                matched_keywords = highlight_keywords(user_text, INFO_OPS[label])
                st.markdown(f"### {label} ({score:.2f} confidence)")
                if matched_keywords:
                    st.write(f"**Matched keywords:** {', '.join(matched_keywords)}")
                else:
                    st.write("_No specific keywords matched; classification based on model understanding._")
        else:
            st.info("No strong info op detected (confidence below threshold).")