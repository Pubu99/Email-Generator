# app/frontend.py

import streamlit as st
import requests

API_URL = "http://localhost:8000"  # Change if your backend is hosted elsewhere

st.title("AI-Powered Email Categorizer & Dynamic Reply Generator")

# Step 1: Paste Email
email_text = st.text_area("📤 Paste your email here", height=200)

if email_text:
    # Step 2: Get Intent Classification
    classify_resp = requests.post(f"{API_URL}/classify", json={"email_text": email_text}).json()
    category = classify_resp.get("intent", "")
    confidence = classify_resp.get("confidence", 0)
    st.markdown(f"**Predicted Category:** {category} (Confidence: {confidence:.2f})")

    # Step 3: Extract Entities
    entities_resp = requests.post(f"{API_URL}/extract_entities", json={"email_text": email_text}).json()
    entities = entities_resp.get("entities", {})

    st.markdown("### Extracted Entities (Edit if needed):")
    # Editable entities fields
    person = st.text_input("Person", value=", ".join(entities.get("PERSON", [])))
    date = st.text_input("Date", value=", ".join(entities.get("DATE", [])))
    time = st.text_input("Time", value=", ".join(entities.get("TIME", [])))
    location = st.text_input("Location", value=", ".join(entities.get("GPE", [])))

    # Prepare entities dict for sending
    entities_for_api = {
        "PERSON": [p.strip() for p in person.split(",") if p.strip()],
        "DATE": [d.strip() for d in date.split(",") if d.strip()],
        "TIME": [t.strip() for t in time.split(",") if t.strip()],
        "GPE": [l.strip() for l in location.split(",") if l.strip()]
    }

    if st.button("🖱️ Generate Reply"):
        with st.spinner("Generating reply..."):
            payload = {
                "email_text": email_text,
                "category": category,
                "entities": entities_for_api
            }
            generate_resp = requests.post(f"{API_URL}/generate", json=payload).json()
            reply = generate_resp.get("reply", "")
            st.markdown("### 💡 Generated Reply")
            reply_text = st.text_area("Edit your reply here", value=reply, height=200)
            st.markdown("You can copy and paste the reply above.")
