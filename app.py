import streamlit as st
from src.predict import predict_ai_content
from src.pdf_extractor import extract_text_from_pdf
from src.web_similarity import check_web_similarity
from src.stylometric_features import extract_stylometric_features

# PDF Report Imports
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
import io
import datetime

st.set_page_config(page_title="Academic Integrity Analysis System", layout="wide")

st.title("📚 Academic Integrity Analysis System")
st.markdown("AI Authorship Detection & Web Similarity Assessment")

st.subheader("📄 Document Input")

user_text = st.text_area("Enter assignment text below:")

uploaded_file = st.file_uploader("Or upload a PDF document", type=["pdf"])

if uploaded_file:
    extracted_text = extract_text_from_pdf(uploaded_file)
    if extracted_text:
        st.success("PDF text extracted successfully.")
        user_text = extracted_text
    else:
        st.error("Could not extract text from PDF.")

# ----------------------------------------------------------------
# MAIN ANALYSIS
# ----------------------------------------------------------------
if user_text and len(user_text.strip()) > 20:

    # ==============================
    # 🤖 AI AUTHORSHIP DETECTION
    # ==============================
    classification, confidence = predict_ai_content(user_text)

    st.subheader("🤖 AI Authorship Assessment")

    if "AI-Generated" in classification:
        st.error(f"🔴 {classification}")
    elif "Mixed" in classification:
        st.warning(f"🟡 {classification}")
    else:
        st.success(f"🟢 {classification}")

    st.write(f"Confidence Score: {confidence}")

    # ==============================
    # 🌐 WEB SIMILARITY CHECK
    # ==============================
    st.subheader("📊 Plagiarism Analysis")

    top_sources = check_web_similarity(user_text)

    if top_sources:
        similarity_score = float(top_sources[0]["similarity"])
    else:
        similarity_score = 0.0

    st.write(f"Plagiarism Rate: {round(similarity_score, 2)}%")

    if similarity_score > 60:
        st.error("🔴 High Risk - Significant similarity detected.")
    elif similarity_score > 30:
        st.warning("🟡 Moderate Risk - Noticeable similarity detected.")
    else:
        st.success("🟢 Low Risk - Minimal similarity detected.")

    # ==============================
    # 🌐 TOP SOURCES
    # ==============================
    st.subheader("🌐 Top 3 Web Reference Sources")

    if top_sources:
        for i, source in enumerate(top_sources):
            st.write(f"{i+1}. {source['title']}")
            st.write(f"Similarity: {source['similarity']}%")
            st.markdown(f"[🔗 Visit Source]({source['url']})")
            st.write("---")
    else:
        st.info("No web sources found.")

    # ==============================
    # ✍️ STYLOMETRIC FEATURES
    # ==============================
    st.subheader("✍ Writing Style Indicators")

    features = extract_stylometric_features(user_text)

    avg_sentence_length = features[0]
    vocab_richness = features[1]
    stopword_ratio = features[2]

    col1, col2, col3 = st.columns(3)

    col1.metric("Avg Sentence Length", round(avg_sentence_length, 2))
    col2.metric("Vocabulary Richness", round(vocab_richness, 3))
    col3.metric("Stopword Ratio", round(stopword_ratio, 3))

    # ==============================
    # 🏆 FINAL INTEGRITY SCORE
    # ==============================
    st.subheader("🏆 Academic Integrity Score")

    integrity_score = max(0, 100 - similarity_score)

    st.write(f"{round(integrity_score, 2)} / 100")

    if integrity_score > 80:
        st.success("Strong academic integrity.")
    elif integrity_score > 50:
        st.warning("Acceptable with minor concerns.")
    else:
        st.error("Serious integrity concerns detected.")

    # ==============================
    # 📥 PDF REPORT GENERATION
    # ==============================
    st.subheader("📥 Download Report")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>Academic Integrity Analysis Report</b>", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(f"Generated On: {datetime.datetime.now()}", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph("<b>AI Authorship Assessment</b>", styles["Heading2"]))
    elements.append(Paragraph(f"Classification: {classification}", styles["Normal"]))
    elements.append(Paragraph(f"Confidence Score: {confidence}", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph("<b>Plagiarism Analysis</b>", styles["Heading2"]))
    elements.append(Paragraph(f"Plagiarism Rate: {similarity_score}%", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph("<b>Top Web Sources</b>", styles["Heading2"]))

    if top_sources:
        for source in top_sources:
            elements.append(Paragraph(
                f"{source['title']} - {source['similarity']}%",
                styles["Normal"]
            ))
            elements.append(Paragraph(source["url"], styles["Normal"]))
            elements.append(Spacer(1, 0.2 * inch))

    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph("<b>Academic Integrity Score</b>", styles["Heading2"]))
    elements.append(Paragraph(f"{round(integrity_score, 2)} / 100", styles["Normal"]))

    doc.build(elements)

    pdf = buffer.getvalue()
    buffer.close()

    st.download_button(
        label="📥 Download Analysis Report (PDF)",
        data=pdf,
        file_name="Academic_Integrity_Report.pdf",
        mime="application/pdf"
    )