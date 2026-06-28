import os
import streamlit as st
import plotly.graph_objects as go
import base64

from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"
)

# --------------------------------------------------
# LOAD CSS
# --------------------------------------------------

def load_css():

    banner = get_base64("assets/banner.jpg")

    with open("assets/style.css") as f:
        css = f.read()

    st.markdown(
        f"""
        <style>

        .stApp{{
            background-image:
                linear-gradient(
                    rgba(0,0,0,0.70),
                    rgba(0,0,0,0.70)
                ),
                url("data:image/jpg;base64,{banner}");

            background-size:cover;
            background-position:center;
            background-repeat:no-repeat;
            background-attachment:fixed;
        }}

        {css}

        </style>
        """,
        unsafe_allow_html=True
    )

def get_base64(image_path):
    with open(image_path, "rb") as image:
        return base64.b64encode(image.read()).decode()
    
load_css()

# --------------------------------------------------
# MODEL PATH
# --------------------------------------------------

MODEL_PATH = "model/lstm_sentiment_model.keras"

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_sentiment_model():

    if not os.path.exists(MODEL_PATH):
        st.error(
            "❌ Model file not found.\n\n"
            "Place 'lstm_sentiment_model.keras' inside the model folder."
        )
        st.stop()

    return load_model(MODEL_PATH)

model = load_sentiment_model()

# --------------------------------------------------
# IMDB WORD INDEX
# --------------------------------------------------

VOCAB_SIZE = 10000
MAXLEN = 200

@st.cache_resource
def load_word_index():

    word_index = imdb.get_word_index()

    word_index = {
        k: (v + 3)
        for k, v in word_index.items()
    }

    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2

    return word_index

word_index = load_word_index()

# --------------------------------------------------
# PREPROCESS REVIEW
# --------------------------------------------------

def preprocess_text(text):

    words = text.lower().split()

    sequence = [
        word_index.get(word, 2)
        for word in words
    ]

    sequence = [
        x
        for x in sequence
        if x < VOCAB_SIZE
    ]

    padded = pad_sequences(
        [sequence],
        maxlen=MAXLEN,
        padding="post"
    )

    return padded

# --------------------------------------------------
# PREDICT SENTIMENT
# --------------------------------------------------

def predict_sentiment(review):

    processed = preprocess_text(review)

    prediction = model.predict(
        processed,
        verbose=0
    )[0][0]

    sentiment = "😊 Positive" if prediction >= 0.5 else "😞 Negative"

    confidence = prediction if prediction >= 0.5 else (1 - prediction)

    return sentiment, float(prediction), float(confidence)

# --------------------------------------------------
# CONFIDENCE GAUGE
# --------------------------------------------------

def create_gauge(confidence):

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            title={"text": "Confidence (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#00C853"},
                "steps": [
                    {"range": [0, 50], "color": "#F44336"},
                    {"range": [50, 75], "color": "#FFC107"},
                    {"range": [75, 100], "color": "#00E676"}
                ]
            }
        )
    )

    fig.update_layout(height=300)

    return fig

# --------------------------------------------------
# SAMPLE REVIEWS
# --------------------------------------------------

positive_example = (
    "This movie was absolutely amazing. "
    "The acting and story were fantastic."
)

negative_example = (
    "Worst movie ever. "
    "I wasted my time watching it."
)

# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.image("assets/logo.png", width=300)

    st.title("🎬 About")

    st.markdown("""
This application predicts the sentiment of movie reviews using an LSTM model trained on the IMDb dataset.

### Model
- LSTM
- TensorFlow/Keras

### Dataset
- IMDb Movie Reviews
- 25,000 Reviews

### Developer
**Gaurav Maurya**
""")

# ==================================================
# MAIN PAGE
# ==================================================

st.image("assets/banner.jpg", use_container_width=True)

st.title("🎬 Movie Sentiment Analyzer")

st.write(
    "Analyze movie reviews using Deep Learning (LSTM)."
)

st.markdown("---")

# ==================================================
# EXAMPLE BUTTONS
# ==================================================

c1, c2 = st.columns(2)

with c1:
    if st.button("😊 Positive Example"):
        st.session_state["review"] = positive_example

with c2:
    if st.button("😞 Negative Example"):
        st.session_state["review"] = negative_example

# ==================================================
# REVIEW INPUT
# ==================================================

review = st.text_area(
    "Enter Movie Review",
    value=st.session_state.get("review", ""),
    height=200,
    placeholder="Example: This movie was amazing..."
)

# ==================================================
# ANALYZE BUTTON
# ==================================================

if st.button("🚀 Analyze Sentiment", use_container_width=True):

    if review.strip() == "":

        st.warning("Please enter a review.")

    else:

        sentiment, probability, confidence = predict_sentiment(review)

        st.markdown("---")

        if probability >= 0.5:

            st.success(sentiment)

        else:

            st.error(sentiment)

        # ----------------------------------------

        left, right = st.columns([2, 1])

        with left:

            st.subheader("Confidence Gauge")

            fig = create_gauge(confidence)

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with right:

            st.subheader("Prediction")

            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

            st.write("😊 Positive")

            st.progress(float(probability))

            st.write(f"{probability*100:.2f}%")

            st.write("😞 Negative")

            st.progress(float(1-probability))

            st.write(f"{(1-probability)*100:.2f}%")

        # ----------------------------------------

        st.markdown("---")

        st.subheader("Review")

        st.info(review)

        # ----------------------------------------

        report = f"""
Movie Sentiment Analysis

Review:
{review}

Prediction:
{sentiment}

Confidence:
{confidence*100:.2f}%

Positive Probability:
{probability*100:.2f}%

Negative Probability:
{(1-probability)*100:.2f}%
"""

        st.download_button(
            "📄 Download Report",
            report,
            file_name="prediction_report.txt",
            mime="text/plain"
        )

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.markdown(
"""
<center>

Made with ❤️ using

TensorFlow • Streamlit • Plotly

</center>
""",
unsafe_allow_html=True
)