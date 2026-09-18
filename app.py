# ============================================================
# SMS SPAM / HAM DETECTOR
# Streamlit + Machine Learning
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import re
import os

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📱",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #666;
    font-size: 18px;
    margin-bottom: 30px;
}

.result-box {
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    margin-top: 20px;
}

.spam-box {
    background-color: #ffe5e5;
    border: 2px solid #ff4b4b;
}

.ham-box {
    background-color: #e5fff0;
    border: 2px solid #00a86b;
}

.result-text {
    font-size: 36px;
    font-weight: 800;
}

.small-text {
    font-size: 16px;
}

.keyword-box {
    background-color: #f1f3f6;
    padding: 10px;
    border-radius: 10px;
    margin: 4px;
}

div[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="title">📱 SMS Spam Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning based Spam & Ham Message Detection'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# PDF FILE LOCATION
# ============================================================

PDF_FILE = "SMSSpamCollection.pdf"


# ============================================================
# KEYWORDS
# ============================================================

KEYWORDS = [

    "free",
    "winner",
    "won",
    "win",
    "congratulations",
    "prize",
    "claim",
    "cash",
    "urgent",
    "offer",
    "click",
    "click here",
    "call now",
    "buy now",
    "limited",
    "guaranteed",
    "download",
    "unsubscribe",
    "loan",
    "credit",
    "money",
    "bonus",
    "reward",
    "gift",
    "selected",
    "exclusive",
    "voucher",
    "promo",
    "promotion",
    "http",
    "https",
    "www",
    "wap",
    "mobile",
    "service msg",
    "service message",
    "account",
    "bank",
    "otp",
    "password",
    "verify",
    "verification"

]


# ============================================================
# PDF DATA READER
# ============================================================

@st.cache_data
def read_pdf_data(pdf_file):

    messages = []
    labels = []

    try:

        with pdfplumber.open(pdf_file) as pdf:

            for page in pdf.pages:

                text = page.extract_text()

                if not text:
                    continue

                lines = text.splitlines()

                for line in lines:

                    line = line.strip()

                    if not line:
                        continue


                    # ------------------------------
                    # SPAM
                    # ------------------------------

                    spam_match = re.match(
                        r"^spam[\s\t]+(.+)$",
                        line,
                        re.IGNORECASE
                    )

                    if spam_match:

                        message = spam_match.group(1).strip()

                        if message:

                            messages.append(message)
                            labels.append("spam")

                        continue


                    # ------------------------------
                    # HAM
                    # ------------------------------

                    ham_match = re.match(
                        r"^ham[\s\t]+(.+)$",
                        line,
                        re.IGNORECASE
                    )

                    if ham_match:

                        message = ham_match.group(1).strip()

                        if message:

                            messages.append(message)
                            labels.append("ham")


    except Exception as e:

        return pd.DataFrame(
            columns=["message", "label"]
        ), str(e)


    df = pd.DataFrame({

        "message": messages,

        "label": labels

    })


    if not df.empty:

        df["message"] = (
            df["message"]
            .astype(str)
            .str.strip()
        )

        df["label"] = (
            df["label"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        df = df[
            df["message"].str.len() > 2
        ]

        df = df.drop_duplicates(
            subset=["message"]
        )

        df = df.reset_index(
            drop=True
        )


    return df, None


# ============================================================
# LOAD DATASET
# ============================================================

if not os.path.exists(PDF_FILE):

    st.error(
        "❌ SMSSpamCollection.pdf GitHub repository "
        "mein nahi mili."
    )

    st.info(
        "PDF ko app.py ke same folder mein upload karo."
    )

    st.stop()


with st.spinner(
    "📄 SMS dataset load ho raha hai..."
):

    df, pdf_error = read_pdf_data(
        PDF_FILE
    )


if pdf_error:

    st.error(
        "PDF read karte time error: "
        + str(pdf_error)
    )

    st.stop()


if df.empty:

    st.error(
        "❌ PDF se Spam/Ham messages nahi mile."
    )

    st.warning(
        "PDF mein dataset ka format "
        "ham/spam + message hona chahiye."
    )

    st.stop()


# ============================================================
# CREATE BINARY LABEL
# ============================================================

df["binary"] = np.where(
    df["label"] == "spam",
    1,
    0
)


# ============================================================
# DATA SUMMARY
# ============================================================

total_sms = len(df)

spam_count = int(
    (df["binary"] == 1).sum()
)

ham_count = int(
    (df["binary"] == 0).sum()
)

spam_percent = (
    spam_count / total_sms
) * 100

ham_percent = (
    ham_count / total_sms
) * 100


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model(messages, labels):

    X_train, X_test, y_train, y_test = train_test_split(

        messages,

        labels,

        test_size=0.20,

        random_state=42,

        stratify=labels

    )


    model = Pipeline([

        (
            "tfidf",

            TfidfVectorizer(

                lowercase=True,

                strip_accents="unicode",

                ngram_range=(1, 2),

                min_df=1,

                max_features=20000

            )
        ),

        (
            "classifier",

            LogisticRegression(

                max_iter=3000,

                class_weight="balanced",

                random_state=42

            )
        )

    ])


    model.fit(
        X_train,
        y_train
    )


    test_prediction = model.predict(
        X_test
    )


    accuracy = accuracy_score(

        y_test,

        test_prediction

    )


    return model, accuracy


with st.spinner(
    "🤖 Machine Learning model train ho raha hai..."
):

    model, accuracy = train_model(

        df["message"],

        df["binary"]

    )


# ============================================================
# TOP METRICS
# ============================================================

st.markdown("## 📊 Dataset Overview")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total SMS",
        f"{total_sms:,}"
    )


with col2:

    st.metric(
        "Spam",
        f"{spam_count:,}"
    )


with col3:

    st.metric(
        "Ham",
        f"{ham_count:,}"
    )


with col4:

    st.metric(
        "Model Accuracy",
        f"{accuracy * 100:.2f}%"
    )


# ============================================================
# DATASET PIE CHART
# ============================================================

st.markdown("## 📊 Dataset Spam vs Ham")


col_left, col_right = st.columns(
    [1, 1]
)


with col_left:

    fig, ax = plt.subplots(
        figsize=(4, 4)
    )


    ax.pie(

        [
            spam_count,
            ham_count
        ],

        labels=[

            f"SPAM\n{spam_percent:.1f}%",

            f"HAM\n{ham_percent:.1f}%"

        ],

        autopct="%1.1f%%",

        startangle=90

    )


    ax.set_title(
        "Spam vs Ham",
        fontsize=13
    )


    ax.axis("equal")


    st.pyplot(
        fig,
        use_container_width=False
    )


with col_right:

    st.markdown(
        f"""
        ### Dataset Result

        **SPAM:** {spam_percent:.2f}%

        **HAM:** {ham_percent:.2f}%

        **Total Messages:** {total_sms:,}

        **Model Accuracy:** {accuracy * 100:.2f}%
        """
    )


# ============================================================
# MESSAGE INPUT
# ============================================================

st.markdown("---")

st.markdown(
    "## ✉️ Check Your Message"
)


st.write(
    "Apna SMS neeche paste karo aur Check Message button dabao."
)


message = st.text_area(

    "Enter your SMS",

    height=150,

    placeholder=
    "Example: Congratulations! You have won a free prize. Click here to claim your reward..."

)


check_button = st.button(

    "🔍 Check Message",

    type="primary",

    use_container_width=True

)


# ============================================================
# CHECK MESSAGE
# ============================================================

if check_button:

    if not message.strip():

        st.warning(
            "⚠️ Please enter a message first."
        )

        st.stop()


    # ========================================================
    # MODEL PROBABILITY
    # ========================================================

    probabilities = model.predict_proba(
        [message]
    )[0]


    classes = model.named_steps[
        "classifier"
    ].classes_


    ham_probability = probabilities[
        list(classes).index(0)
    ]


    spam_probability = probabilities[
        list(classes).index(1)
    ]


    # ========================================================
    # KEYWORD DETECTION
    # ========================================================

    message_lower = message.lower()


    found_keywords = []


    keyword_values = {}


    for keyword in KEYWORDS:

        if keyword.lower() in message_lower:

            keyword_values[keyword] = 1

            found_keywords.append(
                keyword
            )

        else:

            keyword_values[keyword] = 0


    # ========================================================
    # SUSPICIOUS PATTERNS
    # ========================================================

    suspicious_patterns = [

        r"http[s]?://",

        r"www\.",

        r"\bfree\b",

        r"\bwin\b",

        r"\bwinner\b",

        r"\bwon\b",

        r"\bprize\b",

        r"\bclaim\b",

        r"\bcongratulations\b",

        r"\burgent\b",

        r"\boffer\b",

        r"\bbonus\b",

        r"\breward\b",

        r"\bselected\b",

        r"\bdownload\b",

        r"\bwap\b",

        r"\bcall now\b",

        r"\bbuy now\b",

        r"\bclick here\b"

    ]


    pattern_count = 0


    for pattern in suspicious_patterns:

        if re.search(

            pattern,

            message_lower

        ):

            pattern_count += 1


    # ========================================================
    # FINAL SCORE
    # ========================================================

    keyword_boost = min(

        pattern_count * 0.04,

        0.20

    )


    final_spam = min(

        spam_probability +
        keyword_boost,

        0.99

    )


    final_ham = 1 - final_spam


    # Strong spam signal

    if (

        pattern_count >= 4

        or

        len(found_keywords) >= 5

    ):

        final_spam = max(
            final_spam,
            0.80
        )

        final_ham = (
            1 - final_spam
        )


    # ========================================================
    # FINAL RESULT
    # ========================================================

    if final_spam >= 0.50:

        result = "SPAM"

        binary_result = 1

    else:

        result = "HAM"

        binary_result = 0


    confidence = max(

        final_spam,

        final_ham

    ) * 100


    # ========================================================
    # MESSAGE DISPLAY
    # ========================================================

    st.markdown(
        "### 📩 Your Message"
    )


    # Slow continuous animation

    safe_message = (

        message

        .replace("&", "&amp;")

        .replace("<", "&lt;")

        .replace(">", "&gt;")

    )


    st.markdown(

        f"""
        <div style="
            width:100%;
            overflow:hidden;
            background:#111827;
            border-radius:12px;
            border:2px solid #374151;
            padding:15px 0;
            margin-bottom:20px;
        ">

            <div style="
                white-space:nowrap;
                color:#ffffff;
                font-size:18px;
                font-weight:600;
                animation:
                    scrollMessage 60s
                    linear infinite;
                display:inline-block;
                padding-left:100%;
            ">

                {safe_message}

                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

                {safe_message}

            </div>

        </div>

        <style>

        @keyframes scrollMessage {{

            0% {{
                transform:translateX(0);
            }}

            100% {{
                transform:translateX(-50%);
            }}

        }}

        </style>
        """,

        unsafe_allow_html=True

    )


    # ========================================================
    # RESULT BOX
    # ========================================================

    if result == "SPAM":

        st.markdown(

            f"""
            <div class="result-box spam-box">

                <div class="result-text">
                    🚨 SPAM MESSAGE
                </div>

                <div class="small-text">
                    Binary Value: <b>1</b>
                </div>

            </div>
            """,

            unsafe_allow_html=True

        )

    else:

        st.markdown(

            f"""
            <div class="result-box ham-box">

                <div class="result-text">
                    ✅ HAM MESSAGE
                </div>

                <div class="small-text">
                    Binary Value: <b>0</b>
                </div>

            </div>
            """,

            unsafe_allow_html=True

        )


    # ========================================================
    # PROBABILITY
    # ========================================================

    st.markdown(
        "### 📈 Prediction Probability"
    )


    p1, p2, p3 = st.columns(3)


    with p1:

        st.metric(

            "🚨 SPAM",

            f"{final_spam * 100:.2f}%"

        )


    with p2:

        st.metric(

            "✅ HAM",

            f"{final_ham * 100:.2f}%"

        )


    with p3:

        st.metric(

            "🎯 Confidence",

            f"{confidence:.2f}%"

        )


    # ========================================================
    # PIE CHART
    # ========================================================

    st.markdown(
        "### 🥧 Message Result"
    )


    chart_col1, chart_col2 = st.columns(
        [1, 2]
    )


    with chart_col1:

        fig2, ax2 = plt.subplots(

            figsize=(3, 3)

        )


        ax2.pie(

            [

                final_spam,

                final_ham

            ],

            labels=[

                f"SPAM\n{final_spam*100:.1f}%",

                f"HAM\n{final_ham*100:.1f}%"

            ],

            autopct="%1.1f%%",

            startangle=90

        )


        ax2.axis("equal")


        ax2.set_title(
            "Message Type",
            fontsize=11
        )


        st.pyplot(
            fig2,
            use_container_width=False
        )


    with chart_col2:

        st.markdown(
            f"""
            ### Detection Details

            **Result:** {result}

            **Binary:** {binary_result}

            **SPAM:** {final_spam*100:.2f}%

            **HAM:** {final_ham*100:.2f}%

            **Keywords Found:** {len(found_keywords)}

            **Suspicious Patterns:** {pattern_count}
            """
        )


    # ========================================================
    # KEYWORDS
    # ========================================================

    st.markdown("---")

    st.markdown(
        "### 🔎 Keyword Detection"
    )


    if found_keywords:

        st.success(
            "Detected Keywords: "
            +
            " | ".join(
                found_keywords
            )
        )

    else:

        st.info(
            "No suspicious keywords detected."
        )


    # ========================================================
    # 0 / 1 KEYWORD TABLE
    # ========================================================

    keyword_df = pd.DataFrame(

        [
            keyword_values
        ]

    )


    st.markdown(
        "### 0 / 1 Keyword Indicators"
    )


    st.caption(
        "1 = keyword found | 0 = keyword not found"
    )


    st.dataframe(

        keyword_df,

        use_container_width=True,

        hide_index=True

    )


    # ========================================================
    # DOWNLOAD RESULT
    # ========================================================

    result_row = {

        "Message":
            message,

        "Result":
            result,

        "Binary":
            binary_result,

        "Spam_%":
            round(
                final_spam * 100,
                2
            ),

        "Ham_%":
            round(
                final_ham * 100,
                2
            ),

        "Confidence_%":
            round(
                confidence,
                2
            ),

        "Keywords_Found":
            len(found_keywords),

        "Detected_Keywords":
            ", ".join(
                found_keywords
            )

    }


    result_df = pd.DataFrame(
        [result_row]
    )


    csv_data = result_df.to_csv(
        index=False
    )


    st.download_button(

        label="⬇️ Download Prediction",

        data=csv_data,

        file_name="SMS_Prediction.csv",

        mime="text/csv",

        use_container_width=True

    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "📱 About"
    )


    st.write(
        """
        This application uses
        Machine Learning to classify
        SMS messages.
        """
    )


    st.markdown(
        """
        ### Classification

        **SPAM = 1**

        **HAM = 0**

        ### Model

        • TF-IDF

        • Logistic Regression

        • Keyword Analysis

        ### Dataset

        SMS Spam Collection
        """
    )


    st.markdown("---")


    st.write(
        f"📄 Dataset Messages: {total_sms:,}"
    )


    st.write(
        f"🚨 Spam: {spam_count:,}"
    )


    st.write(
        f"✅ Ham: {ham_count:,}"
    )


    st.write(
        f"🎯 Accuracy: {accuracy*100:.2f}%"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(

    """
    <div style="
        text-align:center;
        color:#777;
        padding:10px;
    ">

        SMS Spam Detection App |
        Machine Learning Project

    </div>
    """,

    unsafe_allow_html=True

)
