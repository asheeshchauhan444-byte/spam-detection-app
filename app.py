# ============================================================
#        SMS SPAM / HAM DETECTION MODEL
#        PDF DATASET + MACHINE LEARNING
#        BEGINNER FRIENDLY COMPLETE CODE
# ============================================================


# ============================================================
# 1. REQUIRED LIBRARIES
# ============================================================

import sys
import subprocess
import os
import re


# Missing libraries automatically install karo

libraries = {
    "pandas": "pandas",
    "numpy": "numpy",
    "pdfplumber": "pdfplumber",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "openpyxl": "openpyxl"
}


for import_name, package_name in libraries.items():

    try:

        __import__(import_name)

    except ImportError:

        print("Installing:", package_name)

        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            package_name
        ])


# ============================================================
# 2. IMPORT
# ============================================================

import pandas as pd
import numpy as np
import pdfplumber
import matplotlib.pyplot as plt

from IPython.display import display, HTML

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# ============================================================
# 3. PDF PATH
# ============================================================

# IMPORTANT:
# Yahan apni PDF ka path diya gaya hai.

pdf_file = r"C:\Users\Dev computer\Downloads\SMSSpamCollection.pdf"


print("=" * 70)
print("          SMS SPAM / HAM DETECTION MODEL")
print("=" * 70)


print("\nPDF check ho rahi hai...")

print(pdf_file)


# ============================================================
# 4. CHECK PDF EXISTS
# ============================================================

if not os.path.exists(pdf_file):

    print("\n❌ PDF nahi mili.")

    print("\nCheck this path:")

    print(pdf_file)

    raise FileNotFoundError(
        "PDF file path galat hai."
    )


print("\n✅ PDF mil gayi!")


# ============================================================
# 5. READ PDF
# ============================================================

print("\nPDF read ho rahi hai...")

print("Please wait...")


all_text = ""


with pdfplumber.open(pdf_file) as pdf:

    total_pages = len(pdf.pages)

    print("\nTotal Pages:", total_pages)


    for page_number, page in enumerate(
        pdf.pages,
        start=1
    ):

        text = page.extract_text()


        if text:

            all_text += "\n" + text


print("\n✅ PDF reading complete!")


# ============================================================
# 6. CHECK TEXT
# ============================================================

if not all_text.strip():

    raise ValueError(

        """
❌ PDF se text nahi mila.

Aapki PDF scanned/image based ho sakti hai.
OCR ki zarurat hogi.
"""
    )


print(
    "Total extracted characters:",
    len(all_text)
)


# ============================================================
# 7. SHOW SAMPLE
# ============================================================

print("\n" + "=" * 70)

print("PDF TEXT SAMPLE")

print("=" * 70)

print(
    all_text[:2500]
)


# ============================================================
# 8. CREATE LINES
# ============================================================

lines = []

for line in all_text.splitlines():

    line = line.strip()

    if line:

        lines.append(line)


print("\nTotal lines:", len(lines))


# ============================================================
# 9. FUNCTION:
#    FIND SPAM / HAM
# ============================================================

def get_label(line):

    line = str(line).strip()

    lower = line.lower()


    # HAM

    if re.match(
        r"^ham[\s\t]+",
        lower
    ):

        return "ham"


    # SPAM

    if re.match(
        r"^spam[\s\t]+",
        lower
    ):

        return "spam"


    return None


# ============================================================
# 10. FUNCTION:
#     REMOVE LABEL
# ============================================================

def get_message(line):

    line = str(line).strip()


    line = re.sub(

        r"^(ham|spam)[\s\t]+",

        "",

        line,

        flags=re.IGNORECASE
    )


    return line.strip()


# ============================================================
# 11. CREATE DATASET
# ============================================================

messages = []

labels = []


for line in lines:

    label = get_label(line)


    if label is not None:

        message = get_message(line)


        if len(message) > 2:

            messages.append(message)

            labels.append(label)


# ============================================================
# 12. DATAFRAME
# ============================================================

data = pd.DataFrame({

    "message": messages,

    "label": labels
})


print("\n" + "=" * 70)

print("DATASET RESULT")

print("=" * 70)


print(
    "Messages found:",
    len(data)
)


# ============================================================
# 13. IF DATA NOT FOUND
# ============================================================

if len(data) == 0:

    print(
        """
        
❌ Spam/Ham data nahi mila.

PDF ka format check karo.

Expected format:

ham    Hello how are you?
spam   Congratulations you won a prize!

"""
    )

    raise ValueError(
        "PDF dataset format detect nahi hua."
    )


# ============================================================
# 14. CLEAN DATA
# ============================================================

data["message"] = (

    data["message"]

    .fillna("")

    .astype(str)

    .str.strip()
)


data["label"] = (

    data["label"]

    .astype(str)

    .str.lower()

    .str.strip()
)


# Empty messages remove

data = data[
    data["message"].str.len() > 2
]


# Duplicate messages remove

data = data.drop_duplicates(
    subset=["message"]
)


data = data.reset_index(
    drop=True
)


# ============================================================
# 15. SPAM = 1
#     HAM  = 0
# ============================================================

data["spam_ham"] = np.where(

    data["label"] == "spam",

    1,

    0
)


# ============================================================
# 16. COUNT
# ============================================================

spam_count = (

    data["spam_ham"] == 1

).sum()


ham_count = (

    data["spam_ham"] == 0

).sum()


total_count = len(data)


spam_percent = (

    spam_count /
    total_count

) * 100


ham_percent = (

    ham_count /
    total_count

) * 100


print("\n" + "=" * 70)

print("SMS SUMMARY")

print("=" * 70)


print(
    "Total SMS :",
    total_count
)


print(
    "SPAM      :",
    spam_count
)


print(
    "HAM       :",
    ham_count
)


print(
    "\nSPAM :",
    round(spam_percent, 2),
    "%"
)


print(
    "HAM  :",
    round(ham_percent, 2),
    "%"
)


# ============================================================
# 17. SMALL CIRCLE PIE CHART
# ============================================================

plt.figure(
    figsize=(3, 3)
)


plt.pie(

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


plt.title(
    "Spam vs Ham",
    fontsize=11
)


plt.axis("equal")

plt.tight_layout()

plt.show()


# ============================================================
# 18. SHOW DATA
# ============================================================

print("\n" + "=" * 70)

print("SAMPLE DATA")

print("=" * 70)


display(

    data[
        [
            "message",
            "label",
            "spam_ham"
        ]
    ].head(10)

)


# ============================================================
# 19. KEYWORDS
# ============================================================

keywords = [

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
# 20. KEYWORD DETECTION
# ============================================================

def keyword_value(message, keyword):

    message = str(
        message
    ).lower()


    keyword = str(
        keyword
    ).lower()


    if keyword in message:

        return 1

    else:

        return 0


# ============================================================
# 21. CREATE KEYWORD 0/1 COLUMNS
# ============================================================

for keyword in keywords:

    column = (

        "kw_"

        +

        keyword

        .replace(
            " ",
            "_"
        )
    )


    data[column] = data["message"].apply(

        lambda x:

        keyword_value(
            x,
            keyword
        )

    )


# ============================================================
# 22. TRAINING DATA
# ============================================================

X = data["message"]

y = data["spam_ham"]


# ============================================================
# 23. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


# ============================================================
# 24. MACHINE LEARNING MODEL
# ============================================================

model = Pipeline([

    (

        "tfidf",

        TfidfVectorizer(

            lowercase=True,

            ngram_range=(1, 2),

            min_df=1,

            max_features=15000

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


# ============================================================
# 25. TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)

print("MODEL TRAINING")

print("=" * 70)


print(
    "Model training started..."
)


model.fit(

    X_train,

    y_train
)


print(
    "✅ Model trained successfully!"
)


# ============================================================
# 26. TEST MODEL
# ============================================================

prediction = model.predict(
    X_test
)


accuracy = accuracy_score(

    y_test,

    prediction
)


print("\n" + "=" * 70)

print("MODEL ACCURACY")

print("=" * 70)


print(

    round(
        accuracy * 100,
        2
    ),

    "%"
)


print("\nClassification Report:")


print(

    classification_report(

        y_test,

        prediction

    )
)


# ============================================================
# 27. CONTINUOUS MESSAGE ANIMATION
# ============================================================

def animate_message(message):

    safe_message = (

        str(message)

        .replace(
            "&",
            "&amp;"
        )

        .replace(
            "<",
            "&lt;"
        )

        .replace(
            ">",
            "&gt;"
        )
    )


    html = f"""

    <div style="

        width:100%;

        height:55px;

        overflow:hidden;

        border:2px solid #444;

        border-radius:10px;

        background:#111827;

        display:flex;

        align-items:center;

        box-sizing:border-box;

    ">


        <div style="

            white-space:nowrap;

            display:inline-block;

            color:white;

            font-size:16px;

            font-weight:bold;

            padding-left:100%;

            animation:
                moveMessage
                60s
                linear
                infinite;

        ">

            {safe_message}

            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

            {safe_message}

        </div>

    </div>


    <style>

    @keyframes moveMessage {{

        0% {{

            transform:
                translateX(0);

        }}


        100% {{

            transform:
                translateX(-50%);

        }}

    }}

    </style>

    """


    display(
        HTML(html)
    )


# ============================================================
# 28. CHECK MESSAGE FUNCTION
# ============================================================

def check_message(message):


    # --------------------------------------------------------
    # MESSAGE CLEAN
    # --------------------------------------------------------

    message = str(
        message
    ).strip()


    if message == "":

        print(
            "❌ Message empty hai."
        )

        return


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # KEYWORD DETECTION
    # --------------------------------------------------------

    found_keywords = []


    for keyword in keywords:

        if keyword.lower() in message.lower():

            found_keywords.append(
                keyword
            )


    keyword_count = len(
        found_keywords
    )


    # --------------------------------------------------------
    # SUSPICIOUS PATTERNS
    # --------------------------------------------------------

    patterns = [

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


    for pattern in patterns:

        if re.search(

            pattern,

            message,

            re.IGNORECASE

        ):

            pattern_count += 1


    # --------------------------------------------------------
    # FINAL SPAM SCORE
    # --------------------------------------------------------

    extra_score = min(

        pattern_count * 0.06,

        0.30

    )


    final_spam = min(

        spam_probability +
        extra_score,

        0.99

    )


    final_ham = 1 - final_spam


    # Multiple suspicious words

    if (

        keyword_count >= 4

        or

        pattern_count >= 3

    ):

        final_spam = max(
            final_spam,
            0.75
        )


        final_ham = 1 - final_spam


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    if final_spam >= 0.50:

        result = "SPAM"

        binary = 1

    else:

        result = "HAM"

        binary = 0


    confidence = max(

        final_spam,

        final_ham

    ) * 100


    # ========================================================
    # OUTPUT
    # ========================================================

    print("\n")

    print("=" * 70)

    print(
        "             SMS DETECTION RESULT"
    )

    print("=" * 70)


    print("\n📩 YOUR MESSAGE:")


    # Continuous animation

    animate_message(
        message
    )


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    print("\n")

    print(
        "RESULT :",
        result
    )


    print(
        "BINARY :",
        binary
    )


    print(
        "SPAM = 1 | HAM = 0"
    )


    print(
        "\nCONFIDENCE :",
        round(
            confidence,
            2
        ),
        "%"
    )


    print("\n" + "-" * 70)


    print(
        "SPAM :",
        round(
            final_spam * 100,
            2
        ),
        "%"
    )


    print(
        "HAM  :",
        round(
            final_ham * 100,
            2
        ),
        "%"
    )


    # --------------------------------------------------------
    # KEYWORDS
    # --------------------------------------------------------

    print("\n" + "-" * 70)

    print(
        "KEYWORD DETECTION"
    )


    print(
        "1 = Found | 0 = Not Found"
    )


    # Horizontal table

    keyword_row = {}


    for keyword in keywords:

        keyword_row[keyword] = (

            1

            if keyword.lower()
            in message.lower()

            else 0

        )


    keyword_df = pd.DataFrame(
        [keyword_row]
    )


    display(
        keyword_df
    )


    # --------------------------------------------------------
    # FOUND KEYWORDS
    # --------------------------------------------------------

    print(
        "\nDetected Keywords:"
    )


    if found_keywords:

        print(
            " | ".join(
                found_keywords
            )
        )

    else:

        print(
            "None"
        )


    # ========================================================
    # PIE CHART
    # ========================================================

    print(
        "\n📊 SPAM vs HAM"
    )


    plt.figure(
        figsize=(2.8, 2.8)
    )


    plt.pie(

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


    plt.title(
        "Message Result",
        fontsize=10
    )


    plt.axis("equal")

    plt.tight_layout()

    plt.show()


    # ========================================================
    # SAVE RESULT
    # ========================================================

    result_file = (
        "SMS_Prediction_Results.xlsx"
    )


    new_result = pd.DataFrame([{

        "Message":
            message,

        "Result":
            result,

        "Binary":
            binary,

        "SPAM_%":
            round(
                final_spam * 100,
                2
            ),

        "HAM_%":
            round(
                final_ham * 100,
                2
            ),

        "Confidence_%":
            round(
                confidence,
                2
            ),

        "Keyword_Count":
            keyword_count,

        "Detected_Keywords":
            ", ".join(
                found_keywords
            )

    }])


    if os.path.exists(
        result_file
    ):

        old_result = pd.read_excel(
            result_file
        )


        final_result = pd.concat(

            [
                old_result,
                new_result
            ],

            ignore_index=True
        )

    else:

        final_result = new_result


    final_result.to_excel(

        result_file,

        index=False
    )


    print(
        "\n✅ Prediction Excel mein save ho gaya:"
    )


    print(
        os.path.abspath(
            result_file
        )
    )


# ============================================================
# 29. SAVE CLEAN DATASET
# ============================================================

data.to_excel(

    "SMS_Spam_Processed_Data.xlsx",

    index=False

)


print("\n" + "=" * 70)

print(
    "          🚀 MODEL READY"
)

print("=" * 70)


print(
    """
Ab apna SMS neeche paste karo.

Example:

Congratulations! You have won a free prize.
Click here to claim your reward.

Ya:

Hey, are you coming to college today?

Program band karne ke liye:

exit
"""
)


# ============================================================
# 30. MESSAGE INPUT
# ============================================================

while True:

    user_message = input(
        "\n📩 Paste Your Message: "
    )


    # EXIT

    if user_message.strip().lower() == "exit":

        print(
            "\n✅ Program closed."
        )

        break


    # EMPTY

    if user_message.strip() == "":

        print(
            "❌ Please enter a message."
        )

        continue


    # CHECK

    check_message(
        user_message
    )