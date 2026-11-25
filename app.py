
import streamlit as st
import joblib
import os

st.set_page_config(
    page_title="Spam Detection System",
    page_icon="📩",
    layout="centered"
)

st.title("📩 SMS Spam Detection System")

st.write(
    "Enter any SMS message below to identify "
    "whether it is spam or legitimate."
)

model_path = "models/best_spam_model.pkl"

if os.path.exists(model_path):

    model = joblib.load(model_path)

    message = st.text_area(
        "Enter your SMS message:",
        placeholder="Type your message here..."
    )

    if st.button("Detect Spam"):

        if not message.strip():

            st.warning("Please enter a message.")

        else:

            prediction = model.predict([message])[0]

            if prediction == 1:

                st.error("🚨 SPAM MESSAGE DETECTED!")

            else:

                st.success("✅ LEGITIMATE MESSAGE")

    st.divider()

    st.subheader("Model Information")

    st.write("Algorithms evaluated:")
    st.write("1. Naive Bayes")
    st.write("2. Logistic Regression")
    st.write("3. Support Vector Machine")

else:

    st.error("Model not found. Run train.py first.")
