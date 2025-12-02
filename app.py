import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)

    text=y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    
    return ' '.join(y)


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# App UI
st.set_page_config(page_title="Spam Classifier", page_icon="📧", layout="wide")

# Sidebar with model info
with st.sidebar:
    st.header("ℹ️ Model Information")
    st.markdown("""
    **Algorithm:** Multinomial Naive Bayes  
    **Accuracy:** 97.2%  
    **Precision:** 96.8%  
    **Training Samples:** 5,574 messages  
    **Features:** 3,000 TF-IDF features  
    **Last Trained:** 01/12/2025
    """)
    
    st.divider()
    st.caption("Built with: Streamlit, Scikit-learn, NLTK")


st.title('Email/SMS Spam Classifier')
st.markdown("Detect spam messages using **Multinomial Naive Bayes** machine learning model")

col1, col2 = st.columns([2, 1])

with col1:
    input_sms = st.text_area('Enter your message:', height=150, 
                           placeholder="Type or paste your email/SMS message here...")
    
    if st.button('🔍 Detect Spam', type='primary', use_container_width=True):
        if input_sms.strip():
            with st.spinner('Analyzing message...'):
                # Preprocess
                transformed_sms = transform_text(input_sms)
                
                # Vectorize
                vector_input = tfidf.transform([transformed_sms])
                
                # Predict
                result = model.predict(vector_input)[0]
                probability = model.predict_proba(vector_input)[0]
                
                # Display results
                if result == 1:
                    st.error(f'🚨 **SPAM DETECTED** ({(probability[1]*100):.1f}% confidence)')
                    st.progress(probability[1])
                else:
                    st.success(f'✅ **NOT SPAM** ({(probability[0]*100):.1f}% confidence)')
                    st.progress(probability[0])
                    
                # Show details in expander
                with st.expander("📊 Prediction Details"):
                    st.metric("Spam Probability", f"{probability[1]*100:.2f}%")
                    st.metric("Ham Probability", f"{probability[0]*100:.2f}%")
        else:
            st.warning("Please enter a message to analyze.")

with col2:
    st.subheader("📝 Example Messages")
    
    example_tab1, example_tab2 = st.tabs(["Spam", "Not Spam"])
    
    with example_tab1:
        st.markdown("""
        **Common Spam Patterns:**
        - "WINNER!! You have won $1000!"
        - "URGENT: Your account needs verification"
        - "Free gift card waiting for you"
        - "Click this link to claim your prize"
        """)
        
        
    
    with example_tab2:
        st.markdown("""
        **Legitimate Messages:**
        - "Hey, are we meeting for lunch tomorrow?"
        - "Your package has been delivered"
        - "Meeting reminder: 2 PM today"
        - "Thanks for your email, will respond soon"
        """)
        
        
