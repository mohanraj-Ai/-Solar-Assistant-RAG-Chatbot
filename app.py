import os
import re
import streamlit as st

from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_community.vectorstores import FAISS

# ==========================================
# LOAD ENV
# ==========================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Arrow Solar AI Assistant",
    page_icon="☀️",
    layout="wide"
)

st.title("☀️ Arrow Solar AI Assistant")

st.write(
    "Ask questions about solar energy, subsidies, installation, maintenance, and Arrow Solar services."
)

# ==========================================
# GEMINI MODEL
# ==========================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY
)

# ==========================================
# EMBEDDINGS
# ==========================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# ==========================================
# LOAD FAISS VECTOR STORE
# ==========================================

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# ==========================================
# SOLAR CALCULATOR TOOL
# ==========================================

def solar_calculator(question):

    numbers = re.findall(r"\d+", question.replace(",", ""))

    if not numbers:
        return None

    bill = float(numbers[0])

    if 500 <= bill <= 1500:
        kw = 1
        price = "₹70,000 - ₹90,000"

    elif bill <= 3000:
        kw = 2
        price = "₹1,40,000 - ₹1,55,000"

    elif bill <= 6000:
        kw = 3
        price = "₹1,90,000 - ₹2,30,000"

    elif bill <= 9500:
        kw = 4
        price = "₹2,60,000 - ₹2,70,000"

    elif bill <= 12500:
        kw = 5
        price = "₹2,70,000 - ₹3,00,000"

    else:
        return """
Please contact Arrow Solar for a customized site assessment and quotation.
"""

    return f"""
☀️ Recommended Solar Capacity: **{kw} kW**

💰 Estimated Project Cost: **{price}**

✅ Package Includes:
- Solar Panels
- Solar Inverter
- Mounting Structure
- Complete Installation
- Net Metering Support

🏦 Bank Loan Assistance Available

🎁 Government Subsidy Available under PM Surya Ghar Muft Yojana (subject to eligibility)

📍 Final project cost may vary depending on:
- Site inspection and roof condition
- Solar panel brand selected
- Inverter brand selected
- Installation complexity
- Additional customer requirements

📝 The final quotation will be provided after a site survey.

⏳ Quotation validity: **15 days** from the date of issue.

📞 Contact Arrow Solar for a free site survey and detailed proposal.
"""

# ==========================================
# SESSION MEMORY
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# DISPLAY CHAT HISTORY
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# USER INPUT
# ==========================================

question = st.chat_input(
    "Ask anything about solar energy..."
)

if question:

    # Save User Message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # ==========================================
    # SOLAR CALCULATOR TOOL
    # ==========================================

    bill_keywords = [
        "bill",
        "eb",
        "electricity",
        "current bill",
        "monthly bill"
    ]

    if any(word in question.lower() for word in bill_keywords):

        tool_answer = solar_calculator(question)

        if tool_answer:

            with st.chat_message("assistant"):
                st.markdown(tool_answer)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": tool_answer
                }
            )

            st.stop()

    # ==========================================
    # MEMORY CONTEXT
    # ==========================================

    chat_history = ""

    for msg in st.session_state.messages[-6:]:

        chat_history += (
            f"{msg['role']}: {msg['content']}\n"
        )

    # ==========================================
    # RAG RETRIEVAL
    # ==========================================

    docs = retriever.invoke(question)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # ==========================================
    # PROMPT
    # ==========================================

    prompt = f"""
You are Arrow Solar's professional AI Assistant.

About Arrow Solar:
- Solar installation company
- Service locations: Tamil Nadu, Puducherry, Kerala
- 5 years free service support
- PM Surya Ghar subsidy assistance
- Bank loan assistance available

Rules:

1. Use conversation history for follow-up questions.
2. Use document context for company-specific answers.
3. You may use general solar knowledge when appropriate.
4. Do not invent Arrow Solar specific facts.
5. If information is unavailable, say:
   "I couldn't find that information in Arrow Solar's documents."

Conversation History:
{chat_history}

Document Context:
{context}

Current User Question:
{question}

Answer professionally.
"""

    # ==========================================
    # GEMINI RESPONSE
    # ==========================================

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            response = llm.invoke(prompt)

            answer = response.content

            st.markdown(answer)

    # Save Assistant Response

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )