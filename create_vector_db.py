from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

print("SCRIPT STARTED")

# ==========================================
# LOAD ENV
# ==========================================

print("Step 1: Loading .env")
load_dotenv()

# ==========================================
# LOAD PDF
# ==========================================

print("Step 2: Loading PDF")

loader = PyPDFLoader("data/solar_manual.pdf")
documents = loader.load()

print(f"PDF Pages Loaded: {len(documents)}")

# ==========================================
# SPLIT DOCUMENTS
# ==========================================

print("Step 3: Splitting Documents")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = splitter.split_documents(documents)

print(f"Chunks Created: {len(docs)}")

# ==========================================
# GEMINI EMBEDDINGS
# ==========================================

print("Step 4: Creating Embeddings")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# ==========================================
# CREATE FAISS VECTOR STORE
# ==========================================

print("Step 5: Creating FAISS Vector Store")

vectorstore = FAISS.from_documents(
    docs,
    embeddings
)

# ==========================================
# SAVE VECTOR DB
# ==========================================

print("Step 6: Saving FAISS Index")

vectorstore.save_local("faiss_index")

print("SUCCESS: Vector DB Created Successfully")