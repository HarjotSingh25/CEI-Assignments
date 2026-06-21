from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

loader = PyPDFLoader("documents/sample.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print("Pages Loaded:", len(documents))
print("Total Chunks:", len(chunks))

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding Model Loaded Successfully")

db = FAISS.from_documents(
    chunks,
    embedding_model
)

print("FAISS Database Created Successfully")

print("Loading FLAN-T5 Model...")

tokenizer = AutoTokenizer.from_pretrained(
    "google/flan-t5-base"
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-base"
)

print("Model Loaded Successfully")

while True:

    question = input("\nAsk Question (type exit to quit): ")

    if question.lower() == "exit":
        break

    retrieved_docs = db.similarity_search(
        question,
        k=3
    )

    context = "\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    prompt = f"""
Answer the question using only the context provided.

Context:
{context}

Question:
{question}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=100
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    print("\nAnswer:")
    print(answer)