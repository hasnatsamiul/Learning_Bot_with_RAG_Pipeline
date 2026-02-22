import os
import traceback
import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from unstructured.partition.pdf import partition_pdf

# Silence pdfminer warnings
logging.getLogger("pdfminer").setLevel(logging.ERROR)

load_dotenv()

# Read paths directly from env
SOURCE_DOC_DIR = os.getenv("SOURCE_DOC_DIR", "./pdfs")
IMAGE_DIR = os.getenv("IMAGE_DIR", "./images")
FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "./faiss_index")

# Import embedder AFTER loading env
from entities.embedder import embedder


def main():
    try:
        print("SOURCE_DOC_DIR:", SOURCE_DOC_DIR)
        print("IMAGE_DIR:", IMAGE_DIR)
        print("FAISS_INDEX_DIR:", FAISS_INDEX_DIR)

        os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

        pdf_dir = Path(SOURCE_DOC_DIR)

        if not pdf_dir.exists():
            raise RuntimeError(f"Directory '{SOURCE_DOC_DIR}' does not exist.")

        pdf_files = list(pdf_dir.glob("*.pdf"))

        if not pdf_files:
            raise RuntimeError("No PDFs found in SOURCE_DOC_DIR.")

        text_docs = []

        for pdf_path in pdf_files:
            print("\nProcessing:", pdf_path.name)

            elements = partition_pdf(
                filename=str(pdf_path),
                extract_images_in_pdf=False,  # Disable images for stability
                chunking_strategy="by_title",
                max_characters=3000,
                combine_text_under_n_chars=2000,
                new_after_n_chars=2800,
            )

            added = 0
            for el in elements:
                text = getattr(el, "text", None)
                if not text:
                    continue
                text = text.strip()
                if not text:
                    continue

                text_docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": pdf_path.name},
                    )
                )
                added += 1

            print("Chunks added:", added)

        print("\nTotal chunks:", len(text_docs))

        if not text_docs:
            raise RuntimeError("No text extracted from PDFs.")

        print("Creating FAISS index...")
        faiss_db = FAISS.from_documents(text_docs, embedder)

        print("Saving FAISS index...")
        faiss_db.save_local(FAISS_INDEX_DIR)

        print("Done")

    except Exception as e:
        print("Error:", e)
        traceback.print_exc()


if __name__ == "__main__":
    main()