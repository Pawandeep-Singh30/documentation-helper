import asyncio
import os
import ssl
from typing import List

import certifi
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyExtract, TavilyMap


from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)

load_dotenv()

# configure ssl context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Chroma = chroma(persist_directory="chroma_db", embedding_function=embeddings)
vectorstore = PineconeVectorStore(
    index_name=os.getenv("INDEX_NAME_DOC"),
    embedding=embeddings,
)
DOC_SITE_URL = "https://python.langchain.com/"
TAVILY_EXTRACT_BATCH_SIZE = 20

tavily_extract = TavilyExtract(extract_depth="advanced")
tavily_map = TavilyMap(map_depth=5, max_breadth=20, limit=1000)

async def index_documents_async(documents: List[Document], batch_size: int =50):
    """Process documents in batches asynchronously."""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f" VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )

    # Create batches
    batches = [
        documents[i:i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f" VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each",
    )

    # process all batches asynchronously
    async def add_batch(batch: List[Document], batch_num: int):
        try: 
            await vectorstore.aadd_documents(batch)
            log_success(
                f" VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)",
            )

        except Exception as e:
            log_error(f" VectorStore Indexing: Error adding batch {batch_num} - {e}")
            return False
        return True

    #process batches concurrently
    tasks = [add_batch(batch, i+1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f" VectorStore Indexing: Successfully indexed {successful} batches out of {len(batches)}",
        )
    else:
        log_warning(
            f" VectorStore Indexing: Failed to index {len(batches) - successful} batches. Retrying...",
        )


async def main():
    """ Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        f"TavilyMap: Mapping documentation URLs from {DOC_SITE_URL}",
        Colors.PURPLE,
    )
    site_map = tavily_map.invoke({"url": DOC_SITE_URL})
    mapped_urls = site_map["results"]
    log_success(
        f"TavilyMap: Successfully mapped {len(mapped_urls)} URLs from documentation site",
    )

    log_info(
        f"TavilyExtract: Extracting content from {len(mapped_urls)} URLs",
        Colors.PURPLE,
    )
    all_extract_results = []
    for i in range(0, len(mapped_urls), TAVILY_EXTRACT_BATCH_SIZE):
        batch = mapped_urls[i : i + TAVILY_EXTRACT_BATCH_SIZE]
        batch_num = i // TAVILY_EXTRACT_BATCH_SIZE + 1
        log_info(
            f"TavilyExtract: Extracting batch {batch_num} ({len(batch)} URLs)",
            Colors.PURPLE,
        )
        extract_res = tavily_extract.invoke({"urls": batch})

        if "error" in extract_res:
            log_error(f"TavilyExtract: Batch {batch_num} failed - {extract_res['error']}")
            continue

        all_extract_results.extend(extract_res.get("results", []))

    all_docs = [
        Document(page_content=result["raw_content"], metadata={"source": result["url"]})
        for result in all_extract_results
        if result.get("raw_content")
    ]
    log_success(
        f"TavilyExtract: Successfully extracted {len(all_docs)} documents",
    )

    # split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f" Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents",
    )

    #process documents asynchronously
    await index_documents_async(splitted_docs, batch_size=500)

    log_header("PIPELINE COMPLETE")
    log_success("Documentation Ingestion Pipeline completed successfully")
    log_info("Summary:", Colors.BOLD)
    log_info(f" - URLs Mapped: {len(site_map['results'])}")
    log_info(f" - Documents extracted: {len(all_docs)}")
    log_info(f" - Chunks created: {len(splitted_docs)}")








if __name__ == "__main__":
    asyncio.run(main())