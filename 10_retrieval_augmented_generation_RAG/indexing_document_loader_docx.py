from langchain_community.document_loaders import Docx2txtLoader

docx_loader = Docx2txtLoader(r"10_retrieval_augmented_generation_RAG/docs/Introduction_to_Data_and_Data_Science.docx")
docx_pages = docx_loader.load()
print(docx_pages)