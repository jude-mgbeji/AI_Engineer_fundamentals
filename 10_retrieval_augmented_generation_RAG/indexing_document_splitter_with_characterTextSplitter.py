from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters.character import CharacterTextSplitter

docx_loader = Docx2txtLoader(r"10_retrieval_augmented_generation_RAG/docs/Introduction_to_Data_and_Data_Science.docx")
docx_pages = docx_loader.load()

# go through each page in the doc and remove new line characters
for page in docx_pages:
    page.page_content = " ".join(page.page_content.split())

splitter = CharacterTextSplitter(separator=".", chunk_size=500, chunk_overlap=50)

splitted_docx_pages = splitter.split_documents(docx_pages)

print(len(splitted_docx_pages))
print((splitted_docx_pages[0].page_content))