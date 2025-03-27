from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter

docx_loader = Docx2txtLoader(r"10_retrieval_augmented_generation_RAG/docs/Introduction_to_Data_and_Data_Science 2.docx")
docx_pages = docx_loader.load()

print(len(docx_pages))

# This splitter splits the document based on headers in the doc, Identified by #, ##, ### etc.
# the number of # depicts the header level in the document and should be factored
#  into the tuple argument. 
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "course tittle"), ("##", "lecture tittle")])

splitted_docx_content = splitter.split_text(docx_pages[0].page_content)

print(len(splitted_docx_content))
print((splitted_docx_content))

# go through each page in the doc and remove new line characters
for content in splitted_docx_content:
    content.page_content = " ".join(content.page_content.split())

print(len(splitted_docx_content))
print((splitted_docx_content[0]))