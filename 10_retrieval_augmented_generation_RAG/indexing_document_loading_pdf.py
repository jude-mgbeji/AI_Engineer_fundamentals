from langchain_community.document_loaders import PyPDFLoader
import copy

pdf_loader = PyPDFLoader(r"10_retrieval_augmented_generation_RAG/docs/Introduction_to_Data_and_Data_Science.pdf")
pdf_pages = pdf_loader.load()
print(len(pdf_pages))
print((pdf_pages[0]))

# Notice that there are new line characters in the text that affect the number of tokens in our LLM 
# which subsequently increases cost. To remove the new line characters, we;

# create a copy of the pdf to avoid overriding
pdf_pages_copy = copy.deepcopy(pdf_pages)

# go through each page in the pdf
for page in pdf_pages_copy:
    page.page_content = " ".join(page.page_content.split())

print((pdf_pages_copy[0]))

