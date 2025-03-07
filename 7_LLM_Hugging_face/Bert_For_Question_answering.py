from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import torch

model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

question = "What is the capital of France?"
context_document = "France is a country whose capital is Paris"

encoding = tokenizer.encode_plus(question, context_document)
print(encoding)
input_ids = encoding["input_ids"]
token_type_ids = encoding["token_type_ids"]
tokens = tokenizer.convert_ids_to_tokens(input_ids)

output = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
print(output)
start_scores = output.start_logits
end_scores = output.end_logits

start_index = torch.argmax(start_scores)
end_index = torch.argmax(end_scores)
answer = " ".join(tokens[start_index:end_index+1])
print(answer)
# Output: Paris

# The model outputs two scores for each token: one for the start of the answer and one for the end of the answer.
# The model predicts the start and end of the answer by selecting the token with the highest score for each.
# The start and end indices are then used to extract the answer from the input tokens.

# Building a simple FAQ chatbot using BERT

sunset_motors_context = "Sunset Motors is a renowned automobile dealership that has been a cornerstone of the automotive industry since its establishment in 1978. Located in the picturesque town of Crestwood, nestled in the heart of California's scenic Central Valley, Sunset Motors has built a reputation for excellence, reliability, and customer satisfaction over the past four decades. Founded by visionary entrepreneur Robert Anderson, Sunset Motors began as a humble, family-owned business with a small lot of used cars. However, under Anderson's leadership and commitment to quality, it quickly evolved into a thriving dealership offering a wide range of vehicles from various manufacturers. Today, the dealership spans over 10 acres, showcasing a vast inventory of new and pre-owned cars, trucks, SUVs, and luxury vehicles. One of Sunset Motors' standout features is its dedication to sustainability. In 2010, the dealership made a landmark decision to incorporate environmentally friendly practices, including solar panels to power the facility, energy-efficient lighting, and a comprehensive recycling program. This commitment to eco-consciousness has earned Sunset Motors recognition as an industry leader in sustainable automotive retail. Sunset Motors proudly offers a diverse range of vehicles, including popular brands like Ford, Toyota, Honda, Chevrolet, and BMW, catering to a wide spectrum of tastes and preferences. In addition to its outstanding vehicle selection, Sunset Motors offers flexible financing options, allowing customers to secure affordable loans and leases with competitive interest rates. The dealership's experienced sales team is committed to providing exceptional service and guidance to customers throughout the purchasing process, ensuring a seamless and enjoyable buying experience. Sunset Motors also boasts a state-of-the-art service center staffed by certified technicians who specialize in maintenance, repairs, and diagnostics for all makes and models. Whether customers require routine maintenance, major repairs, or specialized services, Sunset Motors' service department delivers top-notch care and expertise. Sunset Motors' commitment to customer satisfaction extends beyond the showroom and service center, with a dedicated parts department offering genuine OEM parts and accessories for all vehicle types. Customers can trust Sunset Motors to provide high-quality components that meet the manufacturer's specifications, ensuring optimal performance and longevity for their vehicles. Sunset Motors' dedication to excellence, sustainability, and customer service has solidified its reputation as a trusted and respected dealership in the automotive industry. With a rich history of success and a forward-thinking approach to business, Sunset Motors continues to set the standard for automotive excellence in the Central Valley and beyond."
print(sunset_motors_context)

def faq_bot(question):
    context = sunset_motors_context
    encoding = tokenizer.encode_plus(question, context)
    input_ids = encoding["input_ids"]
    token_type_ids = encoding["token_type_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    output = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
    start_scores = output.start_logits
    end_scores = output.end_logits

    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)
    if end_index >= start_index:
        answer = " ".join(tokens[start_index:end_index+1])
    else:
        answer = "Sorry, I couldn't find an answer to that question."
    # remove ## from answer
    answer = answer.replace(" ##", "")
    return answer

print(faq_bot("Where is Sunset Motors located?"))
print(faq_bot("What make of cars are available"))
print(faq_bot("How large is the dealership?"))

