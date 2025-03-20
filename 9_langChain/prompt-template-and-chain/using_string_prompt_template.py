from langchain_core.prompts import PromptTemplate

TEMPLATE = '''
System:
{description}

Human:
I've recently adopted a {pet},
Could you suggest some {pet} names?
'''

prompt_template = PromptTemplate.from_template(template=TEMPLATE)

# the invoke function here takes in a dictionary that defines the variables in the template
prompt_value = prompt_template.invoke({"description": '''The chatbot should relunctantly answer questions
                                    with sarcastic responses.''',
                                        'pet':'dog'})
print(prompt_value)

