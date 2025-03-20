from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import chain


# The RunnableLambda class allows us to convert any function into a Runnable 
# that cab theb invoke and enter as a chain component

sum_number = lambda x: sum(x)
print(sum_number([1,2,3]))

muliply_by_two = lambda x: x*2

print(muliply_by_two(5))

# in order to convert this functions into runnables,
sum_number_runnable = RunnableLambda(sum_number)
print(sum_number_runnable.invoke([1,2,3]))


muliply_by_two_runnable = RunnableLambda(muliply_by_two)
print(muliply_by_two_runnable.invoke(5))

# we can then create a chain, where the output of the sum becomes the input of the multiply
chain1 = sum_number_runnable | muliply_by_two_runnable

print(chain1.invoke([1,2,3]))

# Alternatively
# in order to convert a function to a runnable we can use a decorator annotation
#  directly on the function as shown:

@chain
def sum_runnable(x):
    return sum(x)

@chain
def multiply_runnable(x):
    return x * 2

chain2 = sum_runnable | multiply_runnable

print(chain2.invoke([1,2,3]))






