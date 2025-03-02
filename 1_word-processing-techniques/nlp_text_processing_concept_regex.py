import re

# Ideally in python, \n using indicates new line, assuming we want 
# to define a string tha has such chararcter, there are several we can do this
# The r in from of the string indicates to python that it is a raw string
folder = r"C:desktop\notes"
print(folder)

# Using Regex to search for certain patterns or strings within a string
result_search = re.search("pattern", r"string to obtain the pattern")
# returns the pattern if found or None if not found
print(result_search)

# In order to search and replace a particular pattern from a string we use 
string = r"sara was ablt to help me find the items I nneded quickly"
new_string = re.sub("sara", "sarah", string)
print(new_string)

