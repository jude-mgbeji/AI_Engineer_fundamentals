import streamlit as st

st.title("This is a tittle text")

# text wrapped in underscore indicate itallic formatting
#  a colon and square bracket is change the text color
# the last part is an emoji short code
st.title("_This_ is :blue[a tittle] text :speech_balloon:")

# We can also format mathematical equations by wrapping them in dollar signs
st.title("$E = mc^2$")

st.header("This is a header text")

st.subheader("This is a subheader text")

# text with no formatting
st.text("This is a text text")

st.markdown("# This is a header text \n **This is a bold text** \n - This is a list item 1 \n - This is a list item 2 \n - This is a list item 3 \n - This is a list item 4 \n - This is a list item 5")

st.write("This is a text using the write function")

data = {
    "name": "John Doe",
    "age": 30,
    "city": "New York"
}
st.write(data)