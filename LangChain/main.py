import streamlit as st
from langchain.chains import SequentialChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub

# Directly set your Hugging Face API token
api_token = "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"  # Replace with your actual API token

# Define your model repository ID
repo_id = "google/gemma-2-27b-it"  # Replace with your model's repo ID

# Initialize the Hugging Face model with the API token
llm = HuggingFaceHub(repo_id=repo_id, huggingfacehub_api_token=api_token)

def generate_restaurant_info(country):
    # Create a prompt for restaurant name generation
    name_prompt = PromptTemplate(
        input_variables=["country"],
        template="I want to open a restaurant for {country} food. Suggest a fancy name for this."
    )

    # Create a prompt for menu item generation
    menu_prompt = PromptTemplate(
        input_variables=["restaurant_name"],
        template="Suggest some menu items for a restaurant named {restaurant_name}. Return it as a comma-separated string."
    )

    # Create individual chains
    name_chain = LLMChain(llm=llm, prompt=name_prompt, output_key="restaurant_name")
    menu_chain = LLMChain(llm=llm, prompt=menu_prompt, output_key="menu_items")

    # Define the sequential chain
    chain = SequentialChain(
        chains=[name_chain, menu_chain],
        input_variables=["country"],
        output_variables=["restaurant_name", "menu_items"],
        verbose=True  # Enable verbose mode to debug chain execution
    )

    # Run the chain
    result = chain({"country": country})

    # Clean up the outputs
    restaurant_name = result["restaurant_name"].strip()
    menu_items = result["menu_items"].strip()

    return restaurant_name, menu_items

# Set up the Streamlit interface
st.title("Restaurant Name and Menu Generator")
st.title("🛠️Built using LangChain 🦜🔗 and HuggingFace 🤗")
country = st.text_input("Enter a country:")

if st.button("Generate"):
    if country:
        restaurant_name, menu_items = generate_restaurant_info(country)
        st.subheader("Generated Restaurant Name:")
        st.write(restaurant_name)
        st.subheader("Suggested Menu Items:")
        st.write(menu_items)
    else:
        st.warning("Please enter a country.")