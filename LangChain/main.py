import streamlit as st  # Import the Streamlit library for creating web apps
from langchain.llms import HuggingFaceHub  # Import the HuggingFaceHub class from LangChain for using Hugging Face models
from langchain.prompts import PromptTemplate  # Import the PromptTemplate class from LangChain for creating prompts

# Directly set your Hugging Face API token
api_token = "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"  # Replace with your actual API token

# Define your model repository ID
repo_id = "google/gemma-2-27b-it"  # Replace with your model's repo ID

# Initialize the Hugging Face model with the API token
llm = HuggingFaceHub(repo_id=repo_id, huggingfacehub_api_token=api_token)

# Define a function to generate restaurant names and menu items
def generate_restaurant_info(country):
    # Create a prompt for restaurant name generation
    name_prompt = PromptTemplate(
        input_variables=["country"],
        template="Generate a creative restaurant name for a restaurant in {country}."
    )

    # Create a prompt for menu item generation
    menu_prompt = PromptTemplate(
        input_variables=["country"],
        template="Generate a list of popular menu items for a restaurant in {country}."
    )

    # Generate restaurant name
    restaurant_name = llm(name_prompt.format(country=country))

    # Generate menu items
    menu_items = llm(menu_prompt.format(country=country))

    return restaurant_name, menu_items


# Set up the Streamlit interface
st.title("Restaurant Name and Menu Generator")  # Set the main title of the Streamlit app
st.title("🛠️Built using LangChain 🦜🔗 and HuggingFace 🤗")  # Set a subtitle with emojis for visual appeal
country = st.text_input("Enter a country:")  # Create a text input field for the user to enter a country

if st.button("Generate"):  # Create a button that triggers the generation process
    if country:  # Check if the user has entered a country
        restaurant_name, menu_items = generate_restaurant_info(country)  # Generate restaurant name and menu items
        st.subheader("Generated Restaurant Name:")  # Display a subheader for the restaurant name
        st.write(restaurant_name)  # Display the generated restaurant name
        st.subheader("Suggested Menu Items:")  # Display a subheader for the menu items
        st.write(menu_items)  # Display the generated menu items
    else:
        st.warning("Please enter a country.")  # Display a warning if no country is entered
