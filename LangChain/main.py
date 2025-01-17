import streamlit as st  # Import the Streamlit library for creating web applications
from langchain.chains import SequentialChain, LLMChain  # Import SequentialChain and LLMChain for chaining language model calls
from langchain.prompts import PromptTemplate  # Import PromptTemplate to create structured prompts for the language model
from langchain.llms import HuggingFaceHub  # Import HuggingFaceHub to interface with Hugging Face models

# Directly set your Hugging Face API token
api_token = "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"  # Replace with your actual API token for Hugging Face

# Define your model repository ID
repo_id = "google/gemma-2-27b-it"  # Replace with your model's repository ID on Hugging Face

# Initialize the Hugging Face model with the API token
llm = HuggingFaceHub(repo_id=repo_id, huggingfacehub_api_token=api_token)  # Create an instance of the language model using the specified repo ID and API token

def generate_restaurant_info(country):
    """
    Generate a restaurant name and menu items based on the specified country.

    Parameters:
    country (str): The name of the country for which to generate restaurant information.

    Returns:
    tuple: A tuple containing the generated restaurant name and a comma-separated string of menu items.
    """
    
    # Create a prompt for restaurant name generation
    name_prompt = PromptTemplate(
        input_variables=["country"],  # Specify that the input variable is 'country'
        template="I want to open a restaurant for {country} food. Suggest a fancy name for this."  # Template for generating a restaurant name
    )

    # Create a prompt for menu item generation
    menu_prompt = PromptTemplate(
        input_variables=["restaurant_name"],  # Specify that the input variable is 'restaurant_name'
        template="Suggest some menu items for a restaurant named {restaurant_name}. Return it as a comma-separated string."  # Template for generating menu items
    )

    # Create individual chains for name and menu item generation
    name_chain = LLMChain(llm=llm, prompt=name_prompt, output_key="restaurant_name")  # Chain for generating restaurant name
    menu_chain = LLMChain(llm=llm, prompt=menu_prompt, output_key="menu_items")  # Chain for generating menu items

    # Define the sequential chain that first generates the restaurant name and then the menu items
    chain = SequentialChain(
        chains=[name_chain, menu_chain],  # List of chains to execute in sequence
        input_variables=["country"],  # Input variable for the entire chain, which is 'country'
        output_variables=["restaurant_name", "menu_items"],  # Specify the expected outputs from the chain
        verbose=True  # Enable verbose mode for better debugging information during execution
    )

    # Run the chain with the provided country input
    result = chain({"country": country})  # Execute the chain and pass the country to it

    # Clean up the outputs by stripping any leading/trailing whitespace
    restaurant_name = result["restaurant_name"].strip()  # Get the restaurant name and remove extra whitespace
    menu_items = result["menu_items"].strip()  # Get the menu items and remove extra whitespace

    return restaurant_name, menu_items  # Return the generated restaurant name and menu items as a tuple

# Set up the Streamlit interface
st.title("Restaurant Name and Menu Generator")  # Set the main title of the Streamlit app
st.title("🛠️Built using LangChain 🦜🔗 and HuggingFace 🤗")  # Set a subtitle with emojis for visual appeal
country = st.text_input("Enter a country:")  # Create a text input field for the user to enter a country

# Button to trigger the generation process
if st.button("Generate"):  # Check if the "Generate" button is clicked
    if country:  # Check if the user has provided a country
        restaurant_name, menu_items = generate_restaurant_info(country)  # Call the function to generate restaurant info
        st.subheader("Generated Restaurant Name:")  # Set a subheader for displaying the restaurant name
        st.write(restaurant_name)  # Display the generated restaurant name on the app
        st.subheader("Suggested Menu Items:")  # Set a subheader for displaying the menu items
        st.write(menu_items)  # Display the suggested menu items on the app
    else:
        st.warning("Please enter a country.")  # Show a warning message if no country is entered
