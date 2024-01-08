from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser

output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()

# TEXT BASED
signature_dish_text_prompt = PromptTemplate(
        input_variables=["food"],
        template="""Act as a highly specialized vegetarian chef. You have the following ingredients to use:
        {food}
    
        Question: Create a dish using only these ingredients and say what it is called. Only return the name of the dish.
        No additional explanation or anything related. Just the name
        
        Example:
        'Pizza'
        'Sushi'

        Output:
    """
    )

creative_name_gen_prompt = PromptTemplate(
        input_variables=["cuisine", "cost"],
        template="""Act as a person who completely understands the restaurant naming business and are known to come up with awesome names for new restaurants.
    
        Question: Suggest a creative name for a {cuisine} restaurant that is very {cost} in cost.
        No additional explanation or anything related. Just the name
        
        Example:
        'Gyro Bros'
        'The Melting Pot'

        Output:
    """
    )

retrieval_prompt = PromptTemplate(
        input_variables=["summaries", "question"],
        partial_variables={"format_instructions": format_instructions},
        template="""
        You are a food blogger, helping people find restaurants in the city
        Generate your response by following the steps below:
        1. Break down the question to understand the cuisine and cost of the restaurants 
        2. Select the most relevant information from the context in light of the summary
        3. Generate 5 restaurant names using the selected information
        4. Remove duplicate content from the draft response
        5. Generate your final response after adjusting it to increase accuracy and relevance
        6. Now only show your final response! Do not provide any explanations or details
    
    Summary: 
    {summaries}
    
    Question: {question}
    
    Example:
    1. The Indian Kitchen
    2. Maharaja Bhog

    Format Instructions:
    {format_instructions}

    Return a comma separated python list
    Output:
    """
    )

# VISION BASED
vision_prompt = "Identify all food ingredients on the this image which are food related. List the food item names as a comma separated python list."

signature_dish_image_prompt = PromptTemplate(
    input_variables=["dish_name"],
    template="A very expensive looking {dish_name}",)


res_logo_prompt = PromptTemplate(
    input_variables=["res_name"],
    template="A restaurant logo in English for {res_name}", 
)