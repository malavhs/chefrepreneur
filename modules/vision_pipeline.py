import base64
from utils import models, prompts
from langchain.schema.messages import HumanMessage, AIMessage
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain.chains import SequentialChain, LLMChain, RetrievalQA, RetrievalQAWithSourcesChain


def encode_image(image_file):
    """_summary_

    Parameters
    ----------
    image_file : _type_
       image uploaded via UI

    Returns
    -------
    _type_
        encoded image for Open AI Models
    """
    #with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def image_to_text(image_file, prompt_input):
    """_summary_

    Parameters
    ----------
    image_file : _type_
        Image file that you want to convert to text
    prompt_input : _type_
        Associated prompt in order to generate the desired text
    """
    image = encode_image(image_file)
    message = models.vision_model.invoke(
        [   HumanMessage(
                role = "system",
                content=[
                    {"type": "text", "text": prompt_input},
                    {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{image}"},},
                ]
            )
        ]
)
    response = message.content
    return response


def text_to_text(text, prompt_input):
    """_summary_

    Parameters
    ----------
    text : _type_
        Text that needs to be used to generate another form of text based on prompt
    prompt_input : _type_
        Associated Prompt

    Returns
    -------
    _type_
        Response based on the prompt
    """
    llm = LLMChain(llm=models.llm, prompt=prompt_input)
    chain = llm.run(text)
    return chain

def text_to_image(text, prompt_input):
    """This function takes in a prompt with the name of the signature dish and generates an image based on that using DALL e 3

    Parameters
    ----------
    image_file : .jpeg
        File uploaded via user interface

    Returns
    -------
    url
        ai generated image using DALL E 3 Model
    """
    chain = LLMChain(llm=models.llm, prompt=prompt_input)
    image_url = DallEAPIWrapper(model='dall-e-3').run(chain.run(f"{text}"))
    return image_url
