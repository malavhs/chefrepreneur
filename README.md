# Background

With "Chef-repreneur," I aimed to push the boundaries of AI-driven creativity. Here's a glimpse of what this innovative tool can do:

1️⃣ Food Image Generation: 📸
Upload an image of your ingredients, specify the cuisine you want to create, and set your affordability level. "Chef-repreneur" harnesses the remarkable GPT-4 Vision model and the visionary DALL·E 3 model to generate a stunning food image that matches your selection.

2️⃣ Restaurant Name & Logo: 🏢
Ever wanted to start your own restaurant? "Chef-repreneur" can suggest a unique restaurant name and even generate a custom logo based on your inputs. It's like having an AI-driven branding partner!

3️⃣ Competitor Analysis: 📊
"Chef-repreneur" also offers a competitive edge. Upload a CSV with restaurant data, and it will use Retrieval Augmented Generation (RAG) to provide you with a list of existing competitors, giving you valuable insights for strategic planning.

# Tools used

1. OpenAI base and embedding models (GPT4, GPT4V, DALL E3 etc.)
2. Langchain framework
3. Pinecone Vector Database
4. Streamlit


# Steps to run

1. **Add your Pinecone and OpenAI API keys to bash_profile** 

```
export OPENAI_API_KEY= ""
export PINECONE_API_KEY= ""
```
and source it
```
source ~/.bash_profile
```

2. **Install requirements**

```
pip install -r requirements.txt
```

3. **Run Streamlit App**

```
streamlit run main.py
```