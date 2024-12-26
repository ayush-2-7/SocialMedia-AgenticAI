import streamlit as st
import textwrap
from enum import Enum
from typing import List, Literal, Optional, TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

# Constants for prompts
EDITOR_PROMPT = """
Rewrite for maximum social media engagement:

- Use attention-grabbing, concise language
- Inject personality and humor
- Optimize formatting (short paragraphs)
- Encourage interaction (questions, calls-to-action)
- Ensure perfect grammar and spelling
- Rewrite from first person perspective, when talking to an audience

Use only the information provided in the text. Think carefully.
"""

TWITTER_PROMPT = """
Generate a high-engagement tweet from the given text (atleast 300 words):
1. What problem does this solve?
2. Focus on the main technical points/features
3. Write a short, coherent paragraph (2-3 sentences max)
4. Use natural, conversational language
5. Optimize for virality: make it intriguing, relatable, or controversial
6. Exclude emojis and hashtags
"""

TWITTER_CRITIQUE_PROMPT = """
You are a Tweet Critique Agent. Your task is to analyze tweets and provide actionable feedback to make them more engaging. Focus on:

1. Clarity: Is the message clear and easy to understand?
2. Hook: Does it grab attention in the first few words?
3. Brevity: Is it concise while maintaining impact?
4. Call-to-action: Does it encourage interaction or sharing?
5. Tone: Is it appropriate for the intended audience?
6. Storytelling: Does it evoke curiosity?
7. Remove hype: Does it promise more than it delivers?

Provide 2-3 specific suggestions to improve the tweet's engagement potential.
Do not suggest hashtags. Keep your feedback concise and actionable.
"""

LINKEDIN_PROMPT = """
Write a compelling LinkedIn post from the given text (atleast 300 words). Structure it as follows:

1. Eye-catching headline (5-7 words)
2. Identify a key problem or challenge
3. Provide a bullet list of key benefits/features
4. Highlight a clear benefit or solution
5. Conclude with a thought-provoking question

Maintain a professional, informative tone. Avoid emojis and hashtags.
Keep the post concise (50-80 words) and relevant to the industry.
"""

LINKEDIN_CRITIQUE_PROMPT = """
Your role is to analyze LinkedIn posts and provide actionable feedback to make them more engaging.
Focus on the following aspects:

1. Hook: Evaluate the opening line's ability to grab attention.
2. Structure: Assess the post's flow and readability.
3. Content value: Determine if the post provides useful information or insights.
4. Call-to-action: Check if there's a clear next step for readers.
5. Language: Suggest improvements in tone, style, and word choice.
6. Visual elements: Recommend additions or changes to images, videos, or formatting.

For each aspect, provide:
- A brief assessment (1-2 sentences)
- A specific suggestion for improvement
- A concise example of the suggested change
"""

# Pydantic models
class Post(BaseModel):
    """A post written in different versions"""
    drafts: List[str]
    feedback: Optional[str]

class AppState(TypedDict):
    user_text: str
    target_audience: str
    edit_text: str
    tweet: Post
    linkedin_post: Post
    n_drafts: int

# Graph nodes
def editor_node(state: AppState, llm):
    prompt = f"""
text:
```
{state["user_text"]}
```
""".strip()
    response = llm.invoke([SystemMessage(EDITOR_PROMPT), HumanMessage(prompt)])
    return {"edit_text": response.content}

def tweet_writer_node(state: AppState, llm):
    post = state["tweet"]
    feedback_prompt = (
        ""
        if not post.feedback
        else f"""
Tweet:
```
{post.drafts[-1]}
```

Use the feedback to improve it:
```
{post.feedback}
```
""".strip()
    )

    prompt = f"""
text:
```
{state["edit_text"]}
```

{feedback_prompt}

Target audience: {state["target_audience"]}

Write only the text for the post
""".strip()

    response = llm.invoke([SystemMessage(TWITTER_PROMPT), HumanMessage(prompt)])
    post.drafts.append(response.content)
    return {"tweet": post}

def linkedin_writer_node(state: AppState, llm):
    post = state["linkedin_post"]
    feedback_prompt = (
        ""
        if not post.feedback
        else f"""
LinkedIn post:
```
{post.drafts[-1]}
```

Use the feedback to improve it:
```
{post.feedback}
```
""".strip()
    )

    prompt = f"""
text:
```
{state["edit_text"]}
```

{feedback_prompt}

Target audience: {state["target_audience"]}

Write only the text for the post
""".strip()

    response = llm.invoke([SystemMessage(LINKEDIN_PROMPT), HumanMessage(prompt)])
    post.drafts.append(response.content)
    return {"linkedin_post": post}

def critique_tweet_node(state: AppState, llm):
    post = state["tweet"]
    prompt = f"""
Full post:
```
{state["edit_text"]}
```

Suggested tweet (critique this):
```
{post.drafts[-1]}
```

Target audience: {state["target_audience"]}
""".strip()

    response = llm.invoke([SystemMessage(TWITTER_CRITIQUE_PROMPT), HumanMessage(prompt)])
    post.feedback = response.content
    return {"tweet": post}

def critique_linkedin_node(state: AppState, llm):
    post = state["linkedin_post"]
    prompt = f"""
Full post:
```
{state["edit_text"]}
```

Suggested LinkedIn post (critique this):
```
{post.drafts[-1]}
```

Target audience: {state["target_audience"]}
""".strip()

    response = llm.invoke([SystemMessage(LINKEDIN_CRITIQUE_PROMPT), HumanMessage(prompt)])
    post.feedback = response.content
    return {"linkedin_post": post}

def build_graph(llm):
    def should_rewrite(state: AppState) -> Literal[["linkedin_critique", "tweet_critique"], END]:
        tweet = state["tweet"]
        linkedin_post = state["linkedin_post"]
        n_drafts = state["n_drafts"]
        if len(tweet.drafts) >= n_drafts and len(linkedin_post.drafts) >= n_drafts:
            return END
        return ["linkedin_critique", "tweet_critique"]

    graph = StateGraph(AppState)

    # Add nodes with llm parameter
    graph.add_node("editor", lambda x: editor_node(x, llm))
    graph.add_node("tweet_writer", lambda x: tweet_writer_node(x, llm))
    graph.add_node("tweet_critique", lambda x: critique_tweet_node(x, llm))
    graph.add_node("linkedin_writer", lambda x: linkedin_writer_node(x, llm))
    graph.add_node("linkedin_critique", lambda x: critique_linkedin_node(x, llm))
    graph.add_node("supervisor", lambda x: x)

    # Add edges
    graph.add_edge("editor", "tweet_writer")
    graph.add_edge("editor", "linkedin_writer")
    graph.add_edge("tweet_writer", "supervisor")
    graph.add_edge("linkedin_writer", "supervisor")
    graph.add_conditional_edges("supervisor", should_rewrite)
    graph.add_edge("tweet_critique", "tweet_writer")
    graph.add_edge("linkedin_critique", "linkedin_writer")

    graph.set_entry_point("editor")
    return graph.compile()

def display_drafts(post: Post, platform: str):
    """Helper function to display drafts and feedback for a platform"""
    for i, draft in enumerate(post.drafts):
        with st.expander(f"{platform} Draft #{i+1}"):
            if platform == "Twitter":
                st.write(textwrap.fill(draft, 80))
            else:
                st.write(draft)
    
    if post.feedback:
        with st.expander(f"{platform} Feedback"):
            st.write(post.feedback)

def main():
    st.set_page_config(page_title="Social Media Post Generator", layout="wide")
    
    st.title("Social Media Post Generator (Twitter and LinkedIn)")
    st.write("Generate engaging social media posts using AI")

    # User inputs
    with st.sidebar:
        groq_api_key = st.text_input("Enter your Groq API Key to access Llama 3.3-70b-Versatile model", type="password")
        n_drafts = st.number_input("Number of drafts to generate", min_value=1, max_value=5, value=1)
        target_audience = st.text_input("Target Audience", value="AI/ML engineers and researchers, Data Scientists")
        st.subheader("Agent Diagram")
        st.image("socialmedi_agenticAI.png", caption="Different Agents for Social Media Post Generation", use_container_width=True)


    # Main content
    user_text = st.text_area("Enter your text to convert into social media posts", height=200, max_chars=1000)

    if st.button("Generate Posts") and groq_api_key and user_text:
        try:
            with st.spinner("Generating posts..."):
                # Initialize LLM
                llm = ChatGroq(
                    temperature=0.7,
                    model_name="llama-3.3-70b-versatile",
                    api_key=groq_api_key
                )

                # Build and run graph
                app = build_graph(llm)
                state = app.invoke(
                    {
                        "user_text": user_text,
                        "target_audience": target_audience,
                        "tweet": Post(drafts=[], feedback=None),
                        "linkedin_post": Post(drafts=[], feedback=None),
                        "n_drafts": n_drafts,
                    }
                )

                # Store state in session state for tab display
                st.session_state['generated_state'] = state
                st.session_state['show_results'] = True

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # Display results if available
    if st.session_state.get('show_results', False):
        state = st.session_state['generated_state']
        
        st.subheader("Edited Text")
        with st.expander("View edited text"):
            st.write(state["edit_text"])

        # Create tabs for Twitter and LinkedIn content
        tab1, tab2 = st.tabs(["Twitter Content", "LinkedIn Content"])
        
        with tab1:
            st.subheader("Twitter Drafts")
            display_drafts(state["tweet"], "Twitter")
            
        with tab2:
            st.subheader("LinkedIn Drafts")
            display_drafts(state["linkedin_post"], "LinkedIn")

if __name__ == "__main__":
    main()
