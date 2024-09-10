import streamlit as st
from streamlit_chat import message
from neural_db_loader import generate_queries_chatgpt, search_neural_db, reciprocal_rank_fusion, generate_answers

# Initialize session state to store the conversation
if 'responses' not in st.session_state:
    st.session_state['responses'] = []
    
if 'requests' not in st.session_state:
    st.session_state['requests'] = [] 

# Streamlit Title
st.title("Insurance Bot")

# Check if the initial prompt needs to be added
if len(st.session_state['requests']) == 0 and len(st.session_state['responses']) == 0:
    st.session_state['responses'].append("How can I assist you with your insurance queries?")

# Temporary variable to hold the user input
user_input = st.chat_input("Do you have any questions?", key="unique_query_input")

if user_input:
    st.session_state['requests'].append(user_input)

    # Generate multiple related queries using ChatGPT
    query_list = generate_queries_chatgpt(user_input)
    
    # Search for each query in NeuralDB and collect results
    reference_list = []
    for query in query_list:
        search_results = search_neural_db(query)
        reference_list.append(search_results)
    
    # Apply Reciprocal Rank Fusion (RRF) to combine and rerank the results
    r = reciprocal_rank_fusion(dict(zip(query_list, reference_list)))
    
    # Get ranked reference list
    ranked_reference_list = [i for i in r.keys()]
    
    # Get the final answer using the references
    ans = generate_answers(user_input, ranked_reference_list)
    
    # Append the bot's answer to responses
    st.session_state['responses'].append(ans)
    
    # Clear the input field after submission
    st.experimental_rerun()  # Optional: This forces the page to refresh to clear the input field

# Display the conversation history like a chat interface
for i in range(len(st.session_state['responses'])):
    # Display bot response on the left
    message(st.session_state['responses'][i], key=str(i) + '_bot')
    
    # Display user message on the right, only if the user has submitted a query
    if i < len(st.session_state['requests']):
        message(st.session_state['requests'][i], is_user=True, key=str(i) + '_user')