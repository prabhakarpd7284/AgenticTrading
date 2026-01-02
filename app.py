import streamlit as st

# --- Import your tools & graph ---
from main import (
    fetch_intraday_data_tool,
    fetch_data_from_broker,
    fetch_portfolio_from_broker,
    create_file,
    write_to_file,
    read_file,
    list_files,
    delete_file,
    create_folder,
    llm,
    graph
)
from langgraph.prebuilt import ToolNode

# -------------------------------------------------------------------
# Tool groups
# -------------------------------------------------------------------
FILE_TOOLS = [
    create_folder,
    create_file,
    write_to_file,
    read_file,
    list_files,
    delete_file
]

BROKER_TOOLS = [
    fetch_data_from_broker,
    fetch_portfolio_from_broker
]

DATA_TOOLS = [
    fetch_intraday_data_tool
]

ALL_GROUPS = {
    "File Tools": FILE_TOOLS,
    "Broker Tools": BROKER_TOOLS,
    "Market Data Tools": DATA_TOOLS
}

# UI Title
st.title("🔧 LLM Tool Runner UI")

# -------------------------------------------------------------------
# Prompt Input
# -------------------------------------------------------------------
user_prompt = st.text_area("Enter your prompt:", height=150)

# -------------------------------------------------------------------
# Tool Selection UI
# -------------------------------------------------------------------
st.subheader("Select tools to allow the LLM to use:")

selected_tools = []

for group_name, tool_list in ALL_GROUPS.items():
    with st.expander(group_name, expanded=False):
        selected = st.multiselect(
            f"Select tools from {group_name}",
            options=tool_list,
            format_func=lambda f: f.name,
            key=group_name
        )
        selected_tools.extend(selected)


# -------------------------------------------------------------------
# Run LLM + Tools Graph
# -------------------------------------------------------------------
if st.button("Run LLM"):
    if not user_prompt.strip():
        st.warning("Enter a prompt first.")
    else:
        # Bind LLM with selected tools
        llm_with_selected_tools = llm.bind_tools(selected_tools)

        # Run the graph
        state = {"messages": [user_prompt]}
        result = graph.invoke(state, config={"recursion_limit": 125})

        # Display Output
        st.subheader("LLM Output")
        st.write(result["messages"][-1].content)

        st.subheader("Raw State")
        st.json(result)
