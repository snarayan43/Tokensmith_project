# Tokensmith_project
# Personalize TokenSmith Based on User Context and Needs by Adding a User Context Model #

TokenSmith is RAG system that is designed to answer questions from a textbook using a combination of dense vector search, sparse keyword search, and an LLM for answer generation. While TokenSmith does contain query contextualization, empirical testing revealed that it fails to resolve more unique type of follow up questions. This issue, along with lacking an understanding of the user, results in relatively generic answers.

The purpose of this extension is to introduce a lightweight, 2-stage personalization layer that leverages the user’s input text with an LLM to understand conversation better and provide stronger answers. The first stage includes a rolling window of the recent conversation to deal with follow up questions. The second stage maintains a structured user persona that anticipates user’s learning goals, skill level, and topics to focus by observing query behavior.

Together, these stages are meant to transform TokenSmith from a generic question answering tool to one that adapts to the user to be more helpful during a session.

# Changes include adding a folder called project to the repository with files context.py and persona.py. Changes also include updates to src/main.py to include the running the project files under the run_chat_session method.

