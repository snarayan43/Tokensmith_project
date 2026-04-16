import ollama # Assuming Ollama for local LLM execution

class QueryRefiner:
    def __init__(self, model_name="llama3.2:3b"):
        self.model = model_name

    def refine_query(self, raw_query, persona):
        prompt = f"""
        USER PERSONA:
        {persona}

        TASK:
        Refine the following user query for a vector database search (FAISS/BM25). 
        Incorporate relevant context from the User Persona if it helps clarify intent.
        Keep the output as a single, optimized search string.

        RAW QUERY: {raw_query}
        REFINED QUERY:"""
        
        response = ollama.generate(model=self.model, prompt=prompt)
        return response['response'].strip()

    def summarize_persona(self, history_tail, current_persona):
        # Only take the last 5 entries for the update
        recent_context = json.dumps(history_tail[-5:], indent=2)
        
        prompt = f"""
        EXISTING PERSONA: {current_persona}
        RECENT INTERACTIONS: {recent_context}

        TASK: 
        Update the User Persona based on the recent interactions. 
        Note their technical level, recurring topics, and preferred style.
        Be concise but thorough. Output only the updated persona text.
        """
        
        response = ollama.generate(model=self.model, prompt=prompt)
        return response['response'].strip()