from __future__ import annotations
import os
from functools import cached_property
from typing import Optional

_DEFAULT_MODEL_PATH = os.environ.get(
    "REFINER_MODEL_PATH",
    "models/qwen2.5-3b-instruct-q4_k_m.gguf",
)

_N_THREADS: int = int(os.environ.get("REFINER_N_THREADS", "4"))
_N_CTX: int = 2048


class QueryRefiner:
    """
    refine_query    -- rewrites a raw query using persona + recent context.
    build_persona   -- analyses 5 queries to extract structured observations,
                       then merges those observations into the existing persona.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or _DEFAULT_MODEL_PATH
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Refinement model not found at '{self.model_path}'.\n"
                "Set REFINER_MODEL_PATH env-var or pass model_path= to QueryRefiner()."
            )

    @cached_property
    def _llm(self):
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python is not installed. "
                "Run: pip install llama-cpp-python"
            ) from exc
        print(f"[QueryRefiner] Loading model from {self.model_path} ...")
        return Llama(
            model_path=self.model_path,
            n_ctx=_N_CTX,
            n_threads=_N_THREADS,
            verbose=False,
        )

    def _generate(self, prompt: str, max_tokens: int = 256) -> str:
        output = self._llm(
            prompt,
            max_tokens=max_tokens,
            stop=["<|endoftext|>", "\n\n\n"],
            echo=False,
        )
        return output["choices"][0]["text"].strip()

    # ------------------------------------------------------------------
    # Query refinement
    # ------------------------------------------------------------------

    def refine_query(
        self,
        raw_query: str,
        persona: str,
        context_window: list,
    ) -> str:
        """
        Rewrite raw_query into a short retrieval string.

        context_window  -- last up-to-5 turns (query/answer dicts), used to
                           resolve pronouns and follow-up references.
        persona         -- background style/level context only; never used
                           to inject previous topics into the new query.
        """
        # Build a compact recent-conversation block
        ctx_lines = ""
        if context_window:
            pairs = [f"  Q: {t['query']}\n  A: {t['answer'][:120]}..." for t in context_window[-3:]]
            ctx_lines = "### Recent conversation (for pronoun/reference resolution only)\n" + "\n".join(pairs) + "\n\n"

        prompt = (
            f"{ctx_lines}"
            "### User Persona (style/level context only)\n"
            f"{persona}\n\n"
            "### Raw Query\n"
            f"{raw_query}\n\n"
            "### Task\n"
            "Rewrite the Raw Query into a short search string for FAISS/BM25.\n"
            "Rules:\n"
            "- Output ONE line only: the search string, nothing else.\n"
            "- No explanations, headings, bullet points, or repeated output.\n"
            "- Use Recent conversation ONLY to resolve ambiguous references (e.g. 'it', 'that').\n"
            "- Use Persona ONLY to adjust vocabulary/technical level.\n"
            "- Never inject previous topics unless the Raw Query explicitly refers to them.\n"
            "- If the Raw Query is already clear and self-contained, return it with minimal changes.\n\n"
            "Refined search string:"
        )
        refined = self._generate(prompt, max_tokens=60)

        if not refined or len(refined) < 3:
            return raw_query

        refined = refined.splitlines()[0].strip()

        for prefix in ("refined search string:", "refined query:", "query:", "answer:"):
            if refined.lower().startswith(prefix):
                refined = refined[len(prefix):].strip()

        if len(refined) > 120:
            return raw_query

        return refined

    # ------------------------------------------------------------------
    # Persona building
    # ------------------------------------------------------------------

    def build_persona(self, recent_turns: list, current_persona: str) -> str:
        """
        Two-step persona update:

        Step 1 -- Analyse the 5 most recent queries and extract structured
                  observations: topics, level, and question style (definition /
                  example / comparison / deep-dive).

        Step 2 -- Merge those observations into the existing persona.
                  Fields that have changed are updated; fields with no new
                  evidence are kept from the old persona. The result stays
                  concise -- no elongation.
        """
        queries = "\n".join(f"- {t['query']}" for t in recent_turns)

        # Step 1: extract structured observations from the 5 queries
        analysis_prompt = (
            "Analyse these 5 user queries and output a structured observation.\n\n"
            f"Queries:\n{queries}\n\n"
            "Output exactly these four labelled lines and nothing else:\n"
            "Topics: <comma-separated topic areas>\n"
            "Level: <one of: beginner / intermediate / advanced>\n"
            "Style: <one or more of: definitions, examples, comparisons, deep-dive>\n"
            "Pattern: <one sentence describing the overall question pattern>\n"
        )
        analysis = self._generate(analysis_prompt, max_tokens=120)

        # Step 2: merge observations into existing persona
        merge_prompt = (
            "### Existing Persona\n"
            f"{current_persona}\n\n"
            "### New Observations from last 5 queries\n"
            f"{analysis}\n\n"
            "### Task\n"
            "Rewrite the Existing Persona by merging in the New Observations.\n"
            "Rules:\n"
            "- Update only the fields (topics, level, style, pattern) where the "
            "new observations provide clear evidence of change.\n"
            "- Keep fields from the Existing Persona where the new observations "
            "are silent or ambiguous.\n"
            "- Do NOT concatenate or append -- produce one concise merged paragraph "
            "under 120 words.\n"
            "- Output the merged persona text only. No labels, no preamble.\n\n"
            "Merged persona:"
        )
        merged = self._generate(merge_prompt, max_tokens=200)

        if not merged or len(merged) < 10:
            return current_persona
        merged = merged.strip()
        for suffix in ("### merged persona", "### updated persona", "###"):
            if merged.lower().endswith(suffix):
                merged = merged[: -len(suffix)].strip()

        return merged
