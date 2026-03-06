# ============================================================================
# WORKFLOW 4: WEB RESEARCH AND ONLINE INFORMATION SEARCH
# ============================================================================
# Module dedicated to online search for information about STM32 boards, AI models
# and optimization best practices
#
# Responsibilities:
#   - Classification of search type (ai_model, board_selection, optimization, documentation)
#   - Execution of searches via Google Search / LLM
#   - Formatting results for the user
#
# Dependencies: langgraph, langchain, agno.tools, requests

import os
import logging
from typing import Literal, Optional

from agno.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from src.assistant.configuration import Configuration
from src.assistant.state import MasterState

from agno.tools.duckduckgo import DuckDuckGoTools
from agno.models.ollama import Ollama
import asyncio

# ============================================================================
# DEEPEVAL INTEGRATION
# ============================================================================
# ===== CRITICAL: SET ENVIRONMENT VARIABLES BEFORE IMPORTING DEEPEVAL =====
os.environ["DEEPEVAL_RESULTS_FOLDER"] = "/tmp/deepeval"  # Writable also in Docker
os.environ["DEEPEVAL_DISABLE_TELEMETRY"] = "1"
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"
os.environ["DEEPEVAL_SKIP_PROMPTS_CACHE"] = "1"
os.environ["DEEPEVAL_CACHE_DIR"] = "/tmp/.deepeval"

def _evaluate_summary_sync(
    research_topic: str,
    running_summary: str,
    web_research_results: str
) -> dict:
    """
    Synchronous Evaluation in a separate thread.
    Uses metrics compatible with web search (Faithfulness, AnswerRelevancy).
    """
    triton_enabled = os.environ.get("USE_TRITON_BACKEND", "false").lower() == "true"
    backend_label = "Triton (deepseek-r1)" if triton_enabled else "Ollama (deepseek-r1:latest)"
    print(f"\n🔍 Running DeepEval evaluation with {backend_label}...\n")
    
    try:
        from deepeval.metrics import (
            FaithfulnessMetric,
            AnswerRelevancyMetric,
            ContextualRelevancyMetric,
            HallucinationMetric
        )
        from deepeval.test_case import LLMTestCase
        
        # -----------------------------------------------------------------------
        # EVALUATION MODEL: Custom DeepEval wrapper over get_llm
        # We leverage centralized routing (Triton/Ollama) bypassing the
        # strict 'OPENAI_API_KEY' constraint imposed by native GPTModel class.
        # -----------------------------------------------------------------------
        from src.assistant.utils import get_llm
        from deepeval.models import DeepEvalBaseLLM
        
        class DeepEvalLangChainWrapper(DeepEvalBaseLLM):
            def __init__(self, model_name: str, config: RunnableConfig = None):
                self.model_name = model_name
                # Temperature=0 for deterministic evaluation.
                # stop sequences prevent deepseek-r1 from appending Python test code 
                # after its JSON output (it writes "def test_extract...()" otherwise)
                self.llm = get_llm(config=config, model=model_name, temperature=0, num_predict=512,
                                   stop=["```", "\n```", "# tests", "def test_", "\n\nimport ", "\n\n#"])

            def get_model_name(self):
                return self.model_name

            def load_model(self):
                return self.llm

            def _clean_json(self, text: str) -> str:
                import re
                import json
                
                print(f"\n[DEEPEVAL RAW model]\n{text}\n[/DEEPEVAL RAW model]\n", flush=True) # for testing
                
                # 0. Normalize Python f-string double-braces {{ }} -> { }
                # DeepSeek-R1 was trained on StackOverflow f-string examples and sometimes
                # echoes back template double-braces instead of literal JSON braces.
                cleaned = text.replace("{{", "{").replace("}}", "}")
                
                # 1. Remove <think>...</think> completely (if any)
                cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL).strip()
                
                # 2. Try to find JSON using regex (supports nested objects/arrays)
                pattern = r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\])'
                matches = re.finditer(pattern, cleaned, re.DOTALL)
                
                for match in matches:
                    candidate = match.group(1)
                    try:
                        json.loads(candidate)
                        print(f"\n[DEEPEVAL EXTRACTED (Regex)]\n{candidate}\n[/DEEPEVAL EXTRACTED (Regex)]\n", flush=True) # for testing
                        return candidate
                    except json.JSONDecodeError:
                        continue
                        
                # 3. Fallback: scan backwards from last } bracket until we find valid JSON
                start_idx = min((cleaned.find(c) for c in '{[' if c in cleaned), default=-1)
                if start_idx != -1:
                    end_char = ']' if cleaned[start_idx] == '[' else '}'
                    end_idx = cleaned.rfind(end_char)
                    while end_idx > start_idx:
                        candidate = cleaned[start_idx:end_idx+1]
                        try:
                            json.loads(candidate)
                            print(f"\n[DEEPEVAL EXTRACTED (Fallback)]\n{candidate}\n[/DEEPEVAL EXTRACTED (Fallback)]\n", flush=True) # for testing
                            return candidate
                        except json.JSONDecodeError:
                            end_idx = cleaned.rfind(end_char, 0, end_idx)

                print(f"\n[DEEPEVAL FAILED TO FIND JSON]\n{cleaned}\n", flush=True)  # for testing
                # 4. Last resort: LLM returned plain prose (e.g. "The score is X because...")
                # Wrap it in {"reason": "..."} so DeepEval can parse it without crashing.
                return json.dumps({"reason": cleaned})

            def generate(self, prompt: str) -> str:
                res = self.llm.invoke(prompt)
                return self._clean_json(res.content)

            async def a_generate(self, prompt: str) -> str:
                res = await self.llm.ainvoke(prompt)
                return self._clean_json(res.content)

        # We use gpt-oss-20b: it's the most precise cod model on short structured
        # instructions. We keep it on short prompts (retrieval_ctx truncated below).
        eval_model_name = "gpt-oss-20b" if triton_enabled else "gpt-oss-20b"
        eval_model = DeepEvalLangChainWrapper(model_name=eval_model_name, config=None)

        
        # Define metrics 
        faithfulness = FaithfulnessMetric(threshold=0.55, model=eval_model, async_mode=False) 
        relevancy = AnswerRelevancyMetric(threshold=0.55, model=eval_model, async_mode=False)
        # ContextualRelevancy and Hallucination NOW ACTIVE
        contextual_relevancy = ContextualRelevancyMetric(threshold=0.55, model=eval_model, async_mode=False)
        hallucination = HallucinationMetric(threshold=0.55, model=eval_model, async_mode=False)
        
        # --- Context Truncation ---
        # Cap retrieval context to keep prompts SHORT for gpt-oss-20b.
        # Long contexts cause the model to hallucinate off-topic responses.
        MAX_CTX_ITEMS = 5       # max number of retrieved chunks to pass
        MAX_CTX_CHARS = 300     # max chars per chunk
        MAX_OUTPUT_CHARS = 800  # max chars of actual_output to evaluate

        if isinstance(web_research_results, str):
             retrieval_ctx = [chunk for chunk in web_research_results.split("\n\n") if len(chunk.strip()) > 50]
             if not retrieval_ctx: retrieval_ctx = [web_research_results[:MAX_CTX_CHARS]]
        else:
             retrieval_ctx = web_research_results

        # Apply length caps
        retrieval_ctx = [c[:MAX_CTX_CHARS] for c in retrieval_ctx[:MAX_CTX_ITEMS]]
        truncated_output = running_summary[:MAX_OUTPUT_CHARS]
        
        test_case = LLMTestCase(
            input=research_topic,
            actual_output=truncated_output,
            retrieval_context=retrieval_ctx,
            context=retrieval_ctx # Used by HallucinationMetric as "ground truth"
        )
        
        # Run evaluation manually to avoid 'evaluate()' blocking IO/cache overhead
        metrics_results = {}
        
        # 1. Faithfulness
        try:
            faithfulness.measure(test_case)
            metrics_results["faithfulness"] = faithfulness.score
        except Exception as e:
            print(f"Error measuring faithfulness: {e}")
            metrics_results["faithfulness"] = 0

        # 2. Answer Relevancy
        try:
            relevancy.measure(test_case)
            metrics_results["answer_relevancy"] = relevancy.score
        except Exception as e:
            print(f"Error measuring relevancy: {e}")
            metrics_results["answer_relevancy"] = 0

        # 3. Contextual Relevancy
        try:
            contextual_relevancy.measure(test_case)
            metrics_results["contextual_relevancy"] = contextual_relevancy.score
        except Exception as e:
            print(f"Error measuring contextual relevancy: {e}")
            metrics_results["contextual_relevancy"] = 0

        # 4. Hallucination
        try:
            hallucination.measure(test_case)
            metrics_results["hallucination"] = hallucination.score
        except Exception as e:
            print(f"Error measuring hallucination: {e}")
            metrics_results["hallucination"] = 0
        
        return {
            "completed": True,
            "metrics": metrics_results
        }
        
    except Exception as e:
        print(f"\n❌ Evaluation error: {e}")
        return {"completed": False, "error": str(e)}



logger = logging.getLogger(__name__)

# ============================================================================
# EXTRACTION SCHEMAS - WORKFLOW 4
# ============================================================================

class SearchClassification(BaseModel):
    """Classification of the requested search type"""
    search_type: Literal[
        "ai_model",
        "board_selection",
        "optimization",
        "documentation",
        "none"
    ] = Field(description="Type of search requested by the user")
    
    search_query: str = Field(
        description="Specific query to search online"
    )
    
    reasoning: str = Field(
        description="Explanation of the classification"
    )


# ============================================================================
# EXTRACTION INSTRUCTIONS - WORKFLOW 4
# ============================================================================

search_classification_instructions = """You are a search classifier for an STM32 + AI system.

Analyze the user's request and classify the type of search needed:

1. **ai_model**: Search for AI models compatible with STM32
   - Examples: "which lightweight CNN models for STM32?", "MobileNet vs SqueezeNet for STM32"
   - Keywords: model, network, CNN, RNN, neural network, artificial intelligence
   
2. **board_selection**: Help choose an STM32 board
   - Examples: "which STM32 for a project with AI?", "STM32H7 vs STM32F4"
   - Keywords: board, choice, which, difference, comparison, memory, performance
   
3. **optimization**: AI optimization and compression on STM32
   - Examples: "how to compress the model?", "quantization on STM32"
   - Keywords: optimization, quantization, compression, pruning, embedded
   
4. **documentation**: General documentation, tutorials, best practices
   - Examples: "how to compile for STM32?", "STEdgeAI guides", "tutorial"
   - Keywords: documentation, tutorial, guides, how to, best practice, resources

5. **none**: None of the above or invalid request
   - Examples: "hello", "I don't know", completely unrelated requests

ALWAYS respond in JSON format with three fields:
- "search_type": one of "ai_model", "board_selection", "optimization", "documentation", "none"
- "search_query": the query to search online (in English, detailed and specific)
- "reasoning": explanation of the classification (max 100 characters)

If search_type is "none", you can put search_query and reasoning as empty strings.

Examples:

Input: "Which lightweight models can I use for image classification on STM32H7?"
Output: {
  "search_type": "ai_model",
  "search_query": "lightweight image classification models STM32H7 embedded TensorFlow",
  "reasoning": "Explicit request for AI models for STM32, well-defined task"
}

Input: "Compare STM32F4 and STM32H7 for a project with AI inference"
Output: {
  "search_type": "board_selection",
  "search_query": "STM32F4 vs STM32H7 comparison memory performance AI inference",
  "reasoning": "Comparison between STM32 boards, focus on AI compatibility"
}

Input: "How to quantize a TensorFlow model for STM32?"
Output: {
  "search_type": "optimization",
  "search_query": "TensorFlow model quantization INT8 STM32 embedded optimization",
  "reasoning": "Question about optimization/compression techniques for embedded"
}

Input: "Where can I find the official STEdgeAI documentation?"
Output: {
  "search_type": "documentation",
  "search_query": "STEdgeAI official documentation tutorial guide STMicroelectronics",
  "reasoning": "Request for official documentation and resources"
}

Input: "Hello how are you?"
Output: {
  "search_type": "none",
  "search_query": "",
  "reasoning": "Request unrelated to the STM32+AI system"
}
"""


# ============================================================================
# DYNAMIC PROMPTS FOR SEARCH
# ============================================================================

SEARCH_PROMPTS = {
    "ai_model": """
Search for information about AI models compatible with STM32.
Query: {search_query}

For each model found, provide:
1. Model name
2. Framework (TensorFlow, PyTorch, ONNX, etc.)
3. Size in KB
4. STM32 Compatibility (which MCUs?)
5. Link to documentation
6. Recommended quantization level
7. Performance (inference time, accuracy)
8. Typical use cases

Be concise and practical for embedded developers.
    """,
    
    "board_selection": """
Search for information about STM32 boards.
Query: {search_query}

For each board found, provide:
1. Board name (e.g. STM32F4, STM32H7, STM32U5)
2. FLASH Memory (KB)
3. RAM (KB)
4. Clock speed (MHz)
5. Main peripherals (ADC, DAC, PWM, I2C, SPI, etc.)
6. Approximate price (USD)
7. Recommended use cases
8. Where to buy it (main distributors)

Compare at least 3 boards if relevant.
    """,
    
    "optimization": """
Search for AI optimization techniques on STM32.
Query: {search_query}

Provide:
1. Available compression techniques (quantization, pruning, distillation)
2. Quantization levels (INT8, INT16, FP16, etc.) and impact
3. Accuracy vs model size trade-offs
4. Optimization tools (STEdgeAI, TensorFlow Lite, TVM, etc.)
5. Performance benchmarks (latency, throughput, memory)
6. Best practices and optimization checklists
7. Links to official resources and tutorials

Include concrete metrics (e.g. "from 5MB to 200KB with INT8 quantization").
    """,
    
    "documentation": """
Search for STM32 documentation and guides.
Query: {search_query}

Provide:
1. Links to official STMicroelectronics documentation
2. Step-by-step tutorials for your topic
3. Code examples on GitHub
4. Common FAQs and solved problems
5. Community forums and resources (StackOverflow, ST Community, etc.)
6. Video tutorials (YouTube, Udemy, Coursera, etc.)
7. Recommended books if relevant

Prioritize official and recent sources.
    """
}


# ============================================================================
# WORKFLOW 4: WEB RESEARCH (OPTIMIZED WITH DYNAMIC PROMPT)
# ============================================================================

def classify_search(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """Classifies the type of search requested by the user."""
    
    logger.info(f"🔍 Search classification: {state.message}")
    
    try:
        cfg = Configuration.from_runnable_config(config)
        
        from src.assistant.utils import get_llm
        llm_classifier = get_llm(
            config=config,
            structured_schema=SearchClassification,
        )
        
        result = llm_classifier.invoke([
            SystemMessage(content=search_classification_instructions),
            HumanMessage(content=f"Request: {state.message}")
        ])
        
        state.search_type = result.search_type
        state.search_query = result.search_query
        
        logger.info(f"✓ Search type: {state.search_type}")
        logger.info(f"  Query: {state.search_query}")
        logger.info(f"  Reasoning: {result.reasoning}")
        
        if state.search_type == "none":
            logger.warning("No search type recognized")
            state.route = "unknown"
        
    except Exception as e:
        logger.error(f"❌ Search classification error: {str(e)}")
        logger.exception(e)
        state.route = "unknown"
    
    return state


def search_type_decision(state: MasterState) -> Literal["execute_web_search", "clarify"]:
    """Simplified routing: if the search type is valid, go to execute_web_search."""
    if state.search_type in ["ai_model", "board_selection", "optimization", "documentation"]:
        logger.info(f"→ Executing search: {state.search_type}")
        return "execute_web_search"
    else:
        logger.warning(f"⚠️  Invalid search type: {state.search_type}")
        return "clarify"


def execute_web_search(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """
    Single search node that adapts the prompt dynamically.
    Uses duckduckgo_search directly (no agno Agent) and then get_llm() to synthesize.
    This approach is compatible with both Triton and Ollama without requiring
    model tool-calling/function-calling.
    """
    
    logger.info(f"🔍 Web search: type={state.search_type}, query={state.search_query}")
    
    try:
        from duckduckgo_search import DDGS
        from langchain_core.messages import SystemMessage, HumanMessage
        from src.assistant.utils import get_llm
        
        base_prompt = SEARCH_PROMPTS.get(state.search_type, SEARCH_PROMPTS["documentation"])
        search_prompt = base_prompt.format(search_query=state.search_query)
        
        logger.info(f"📋 Prompt used for {state.search_type} (length: {len(search_prompt)} char)")
        
        # -----------------------------------------------------------------
        # STEP 1: Direct DuckDuckGo search (no LLM, no tool calling)
        # We use the duckduckgo_search library directly. This way we
        # avoid the dependency on agno Agent + model function calling,
        # which is incompatible with the Triton backend.
        # -----------------------------------------------------------------
        logger.info(f"🌐 Executing direct DuckDuckGo search for: {state.search_query}")
        raw_results = []
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(state.search_query, max_results=6))
                for r in results:
                    snippet = f"**{r.get('title', '')}**\n{r.get('body', '')}\nSource: {r.get('href', '')}"
                    raw_results.append(snippet)
            logger.info(f"   ✓ DuckDuckGo: {len(raw_results)} results found")
        except Exception as search_err:
            logger.warning(f"⚠️ DuckDuckGo search error: {search_err}")
            raw_results = [f"No results for: {state.search_query}"]
        
        raw_text = "\n\n---\n\n".join(raw_results)
        
        # -----------------------------------------------------------------
        # STEP 2: Synthesis with LLM (routed via get_llm → Triton or Ollama)
        # -----------------------------------------------------------------
        logger.info(f"🧠 Synthesizing results with LLM...")
        llm = get_llm(config=config)
        
        synthesis_messages = [
            SystemMessage(content=(
                "You are an expert technical assistant in STM32 and embedded AI systems. "
                "You are provided with web research extracts. Synthesize them into a "
                "clear and structured answer in English, keeping links to the sources. "
                "Answer in a concise and technical way."
            )),
            HumanMessage(content=(
                f"Search query: {state.search_query}\n\n"
                f"Search type: {state.search_type}\n\n"
                f"Web extracts:\n{raw_text}\n\n"
                f"Original prompt: {search_prompt}"
            ))
        ]
        
        synthesis_response = llm.invoke(synthesis_messages)
        synthesized = synthesis_response.content if hasattr(synthesis_response, 'content') else str(synthesis_response)
        
        state.search_results = synthesized
        
        # Populate search_results_list for DeepEval (uses raw chunks, more faithful)
        state.search_results_list = [
            chunk.strip()
            for chunk in raw_results
            if len(chunk.strip()) > 20
        ]
        
        state.web_research_success = True
        logger.info(f"✓ Search completed ({len(state.search_results)} chars, {len(state.search_results_list)} chunks)")
        
    except Exception as e:
        logger.error(f"❌ Web search error: {str(e)}")
        logger.exception(e)
        state.search_results = f"Error in search: {str(e)}"
        state.search_results_list = []
        state.web_research_success = False
    
    return state


def summarize_search_results(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """
    Node that summarizes the web search results.
    Input: state.search_results (RAW text)
    Output: state.search_summary (Processed summary)
    """
    logger.info("📝 Summarizing search results...")
    
    if not state.web_research_success:
        logger.warning("Skipping summary due to failed search.")
        return state

    try:
        cfg = Configuration.from_runnable_config(config)
        
        # Prompt di validazione/riassunto (English for better performance)
        summary_prompt = f"""
        You are an expert technical writer for STM32 embedded systems.
        
        Objective: Summarize the web search results to answer the user's question perfectly.
        
        User Question: {state.search_query}
        
        Raw Web Results:
        {state.search_results[:10000]}  # Limit context to avoid overflow
        
        Instructions:
        1. Analyze the web results carefully.
        2. Synthesize a clear, direct, and technical answer in English.
        3. Cite sources (URLs) if present in the results.
        4. If results are irrelevant to the query, state it clearly.
        5. Use Bullet Points for readability.
        
        Answer:
        """
        
        from src.assistant.utils import get_llm
        llm = get_llm(
            config=config,
            temperature=0.2 # Low temp for factual summary
        )
        
        response = llm.invoke(summary_prompt)
        state.search_summary = response.content
        logger.info(f"✓ Summary generated ({len(state.search_summary)} chars)")
        
    except Exception as e:
        logger.error(f"❌ Summary error: {e}")
        state.search_summary = "Unable to generate the summary. See raw results."
        
    return state


def finalize_search(state: MasterState, config: RunnableConfig = None) -> MasterState:
    """Final node that presents search results (Summary + Eval)."""
    
    if state.web_research_success:
        print("\n" + "="*70)
        print(f"📊 SEARCH RESULTS: {state.search_type.upper()}")
        print("="*70)
        # Show summary, not raw results
        print(state.search_summary) 
        print("="*70 + "\n")
        logger.info("✓ Search successfully completed")
    else:
        print(f"\n❌ Error during search:\n{state.search_results}\n")
        logger.error(f"Search failed: {state.search_results}")
    
    # ===== DEEPEVAL EVALUATION =====
    if state.web_research_success:
        try:
            print("\n" + "="*70)
            print("⚖️  EVALUATING RESULT QUALITY (DeepEval)")
            print("="*70)
            
            # Context Separation for DeepEval:
            # Actual Output = The generated summary (search_summary)
            # Retrieval Context = LIST of chunks (search_results_list)
            
            eval_result = asyncio.run(asyncio.to_thread(
                _evaluate_summary_sync,
                state.search_query,      # Input (User Query)
                state.search_summary,    # Actual Output (LLM Summary)
                state.search_results_list if state.search_results_list else state.search_results # Fallback
            ))
            
            if eval_result["completed"]:
                metrics = eval_result["metrics"]
                print(f"✅ Faithfulness Score:       {metrics.get('faithfulness', 0):.2f}")
                print(f"✅ Answer Relevancy Score:    {metrics.get('answer_relevancy', 0):.2f}")
                print(f"✅ Contextual Relevancy:      {metrics.get('contextual_relevancy', 0):.2f}")
                print(f"✅ Hallucination Score:       {metrics.get('hallucination', 0):.2f}")
            else:
                print(f"⚠️ Evaluation skipped: {eval_result.get('error')}")
                
            print("="*70 + "\n")
            
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")

    return state

