# Technical Deep Dive: Implementation Details

This document provides the technical "hidden details" behind the framework's development, suitable for explaining the architectural choices and implementation steps in the thesis.

---

## 0. Key Definitions (Glossary)
*   **VRAM (Video RAM)**: Specialized memory used by GPUs to store model weights, intermediate activations, and the KV (Key-Value) cache for transformers. Unlike system RAM, VRAM has significantly higher bandwidth, which is critical for the matrix multiplications performed during LLM inference.
*   **vLLM**: An open-source, high-throughput LLM serving engine. It uses a technique called **PagedAttention** (inspired by virtual memory paging in OS) to manage VRAM efficiently. Instead of pre-allocating large contiguous blocks of VRAM for the KV cache (which leads to fragmentation), vLLM allocates cache in small "pages," allowing it to handle more concurrent requests and larger context windows on the same hardware.
*   **Heavy SDKs**: Refer to large, high-level software development kits (like the full NVIDIA Triton Python Client or complex cloud-provider libraries) that often include many dependencies and high abstraction layers. In this project, we avoid them in favor of raw `urllib.request` calls to minimize the Docker image footprint, reduce startup latency, and ensure maximum transparency in the communication with the Triton V2 API.

---

## 1. VS Code Interface: The Chat Participant API
**Objective:** Provide a native, integrated development experience without forcing the user to leave the IDE.

*   **Technology Stack**: TypeScript, VS Code Extension API (`vscode-extension/src/extension.ts`).
*   **Mechanism**:
    *   **Chat Participant**: The extension registers a `ChatParticipant` using the `stm32-ai.assistant` ID. This allows it to appear in the GitHub Copilot chat panel.
    *   **Asynchronous Streaming**: Communicates with the Python backend via `FastAPI` using **NDJSON (Newline Delimited JSON)**. 
    *   **Event Transformation**: The backend (`server.py`) catches internal LangGraph events and transforms them into UI-friendly packets:
        *   `type: progress`: Updates the "Sto analizzando..." bullet points in real-time.
        *   `type: markdown`: Streams the actual text response or code blocks.
        *   `type: error`: Displays a red-highlighted error box.
    *   **UI Mapping**: A dictionary `NODE_LABELS` in the server maps technical node names (e.g., `generate_cubemx_script`) to descriptive labels (e.g., `📝 Generazione script CubeMX`).

---

## 2. NVIDIA Triton Architecture & VRAM Management
**Objective:** Decouple inference from orchestration and solve the "16GB VRAM constraint."

*   **The Problem**: Serving multiple specialized LLMs (Mistral 7B, DeepSeek-Coder 6.7B, GPT-OSS 20B) on a single 16GB GPU (NVIDIA A4000) is physically impossible if all are loaded simultaneously. A 7B model in 4-bit quantization takes ~5GB; adding context overhead quickly exceeds 16GB.
*   **The Solution: Explicit Model Control**:
    *   The Triton server is configured with `--model-control-mode=explicit`. This prevents Triton from auto-loading models into VRAM at boot.
    *   **The Triton Client (`src/assistant/triton_client.py`)**: Acts as a "VRAM Traffic Controller."
    *   **Model Swapping Logic**: Before every inference call, the client:
        1. Checks if the required model (e.g., `mistral`) is already `READY`.
        2. If not, it calls the Triton V2 API to `UNLOAD` any currently active LLMs to free VRAM.
        3. It then sends a `LOAD` request for the desired model.
        4. It polls the health endpoint until the CUDA context is fully established.
*   **vLLM Integration**: Inside the Triton container, the models are served via the **vLLM backend**. This allows the framework to leverage PagedAttention, ensuring that even with limited VRAM, we can maximize the context length (KV cache) for complex firmware code generation.

### 2.1 Implementation Details: The Swapping Algorithm
The logic is encapsulated in `ChatTriton._ensure_model_loaded()` within `src/assistant/triton_client.py`:

1.  **Mutual Exclusion (Lines 232-241)**: To avoid OOM, the client iterates over a list of heavy LLMs (`all_llms = ["mistral", "deepseek-r1", "gpt-oss-20b"]`). For any model that isn't the current target and is currently loaded, it sends an `UNLOAD` command via the repository API (`/v2/repository/models/{model}/unload`).
2.  **Graceful Synchronization (Lines 238-241)**: The client doesn't just send the command; it polls with `_wait_for_status` until the previous model is truly `UNAVAILABLE`, ensuring a 2-second cooldown to let the Python backend release the GPU memory handles.
3.  **The "READY" vs. "LIVE" Gap (Lines 292-323)**: Reachable via `_wait_for_endpoint_live()`.
    *   **The issue**: Triton might report a model as `READY` in the repository registry as soon as the binary is loaded. However, vLLM/Python backends often require additional seconds to capture the **CUDA Graph** or initialize the attention cache.
    *   **The solution**: The client performs a "Probe Inference" (a dummy "hi" request) to the native `/v2/models/{model}/infer` endpoint. It retries with a 2-second sleep until the first successful token is returned, bridging the gap between "registry ready" and "compute ready."
4.  **Raw HTTP Communication**: instead of using heavy SDKs, the client uses `urllib.request` directly. This minimizes the Docker image size and provides total control over timeout and retry headers for the Triton V2 API.

---

## 3. LangGraph Orchestration: The Master Orchestrator
**Objective:** Manage complex, non-linear workflows with state persistence.

*   **MasterState**: A centralized, Pydantic-based object (`src/assistant/state.py`) that follows the request through every node. It eliminates the "Context Window" bottleneck by only passing relevant parameters instead of the entire chat history.
*   **Dual-Layer Memory (Redis)**:
    1.  **Short-term (Session)**: Managed by `AsyncRedisSaver`. Every node execution is "checkpointed." If a container restarts, the assistant resumes exactly where it left off.
    2.  **Long-term (Profile)**: A separate Redis key-value store persists user hardware preferences (e.g., "Always use Nucleo-H743ZI").
*   **Workflow Branching**: Uses conditional edges. The `route_request` node uses a fast model to classify intent and jump into specialized subgraphs.

---

## 4. Hardware Integration & Manual Intervention
**Objective:** Control the "Bridge" between the AI and host tools (CubeMX).

*   **Manual Intervention (Interrupts)**:
    *   **Where in Code**: This is implemented using the `langgraph.types.interrupt` function. In `src/assistant/graph.py` (e.g., `clarify_request`) and `src/assistant/workflow1_firmware.py` (e.g., `collect_project_info`), the graph execution **actually pauses**.
    *   **The "Wait" Mechanism**: When `interrupt()` is called, the state is saved to Redis and the generator yields control. The VS Code extension receives a special `NDJSON` packet, prompts the user (e.g., "Check the Clock Tree in CubeMX"), and waits. The process only resumes when the user sends a new message/confirmation, which triggers an `update_state` in `server.py`.
*   **Peripheral Configuration Injection**:
    *   **Extraction**: The LLM uses the `ProjectInfoExtraction` schema (`src/assistant/workflow1_firmware.py`) to parse natural language requirements (e.g., "Activate PA5 and Timer 1").
    *   **Script Injection**: In the `generate_cubemx_script` node, the framework iterates through `state.peripheral_config` and appends the commands directly to the `.scr` file:
      ```python
      # In src/assistant/workflow1_firmware.py
      for cmd in state.peripheral_config:
          lines.append(cmd) # e.g., "set_pin PA5 GPIO_Output"
      ```
    *   **CubeMX Generation**: The final script is executed via `subprocess` calling the `STM32CubeMX` CLI with the `-q` (quiet) flag, which processes these commands as if a user had manually clicked them in the GUI.

---

## 5. AI Customization: Neural Network Surgery
**Objective:** Automate model optimization for resource-constrained hardware.

*   **Architecture Modification**: Uses Workflow 5 to replace incompatible layers (e.g., pruning unsupported activations).
*   **NNI Integration**: Automatically generates a trial script and uses the NNI `experiment.run()` API for headless hyperparameter search, selecting the best model based on the "Memory vs. Accuracy" pareto front.
