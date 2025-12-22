# Evaluation Metrics: Automated STM32 AI Workflow vs. Manual Expert

To scientifically evaluate the benefits of this automated graph workflow compared to a professional using standard tools (STM32CubeMX + GitHub + IDE), we can define a set of quantitative and qualitative metrics.

## 1. Quantitative Metrics (Measurable Data)

These metrics provide hard data to prove efficiency and performance.

| Metric Categories | Specific Metric | Description | Why it matters |
| :--- | :--- | :--- | :--- |
| **Speed & Efficiency** | **Time-to-Hello-World** | Time from "I want a model for X" to "Code running on board". | Measures the raw speed of the initial setup. |
| | **Iteration Cycle Time** | Time to replace the model (e.g., change from ResNet to MobileNet) and redeploy. | **Critical:** This is where automation usually wins. Humans make mistakes when repeating steps; scripts don't. |
| | **Context Switch Count** | Number of different tools the user must interact with (Browser -> Terminal -> CubeMX -> IDE). | High context switching kills productivity and increases cognitive load. |
| **Quality & Reliability** | **Compilation Success Rate** | Percentage of generated projects that compile without manual fixing. | A "pro" might make a typo; the graph should be deterministic. |
| | **Configuration Error Rate** | Frequency of mismatched settings (e.g., wrong clock speed, insufficient heap size). | Common source of "hard to debug" embedded issues. |
| **Model Optimization** | **Inference Latency (ms)** | Time per inference on the device. | Did the automation find a better quantization/compression setting than the human default? |
| | **RAM/Flash Usage (KB)** | Memory footprint of the deployed solution. | Automated search can optimize for constraints better than manual trial-and-error. |

## 2. Qualitative Metrics (User Experience)

These metrics evaluate the "soft" benefits that are crucial for adoption.

*   **Barrier to Entry:** What is the minimum skill level required?
    *   *Manual:* Needs knowledge of C, Python, TensorFlow/PyTorch, CubeMX, Makefiles/CMake.
    *   *Automated:* Needs high-level intent (e.g., "Detect anomalies in audio").
*   **Reproducibility:** If you run the process today and next month, is the result identical? Humans drift; code is versioned.
*   **Documentation Quality:** Does the automation generate a report of *what* it did? (e.g., "Selected MobileNetV2 because X").

## 3. Experimental Protocol: "The Human vs. Machine Challenge"

To get "real data" as you requested, you can run a controlled experiment.

### Setup
*   **Task:** "Deploy an audio classification model on an STM32F4 to detect 3 specific keywords."
*   **Subject A (The Pro):** An embedded engineer with CubeMX and internet access.
*   **Subject B (The Graph):** Your automated workflow.

### Procedure
1.  **Round 1 (Greenfield):** Start from scratch. Measure total time to working binary.
2.  **Round 2 (Change Request):** "The model is too big, switch to a smaller architecture." Measure time to re-deploy.
3.  **Round 3 (Optimization):** "Maximize inference speed." Measure the final latency achieved after 1 hour of effort.

### Expected Results (Hypothesis)
*   **Round 1:** The Pro might be competitive if they have a template ready. The Graph will likely be faster but maybe less flexible initially.
*   **Round 2:** **The Graph should win by a landslide.** Re-running a pipeline is O(1) effort for a machine, but O(N) for a human who has to re-export, re-integrate, and re-compile.
*   **Round 3:** The Graph can sweep through 50 configurations (quantization, compression) in the time the Pro tests 2.

## 4. Academic & Industry Context

Research in MLOps (Machine Learning Operations) for embedded systems supports these metrics:

*   **Google's "Hidden Technical Debt in Machine Learning Systems" (Sculley et al.):** Highlights that the "ML Code" is a tiny fraction of the system; the "Plumbing" (what your graph does) is where the complexity and bugs lie. Automation reduces this debt.
*   **Continuous Integration/Deployment (CI/CD) Benefits:** Industry reports consistently show that automated pipelines reduce deployment failures by orders of magnitude compared to manual releases.

## 5. Unique Value Proposition: "Neural Network Modification"

The neural network modifications are already a big advantage.

*   **Manual:** A human modifying a Keras model to fit an STM32 (e.g., replacing unsupported layers, slicing the graph) is extremely error-prone and requires deep DL knowledge.
*   **Automated:** This graph can automatically apply "surgery" to the model (e.g., `Replace(LayerX, LayerY)`) based on hardware constraints.
*   **Metric:** "Success rate of deploying 'wild' models found on GitHub." (How many random HuggingFace models can the Pro deploy vs. the Graph?)
## 6. Time-Budget Breakdown: Expert vs. Automation

To quantify the savings, we break down a typical "Iteration Cycle" (e.g., testing a new model architecture on custom data).

**Scenario:** From "I have a folder of new WAV files" to "Running inference on STM32".

| Step | Expert (Manual Tools) | Automated Graph | Saving | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **1. Data Preprocessing** | 60 - 120 mins | 2 mins | **~98%** | Expert writes Python scripts to slice/augment audio and convert to Spectrograms. Graph has built-in `AudioToSpectrogram` node. |
| **2. Model Selection** | 30 mins (Research) | 1 min (Retrieval) | **~96%** | Expert reads papers/GitHub to find a compatible model. Graph retrieves "MobileNetV2-0.35" via RAG based on constraints. |
| **3. Training & Export** | 60 mins + Overhead | 60 mins + 0 Overhead | **Overhead** | Training time is similar (GPU bound), but Graph handles Keras->TFLite->C export automatically (0 click). |
| **4. Middleware Setup** | 45 mins | 0 mins | **100%** | Expert opens CubeMX, enables CRC, configures Clock, sets Heap/Stack. Graph generates `.ioc` automatically. |
| **5. Integration Code** | 90 mins | 0 mins | **100%** | Expert writes `process_data()`, buffer management, and calls `ai_run()`. Graph generates `model_adapter.c`. |
| **6. Debugging** | ~60 mins (avg) | ~5 mins | **~90%** | Manual integration often has HardFaults (stack overflow, type mismatch). Graph code is pre-validated. |
| **TOTAL (1 Iteration)** | **~5 - 6 Hours** | **~1 Hour (Training dominated)** | **~5x Speedup** | **The human bottleneck is removed.** |

*Note: The "Training" time is excluded from savings as it depends on GPU speed, but the "Human Active Time" drops from hours to minutes.*
