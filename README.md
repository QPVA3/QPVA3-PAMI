# Parse, Align and Aggregate: Graph-driven Compositional Reasoning for Video Question Answering (QPVA $^3$)

<div align="center">

[📁 Project Structure](#-project-structure) • [🚀 Quick Start](#-quick-start) • [📊 Benchmark](#-benchmark) • [🎨 Visualization](#-visualization)

</div>

---

## 🌟 What is QPVA $^3$?

This paper introduces the QPVA³ framework, a novel approach to Video Question-Answering (VideoQA) that enhances the transparency and verifiability of AI reasoning. Current models often provide answers without clear explanations. Our framework addresses this by breaking down a complex question into a compositional graph of simpler, interconnected sub-questions. It then aligns these sub-questions with relevant video clips to find answers and logically aggregates them into a final, coherent response. To evaluate our method, we also introduce QPVA³Bench, a new benchmark dataset with detailed annotations, and novel metrics to measure reasoning consistency. Experiments show our framework significantly improves both accuracy and consistency, leading to a more transparent and trustworthy VideoQA system.

<div align="center">
  <img src="assets/img/QPVA3-framework.png" alt="QPVA framework" width="800">
</div>

## 📁 Project Structure

Our project is adapted from [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)
We modified the project and extended it to suit the infreence pipeline of QPVA3.

```
QPVA3/
├── assets/                   
│   └── image/                
├── lmm_eval/                 
│   ├── lmms_eval/
│   |   ├── models
│   |   |   ├── chat            # new chat api, recommended
│   |   |   |   ├── qpva3.py    # our QPVA3 model
|   |   |   |   └── ...
│   |   |   ├── simple          # the old simple api model, depreciated
│   |   |   |   ├── qpva3.py    # our QPVA3 model
|   |   |   |   └── ...         
│   |   |   ├── model_utils
│   |   |   |   ├── transformer_models   
│   |   |   |   |  ├── qpva3.py # the transformer vertion of QPVA3 model
|   |   |   |   |  └── ...
|   |   |   |   └── ...
│   |   ├── tasks
│   |   |   ├── agqa_decomp     # task for evaluating AGQA-Decomp
│   |   |   ├── qpva3_bench     # task for evaluating QPVA3Bench
|   |   |   └── ...
|   |   └── ...
│   └── ...                 
└── requirements.txt           # Python package dependencies 
```

## 🚀 Quick Start

### 1. clone the repo
``` bash
git clone https://github.com/QPVA3/QPVA3-PAMI.git
```
install lmms-eval following：https://github.com/EvolvingLMMs-Lab/lmms-eval

install the requirements via
``` bash
pip install -r requirements.txt
```

### 2. Train the models
Download the train data from [AGQA-Decomp](https://agqa-decomp.cs.washington.edu/)

Fintune the models with [llama-factory](https://github.com/hiyouga/LLaMA-Factory)

### 3. Evaluate the models with lmms-eval
Run your trained model with the following command to test the performance on AGQA-Decomp.
```bash
lmms_eval --model qpva3 --model_args.answerer_path Qwen/Qwen2.5-VL-7B-Instruct --model_args.max_pixels 12845056 --model_args.attn_implementation flash_attention_2 --model_args.interleave_visuals False --tasks agqa_decomp --batch_size 1 --cache_requests true --verbosity=DEBUG
```

## 📊 Benchmark

## 🎨 Visualization

### Full Reasoning Case
<div align="center">
  <img src="assets/img/visualization.png" alt="Sample Results Table 1" width="800">

*Qualitative example showcasing how QPVA $^3$ framework achieves successful successful in-depth analysis for VideoQA.*
</div>

* We first present an example that highlights compositional inconsistencies despite a correctly answered main question, demonstrating the capability of QPVA $^{3}$ to provide transparent analysis of MLLM outputs.
This case also serves to illustrate certain limitations of our approach.
Although VideoLLaMA3 accurately shows that the man spins the woman twice, it misinterprets the skating motion as dancing. 
While this perceptual error does not affect the recognition of the interaction between the man and the woman, it may yield unreliable responses for related sub-questions (*e.g.*, sub-question $A_5$), potentially impacting the user experience during conversational interactions.
Our framework exposes such potential failures in MLLMs; however, errors in spatial understanding remain challenging to correct, as the temporal video alignment mechanism offers limited support when nearly the entire video is deemed relevant (as seen in the alignment results for $I_1$ , $A_1$, $A_2$, and $A_5$).
Furthermore, since atomic questions form the basis for subsequent logical reasoning, errors at this initial stage are difficult for the reasoner to correct.
In the future, as the perceptual capabilities of MLLMs improve, our framework is expected to deliver more accurate and robust reasoning ability for these models.

### Conflict Reasoner Cases
<div align="center">
  <img src="assets/img/corr.png" alt="Sample Results Table 2" width="800">

  *Successful Case*
</div>

<div align="center">
  <img src="assets/img/err.png" alt="Sample Results Table 2" width="800">

  *Failure Case*
</div>

* We also present two cases showing how the reasoner handle conflict situations.
* In the successful case, the baseline model (VideoLLaMA3 + Aligner) incorrectly judges the cat's behavior as ``a bit anxious``. However, our reasoner aggregates evidence from atomic sub-questions, such as the cat's emotion being ``Friendly and Comfortable`` ($A_5$) and its action of sitting on the person. This creates a logical deduction that conflicts with the baseline's direct but incorrect answer. The reasoner successfully resolves this conflict by prioritizing the deduction derived from the detailed sub-question evidence, thus correcting the final answer to ``Friendly and Comfortable``. This case demonstrates the effectiveness of the first stage of conflict resolution, where aggregated evidence from sub-questions is used to challenge and override a flawed holistic judgment from the baseline model.
* Conversely, the failure case exposes a limitation in the second stage of conflict resolution. In this instance, the reasoner correctly aggregates the counts from the sub-questions (three black cats plus one orange cat) to arrive at the correct total of ``Four``. This correct aggregation creates a conflict with the baseline's incorrect direct answer, ``Three``. However, during the final answer selection step, the reasoner attempts to re-verify the count from the video but makes a perceptual error, failing to spot the orange cat. It consequently discards its own correct, logically derived answer (``Four``) and erroneously reverts to the incorrect answer (``Three``). This highlights a failure mode where the final self-correction mechanism itself is flawed, leading it to overturn a correct intermediate conclusion. This shows that while our two-stage process is powerful, its final output can still be compromised by errors in its own final verification step.
