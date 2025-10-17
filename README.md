# Parse, Align and Aggregate: Graph-driven Compositional Reasoning for Video Question Answering (QPVA $^3$)

<div align="center">

[📁 Project Structure](#-project-structure) • [🚀 Quick Start](#-quick-start) • [📊 QPVA $^3$ Bench](#-qpva3bench) • [🎨 Visualization](#-visualization)

</div>

---

## 🌟 What is QPVA $^3$?

This paper introduces the QPVA³ framework, a novel approach to Video Question-Answering (VideoQA) that enhances the transparency and verifiability of AI reasoning. Current models often provide answers without clear explanations. Our framework addresses this by breaking down a complex question into a compositional graph of simpler, interconnected sub-questions. It then aligns these sub-questions with relevant video clips to find answers and logically aggregates them into a final, coherent response. To evaluate our method, we also introduce QPVA³Bench, a new benchmark dataset with detailed annotations, and novel metrics to measure reasoning consistency. Experiments show our framework significantly improves both accuracy and consistency, leading to a more transparent and trustworthy VideoQA system.

<div align="center">
  <img src="assets/img/QPVA3-framework.png" alt="QPVA framework" width="800">
</div>

## 📁 Project Structure

```
QPVA3/
├── assets/                # Configuration templates & resources
│   └── image/             #    Example configs for models & datasets
├── framework/             #
│   ├── xxx/               #
│   └── xxx/               #
├── benchmark/             #
│   ├── dataset/           #
│   └── processor/         #
├── inference.py           # 
├── train.py               # 
└── requirements.txt       # Python package dependencies 
```

## 🚀 Quick Start

## 📊 QPVA $^3$ Bench

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