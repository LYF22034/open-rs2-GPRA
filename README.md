# **GPRA**: Enhancing Small Model **Reasoning** Under Resource Constraints via LoRA Policy Optimization with Implicit Process Rewards

**Yifan Li**

Nanyang Technological University

📧 LIYI0115@e.ntu.edu.sg

---

📄 [Paper (PDF)](./report.pdf) &nbsp;|&nbsp; 🤗 [Model Weights](https://huggingface.co/whalexdfsa/open-rs2-GPRA) &nbsp;|&nbsp; 📊 [Training Logs (WandB)](https://wandb.ai/)

---

## Abstract

Reinforcement learning with verifiable rewards (RLVR) has emerged as a powerful paradigm for enhancing the reasoning capabilities of large language models (LLMs). However, most existing approaches demand substantial computational resources, limiting accessibility for researchers with constrained GPU budgets. In this work, we investigate a resource-efficient approach to improving mathematical reasoning in small language models by combining two recent innovations: (1) LoRA-based parameter-efficient RL training, as demonstrated by the Tina framework, and (2) online implicit process reward models (PRMs) from the PRIME framework. We term our combined method **GPRA** (**G**RPO + **P**RIME + Lo**RA**), which applies LoRA for the policy model update while maintaining a full-parameter implicit PRM that is updated online. Starting from `DeepSeek-R1-Distill-Qwen-1.5B`, we first reproduce a LoRA+GRPO baseline on 2×A100-80GB GPUs, then augment it with GPRA. Our method achieves a best average accuracy of **52.19%** across four mathematical reasoning benchmarks (AIME 2024, AIME 2025, AMC 2023, MATH-500), representing a **4.73 percentage point improvement** over the LoRA+GRPO baseline (47.46%). Notably, GPRA reaches higher peak performance in fewer training steps, demonstrating improved sample efficiency consistent with the findings of PRIME. Our results validate that dense process rewards can meaningfully benefit parameter-efficient RL training on small models under tight resource constraints.

## Key Results

### LoRA+GRPO Baseline (2×A100-80GB, DP)

| Step | AIME24 | AIME25 | AMC23 | MATH500 | Average |
|:----:|:------:|:------:|:-----:|:-------:|:-------:|
| 250 | 26.67 | 16.67 | **62.50** | **78.00** | 45.96 |
| **300** | **36.67** | 20.00 | 60.00 | 73.20 | **47.46** |
| 350 | 26.67 | 20.00 | 50.00 | 76.00 | 43.17 |
| 400 | 20.00 | **23.33** | 57.50 | 75.80 | 44.16 |
| 450 | 33.33 | 13.33 | 55.00 | 73.20 | 43.72 |
| 500 | 26.67 | 20.00 | 60.00 | 75.40 | 45.52 |
| 550 | 20.00 | 20.00 | **62.50** | 75.00 | 44.38 |

### GPRA (Our Method, 2×A100-80GB, DP)

| Step | AIME24 | AIME25 | AMC23 | MATH500 | Average |
|:----:|:------:|:------:|:-----:|:-------:|:-------:|
| 150 | 33.33 | 23.33 | 67.50 | 75.40 | 49.89 |
| 200 | **40.00** | 20.00 | 52.50 | 75.60 | 47.02 |
| 250 | 36.67 | 23.33 | 62.50 | 78.00 | 51.38 |
| **300** | 26.67 | **30.00** | **72.50** | **79.60** | **52.19** |
| 350 | 26.67 | 10.00 | 60.00 | 76.80 | 43.37 |
| 400 | 30.00 | 23.33 | 67.50 | 75.80 | 49.16 |
| 450 | 26.67 | 26.67 | 62.50 | 76.40 | 48.06 |

### Summary Comparison

| Model | AIME24 | AIME25 | AMC23 | MATH500 | Avg. |
|:------|:------:|:------:|:-----:|:-------:|:----:|
| DeepSeek-R1-Distill-Qwen-1.5B | 23.33 | 16.67 | 62.50 | 79.60 | 46.19 |
| Eurus-2-7B-Prime | 20.00 | 16.67 | 50.60 | 78.20 | 41.37 |
| Llama-3.1-70B-Inst. | 20.00 | 23.33 | 52.00 | 65.00 | 40.02 |
| LoRA+GRPO (ours) | 36.67 | 20.00 | 60.00 | 73.20 | 47.46 |
| **GPRA (ours)** | 26.67 | **30.00** | **72.50** | **79.60** | **52.19** |
| **Improvement** | -- | +10.00 | +12.50 | +6.40 | **+4.73** |

## Method Overview

**GPRA** = **G**RPO + **P**RIME + Lo**RA**

- **Policy Model**: Updated with **LoRA** (parameter-efficient) during GRPO training
- **Implicit PRM**: Updated with **full parameters** online, providing dense token-level process rewards
- **Reference Model**: Frozen base model for computing implicit rewards

The core idea: LoRA handles policy format adaptation efficiently, while a full-parameter PRM provides expressive dense reward signals for better credit assignment.

### Architecture

```
GPU 1 (vLLM Inference)          GPU 2 (Training)
┌─────────────────────┐    ┌──────────────────────────┐
│                     │    │  Policy Model (LoRA)     │
│  Rollout Generation │───▶│  + Full-Param PRM        │
│  (K responses/prompt)│    │  + Reference Model       │
│                     │    │                          │
└─────────────────────┘    │  1. PRM forward & update │
                           │  2. Compute advantages   │
                           │  3. Policy LoRA update   │
                           └──────────────────────────┘
```

### Key Design Choices

1. **Asymmetric updates**: LoRA for policy (format adaptation) + full-parameter for PRM (reward expressiveness)
2. **Online PRM update**: PRM learns from policy rollouts using only outcome labels, avoiding reward hacking
3. **Online prompt filtering**: Keeps prompts with accuracy in (0.2, 0.8) to balance difficulty and PRM training distribution

## Training Configuration

| Hyperparameter | Value |
|:---------------|:------|
| Base Model | DeepSeek-R1-Distill-Qwen-1.5B |
| LoRA Rank | 64 |
| LoRA Alpha | 128 |
| LoRA Dropout | 0.05 |
| LoRA Targets | query, key, value, dense |
| Policy LR | 1e-6 |
| PRM LR | 1e-6 |
| PRM β | 0.05 |
| Precision | BF16-mixed |
| Generations per prompt (K) | 4 |
| Max Completion Length | 3584 |
| Hardware | 2× A100-80GB (RunPod) |

## Quick Start

### Installation

```bash
git clone https://github.com/LYF22034/open-rs2-GPRA.git
cd open-rs2-GPRA
pip install -r requirements.txt
```

### Training

```bash
# Stage 1: LoRA+GRPO baseline
# (see training scripts in the repo)

# Stage 2: GPRA (LoRA+GRPO+PRIME)
# (see training scripts in the repo)
```

### Model Weights

Our trained model checkpoints are available on Hugging Face:

🤗 [whalexdfsa/open-rs2-GPRA](https://huggingface.co/whalexdfsa/open-rs2-GPRA)

## Key Findings

1. **+4.73% improvement**: GPRA achieves 52.19% vs. 47.46% for the LoRA+GRPO baseline across four math benchmarks.

2. **~2× sample efficiency**: GPRA reaches baseline-level performance (49.89%) at step 150, while baseline peaks at step 300 (47.46%).

3. **Lower KL, larger gradients**: GPRA maintains lower KL divergence from the reference model despite larger gradient norms, suggesting dense rewards provide more precisely directed optimization signals.

4. **Dense rewards + LoRA synergy**: The limited expressiveness of LoRA adapters benefits particularly from fine-grained process reward guidance.

## Citation

```bibtex
@misc{li2025gpra,
  title={GPRA: Enhancing Small Model Reasoning Under Resource Constraints via LoRA Policy Optimization with Implicit Process Rewards},
  author={Yifan Li and Lee Nara and Chengzhi Zhang and Fandi Meng and Ruiqi Zhu},
  year={2025},
  institution={Nanyang Technological University}
}
```

## Acknowledgements

This work builds upon the following open-source projects:

- [Tina: Tiny Reasoning Models via LoRA](https://arxiv.org/abs/2504.15777)
- [PRIME: Process Reinforcement through Implicit Rewards](https://arxiv.org/abs/2502.01456)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [Open-RS](https://arxiv.org/abs/2503.16219)
- [OpenR1](https://github.com/huggingface/open-r1)

## License

This project is released under the MIT License.
