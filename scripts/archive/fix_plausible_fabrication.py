"""Fix plausible_fabrication entries: replace template-generated titles and repeated authors."""

import json
from pathlib import Path

# 60 unique, grammatically correct, plausible ML paper titles (30 per split)
DEV_TITLES = [
    "Structured Pruning via Differentiable Gating for Vision Transformers",
    "On the Convergence of Federated Averaging with Partial Worker Participation",
    "Contrastive Alignment of Multimodal Representations in Low-Resource Settings",
    "HyperPrompt: Prompt-Based Task-Conditioning of Transformers",
    "Efficient Posterior Approximation for Bayesian Neural Networks using Stein Variational Gradient Descent",
    "Causal Discovery from Heterogeneous Environments via Invariant Prediction",
    "Spectral Normalization for Robust Out-of-Distribution Detection",
    "Rethinking Self-Attention: Towards Interpretability in Neural Networks",
    "Distributionally Robust Optimization with Informative Priors for Fair Classification",
    "Temporal Knowledge Graph Completion via Recurrent Event Modeling",
    "Memory-Efficient Training of Large Language Models with Gradient Checkpointing",
    "Implicit Neural Representations for Continuous-Resolution Medical Image Segmentation",
    "Score-Based Diffusion Models for Conditional Molecule Generation",
    "Disentangled Representation Learning via Variational Information Bottleneck",
    "Gradient-Free Hyperparameter Optimization through Population-Based Training",
    "Multi-Scale Feature Aggregation for Dense Object Detection in Aerial Imagery",
    "Policy Gradient Methods for Risk-Sensitive Reinforcement Learning under CVaR Constraints",
    "Sample-Efficient Exploration in Reward-Free Markov Decision Processes",
    "Neural Implicit Surfaces for Multi-View Stereo Reconstruction without Masks",
    "Hierarchical Graph Transformers for Long-Range Dependency Modeling in Point Clouds",
    "Uncertainty-Aware Meta-Learning for Cross-Domain Few-Shot Classification",
    "Label-Efficient Semantic Segmentation via Self-Training with Confident Pseudo-Labels",
    "Provably Efficient Offline Reinforcement Learning with Linear Function Approximation",
    "Data-Dependent Stability Bounds for Stochastic Gradient Langevin Dynamics",
    "Energy-Based Models for Compositional Scene Generation and Editing",
    "Fast and Accurate Neural Architecture Search via Progressive Cell Pruning",
    "On the Role of Negative Sampling Strategies in Contrastive Self-Supervised Learning",
    "Differentiable Rendering for Inverse Graphics with Physically-Based Materials",
    "Robust Point Cloud Registration via Deep Feature Matching under Partial Overlap",
    "Neural Operator Learning for Parametric Partial Differential Equations on Irregular Meshes",
]

TEST_TITLES = [
    "Adaptive Curriculum Learning for Imbalanced Multi-Label Text Classification",
    "Toward Certified Robustness against Realistic Adversarial Perturbations in NLP",
    "GFlowNets for Combinatorial Optimization over Discrete Probabilistic Models",
    "Variational Inference with Normalizing Flows for Molecular Conformation Generation",
    "Vision-Language Pre-Training with Cross-Modal Contrastive Objectives",
    "Density-Aware Graph Neural Networks for Semi-Supervised Node Classification",
    "Scalable Bayesian Optimization with Thompson Sampling for High-Dimensional Spaces",
    "Unified Framework for Task-Incremental Continual Learning via Knowledge Consolidation",
    "Equivariant Message Passing for Molecular Property Prediction on 3D Graphs",
    "Self-Supervised Monocular Depth Estimation with Cross-View Consistency Constraints",
    "Calibrated Predictive Distributions via Multi-Output Gaussian Processes",
    "Adversarial Training with Curriculum Difficulty Scheduling for Robust Image Classification",
    "Instance-Conditional Knowledge Distillation for Compact Object Detection Models",
    "PAC-Bayesian Bounds for Domain Adaptation with Representation Alignment",
    "Multi-Agent Cooperative Exploration via Intrinsic Reward Sharing",
    "Sparse Attention Mechanisms for Efficient Long-Document Summarization",
    "Topology-Preserving Dimensionality Reduction via Persistent Homology Regularization",
    "Offline-to-Online Reinforcement Learning via Conservative Policy Iteration",
    "Prompt Tuning for Parameter-Efficient Transfer Learning in Vision Transformers",
    "Stochastic Differential Equations for Generative Modeling of Temporal Point Processes",
    "Test-Time Adaptation via Feature Distribution Alignment for Domain Shift Robustness",
    "Fair Representation Learning with Mutual Information Constraints under Group Imbalance",
    "Deep Equilibrium Models for Sequence-to-Sequence Prediction with Fixed-Point Iteration",
    "Information-Theoretic Regularization for Multi-Task Representation Learning",
    "Graph Contrastive Learning with Adaptive Augmentation Strategies for Node Classification",
    "Multi-Fidelity Bayesian Neural Networks for Scientific Surrogate Modeling",
    "Amortized Variational Inference for Large-Scale Inverse Problems in Imaging",
    "Compositional Zero-Shot Learning via Attribute-Guided Feature Synthesis",
    "Efficient Transformers for Audio Spectrogram Classification with Linear Attention",
    "Neural Radiance Field Compression via Learned Codebook Quantization",
]

# 60 unique author combinations (diverse, 3-5 authors each)
AUTHOR_POOL = [
    "Yichen Li and Maximilian Braun and Priya Raghavan",
    "Dongwei Zhao and Catherine Beaumont and Takeshi Moriyama",
    "Ananya Mukherjee and Jonas Lindqvist and Fatima El-Khoury",
    "Stefan Bauer and Mei-Ling Shyu and Alejandro Vidal",
    "Hye-Jin Park and Christoph Meinel and Oluwadara Adeyemi",
    "Gabriel Moreira and Xinyi Huang and Deepak Narayan and Ingrid Olsen",
    "Tomoko Watanabe and Bernhard Scholkopf and Youssef Mroueh",
    "Leonardo Ricci and Sunita Verma and Mikhail Petrov and Eva Lindgren",
    "Kexin Pei and Alessandro Ferraro and Nkechi Ogbonna",
    "Jieyu Zhang and Matthieu Cord and Raul Gomez-Brubaker",
    "Pavel Tokmakov and Shreya Ghosh and Francisco Vargas",
    "Daphne Cornelisse and Min-Hsuan Tsai and Ibrahim Alabdulmohsin",
    "Elena Vorontsova and Kenji Fukumizu and Beatriz Sanchez-Oro",
    "Aditya Grover and Nathalie Baracaldo and Hugo Larochelle",
    "Ziqian Zhong and Margherita Grossi and Abdul-Rashid Ibrahim and Sanna Wager",
    "Ryuichi Yamamoto and Federica Mandreoli and Ashwin Kalyan",
    "Tamara Broderick and Chenglong Wang and Anish Athalye",
    "Jessica Hamrick and Zhaohan Guo and Amina Adadi and Enrique Solano",
    "Ruiqi Gao and Fatemeh Nargesian and Marcel Nassar",
    "Anton Obukhov and Nayoung Lee and Santiago Gonzalez",
    "Wenda Chu and Katrin Erk and Vikash Sehwag and Mariya Toneva",
    "Robin Rombach and Neeraj Kumar and Esther Rolf",
    "Yuntao Bai and Joelle Pineau and Shivam Garg",
    "Hanlin Goh and Claudia Shi and Raghunathan Rengaswamy and Aurora Pons-Porrata",
    "Tiancheng Zhao and Marie-Francine Moens and Devi Parikh",
    "Georg Martius and Yanai Elazar and Salimeh Yasaei Sekeh",
    "Minjoon Seo and Rosanne Liu and Xiang Lorraine Li and Brenden Lake",
    "Ilia Sucholutsky and Marianna Apidianaki and Taesup Moon",
    "Fei Sha and Desmond Elliott and Volkan Cevher",
    "Lin Gui and Yuval Kluger and Saining Xie and Rachael Tatman",
    "Ekin Dogus Cubuk and Yaniv Taigman and Chelsea Finn",
    "Shuang Li and Florent Krzakala and Masashi Sugiyama",
    "Yuxin Wu and Hanie Sedghi and Sanjeev Arora",
    "Zhouhan Lin and Sara Hooker and Ludwig Schmidt and Timo Aila",
    "Yao Fu and Aapo Hyvarinen and Dhanya Sridhar",
    "Jiatao Gu and Ari Morcos and Emma Pierson and Kyunghyun Cho",
    "Xi Chen and Douwe Kiela and Tengyu Ma",
    "Po-Sen Huang and Dieuwke Hupkes and Xiaoyu Li and Oriol Vinyals",
    "Junxian He and Simon Kornblith and Yann Dauphin",
    "Vikas Sindhwani and Myle Ott and Yejin Choi and Pierre-Luc Bacon",
    "Nikolay Malkin and Ariel Procaccia and Tobias Pfaff",
    "Polina Kirichenko and Michael Zhang and Laura Hanu",
    "Arjun Sharma and Lily Weng and Omer Levy",
    "Wenhu Chen and Silvia Chiappa and David Ha and Danilo Rezende",
    "Jianlin Su and Naila Murray and Maximilian Nickel",
    "Adrien Ali Taiga and Hanna Wallach and Ilya Kostrikov",
    "Yifan Hou and Amal Rannen Triki and Raquel Urtasun and Stephan Mandt",
    "Yonatan Bisk and Samy Bengio and Jacob Andreas",
    "Lili Chen and Andre Esteva and Shengjia Zhao and Yingtao Tian",
    "Zhirong Wu and Andrew Wilson and Shiyu Chang",
    "Karen Hambardzumyan and Rishabh Agarwal and Shuangfei Zhai and Joshua Tenenbaum",
    "Jiaxin Shi and Animashree Anandkumar and Swarat Chaudhuri",
    "Tri Dao and Aravind Rajeswaran and Julian Togelius and Monica Lam",
    "Junier Oliva and Liwei Wang and Muhan Zhang",
    "Yelong Shen and Jeff Clune and Alexander Rush",
    "Yixin Wang and Hao Peng and John Schulman and Pieter Abbeel",
    "Shaoqing Ren and Luke Melas-Kyriazi and Been Kim",
    "Zachary Lipton and Mikhail Belkin and Jian Tang and Rose Yu",
    "Tatsunori Hashimoto and Eric Xing and Han Zhao",
    "Da Li and Pengtao Xie and Marin Bilokapic and Jimmy Ba",
]


def fix_file(filepath: Path, titles: list[str]) -> None:
    with open(filepath) as f:
        entries = [json.loads(line) for line in f]

    author_idx = 0
    title_idx = 0
    pf_count = 0

    for entry in entries:
        if entry.get("hallucination_type") != "plausible_fabrication":
            continue

        entry["fields"]["title"] = titles[title_idx]
        entry["fields"]["author"] = AUTHOR_POOL[author_idx]
        # Update explanation to reference the new title
        entry["explanation"] = "Completely fabricated paper with plausible metadata at real venue"

        title_idx += 1
        author_idx += 1
        pf_count += 1

    assert pf_count == 30, f"Expected 30 plausible_fabrication entries, got {pf_count}"
    assert title_idx == 30, f"Used {title_idx} titles, expected 30"

    with open(filepath, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Fixed {pf_count} plausible_fabrication entries in {filepath}")


def main() -> None:
    data_dir = Path("data/v1.0")
    fix_file(data_dir / "dev_public.jsonl", DEV_TITLES)
    fix_file(data_dir / "test_public.jsonl", TEST_TITLES)

    # Verify
    for name in ["dev_public.jsonl", "test_public.jsonl"]:
        path = data_dir / name
        with open(path) as f:
            entries = [json.loads(line) for line in f]
        pf = [e for e in entries if e.get("hallucination_type") == "plausible_fabrication"]
        titles = [e["fields"]["title"] for e in pf]
        authors = [e["fields"]["author"] for e in pf]
        assert len(set(titles)) == 30, f"Duplicate titles in {name}!"
        assert len(set(authors)) == 30, f"Duplicate authors in {name}!"
        print(f"  {name}: {len(entries)} total, {len(pf)} plausible_fabrication, all unique")


if __name__ == "__main__":
    main()
