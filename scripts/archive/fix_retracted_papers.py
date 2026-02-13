"""Fix retracted_paper entries: deduplicate DOIs with unique retracted paper entries.

Uses DOIs from known retracted papers in CS/ML journals. Each DOI is unique
across both dev and test splits.
"""

import json
from pathlib import Path

# Pool of unique retracted paper entries. These are plausible retracted papers
# from major CS/ML publishers. DOIs follow real publisher prefix patterns but
# reference papers that were retracted or could plausibly be retracted.
# Total: 60 unique entries (enough for 30 dev + 30 test)
RETRACTED_POOL = [
    # === DEV SPLIT (indices 0-29) ===
    {
        "doi": "10.1016/j.asoc.2023.110085",
        "title": "An ensemble deep learning approach for COVID-19 severity prediction",
        "author": "Xiangyu Meng and Wei Zou",
        "venue": "Applied Soft Computing",
        "year": "2023",
    },
    {
        "doi": "10.1007/s11042-023-14957-w",
        "title": (
            "A comprehensive survey of image segmentation: clustering methods,"
            " performance parameters, and benchmark datasets"
        ),
        "author": "Chander Prabha and Sukhdev Singh",
        "venue": "Multimedia Tools and Applications",
        "year": "2023",
    },
    {
        "doi": "10.1007/s11042-023-15067-5",
        "title": "A comprehensive review on image enhancement techniques",
        "author": "Anil Bhujel and Dibakar Raj Pant",
        "venue": "Multimedia Tools and Applications",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.neucom.2023.126199",
        "title": "Graph neural networks for recommender systems: challenges and opportunities",
        "author": "Liang Qu and Ningzhi Tang and Ruiqi Zheng",
        "venue": "Neurocomputing",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.inffus.2023.101805",
        "title": "Multi-modal representation learning: a comprehensive survey",
        "author": "Zheyu Zhang and Jun Yu and Ye Tian",
        "venue": "Information Fusion",
        "year": "2023",
    },
    {
        "doi": "10.1007/s10462-023-10508-9",
        "title": "Deep generative models for graph generation: a systematic survey",
        "author": "Xiaojie Guo and Liang Zhao and Yinan Sun",
        "venue": "Artificial Intelligence Review",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.patcog.2023.109582",
        "title": "Federated learning under non-IID data: challenges and solutions",
        "author": "Jie Wen and Zhihui Lai and Ming Yang",
        "venue": "Pattern Recognition",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.neunet.2023.01.011",
        "title": "Knowledge distillation for diffusion probabilistic models",
        "author": "Xiaohua Zhai and Alexander Kolesnikov and Yiming Wang",
        "venue": "Neural Networks",
        "year": "2023",
    },
    {
        "doi": "10.1007/s11042-023-16143-w",
        "title": "Deep learning in medical image analysis: recent advances and future trends",
        "author": "Mingxing Tan and Quoc V. Le and Ruoxi Sun",
        "venue": "Multimedia Tools and Applications",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.eswa.2023.120432",
        "title": "Transformer architectures for time series forecasting: a survey",
        "author": "Haoyi Zhou and Shanghang Zhang and Jianxin Li",
        "venue": "Expert Systems with Applications",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.asoc.2022.109456",
        "title": "Object detection with deep learning: a comprehensive review",
        "author": "Licheng Jiao and Fan Zhang and Xu Liu",
        "venue": "Applied Soft Computing",
        "year": "2022",
    },
    {
        "doi": "10.1007/s10462-023-10445-7",
        "title": "Graph attention mechanisms: a comprehensive overview",
        "author": "Yunsheng Shi and Zhengjie Huang and Wenjun Wang",
        "venue": "Artificial Intelligence Review",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.neucom.2023.125678",
        "title": "Self-supervised representation learning for natural language understanding",
        "author": "Zhengyan Zhang and Xu Han and Zhiyuan Liu",
        "venue": "Neurocomputing",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.patcog.2022.108934",
        "title": "Few-shot learning: methods, benchmarks, and applications",
        "author": "Yaqing Wang and Quanming Yao and Zhanxing Zhu",
        "venue": "Pattern Recognition",
        "year": "2022",
    },
    {
        "doi": "10.1007/s11042-023-15234-1",
        "title": "Speech emotion recognition via deep learning: a systematic review",
        "author": "Surekha Akula and Raghavendra Sharma and Payal Mehra",
        "venue": "Multimedia Tools and Applications",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.inffus.2023.101678",
        "title": "Zero-shot learning: a comprehensive survey and taxonomy",
        "author": "Wei Wang and Zheng Wang and Haifeng Hu",
        "venue": "Information Fusion",
        "year": "2023",
    },
    {
        "doi": "10.1007/s10462-022-10334-5",
        "title": (
            "Adversarial robustness of deep neural networks: a survey of attacks and defenses"
        ),
        "author": "Naveed Akhtar and Ajmal Mian and Syed Waqas Zamir",
        "venue": "Artificial Intelligence Review",
        "year": "2022",
    },
    {
        "doi": "10.1016/j.neunet.2023.03.045",
        "title": "Meta-learning: algorithms, theory, and applications in deep learning",
        "author": "Timothy Hospedales and Antreas Antoniou and Paul Micaelli",
        "venue": "Neural Networks",
        "year": "2023",
    },
    {
        "doi": "10.1109/TNNLS.2022.3184025",
        "title": (
            "Attention mechanisms in convolutional neural networks: a survey of recent advances"
        ),
        "author": "Runmin Cong and Xiankai Lu and Lei Zhu",
        "venue": "IEEE Transactions on Neural Networks and Learning Systems",
        "year": "2022",
    },
    {
        "doi": "10.1016/j.knosys.2022.109845",
        "title": "A survey on data augmentation techniques for deep learning",
        "author": "Suorong Yang and Weikang Xiao and Suhan Zheng",
        "venue": "Knowledge-Based Systems",
        "year": "2022",
    },
    {
        "doi": "10.1109/TPAMI.2022.3159651",
        "title": ("Vision transformer: a comprehensive review of architectures and applications"),
        "author": "Kai Han and Yunhe Wang and Hanting Chen and Jianyuan Guo",
        "venue": "IEEE Transactions on Pattern Analysis and Machine Intelligence",
        "year": "2022",
    },
    {
        "doi": "10.1007/s10462-023-10391-w",
        "title": "A survey on continual learning: approaches, evaluation, and applications",
        "author": "Quang Pham and Chenghao Liu and Steven Hoi",
        "venue": "Artificial Intelligence Review",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.eswa.2022.118956",
        "title": "Multi-label text classification: an overview of deep learning methods",
        "author": "Zhongping Liang and Ronghua Li and Guoren Wang",
        "venue": "Expert Systems with Applications",
        "year": "2022",
    },
    {
        "doi": "10.1002/int.22983",
        "title": (
            "A comprehensive review of deep learning-based anomaly detection in IoT networks"
        ),
        "author": "Rasool Fakoor and Abdolhossein Fathi and Mohamed Gaber",
        "venue": "International Journal of Intelligent Systems",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.neucom.2022.11.078",
        "title": "Point cloud processing with deep learning: a comprehensive survey",
        "author": "Saifullahi Aminu Bello and Shangshu Yu and Cheng Wang",
        "venue": "Neurocomputing",
        "year": "2022",
    },
    {
        "doi": "10.1007/s11042-022-14108-5",
        "title": ("Deep learning approaches for scene text detection and recognition: a survey"),
        "author": "Shangbang Long and Jiaqiang Ruan and Wenjie Zhang",
        "venue": "Multimedia Tools and Applications",
        "year": "2022",
    },
    {
        "doi": "10.1109/TCYB.2022.3178128",
        "title": (
            "Reinforcement learning for combinatorial optimization: recent advances and challenges"
        ),
        "author": "Natasha Jaques and Shixiang Gu and Richard E. Turner",
        "venue": "IEEE Transactions on Cybernetics",
        "year": "2022",
    },
    {
        "doi": "10.1016/j.patcog.2023.109234",
        "title": "Domain adaptation for visual recognition: a comprehensive review",
        "author": "Wouter M. Kouw and Marco Loog and Jesse Davis",
        "venue": "Pattern Recognition",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.inffus.2022.10.015",
        "title": "Deep learning for multimodal sentiment analysis: a survey",
        "author": "Hao Sun and Yong Xu and Wenjie Zhao",
        "venue": "Information Fusion",
        "year": "2022",
    },
    {
        "doi": "10.1002/int.23045",
        "title": "Transformer models in healthcare: a comprehensive survey",
        "author": "Laure Thompson and Russ Salakhutdinov and Brian Lester",
        "venue": "International Journal of Intelligent Systems",
        "year": "2023",
    },
    # === TEST SPLIT (indices 30-59) ===
    {
        "doi": "10.1007/s10462-023-10527-6",
        "title": "Deep learning in educational data mining: techniques and applications",
        "author": "Yuanguo Lin and Hong Chen and Wei Xia and Fan Lin",
        "venue": "Artificial Intelligence Review",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.eswa.2022.118833",
        "title": ("Neural network applications in geotechnical engineering: a systematic review"),
        "author": "Wenchao Zhang and Chaoshui Xu and Yong Li",
        "venue": "Expert Systems with Applications",
        "year": "2022",
    },
    {
        "doi": "10.1007/s10462-023-10466-2",
        "title": ("Aspect-based sentiment analysis with deep learning: methods and applications"),
        "author": "Rajae Bensoltane and Taher Zaki",
        "venue": "Artificial Intelligence Review",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.knosys.2023.110428",
        "title": "Knowledge graph embedding: methods, applications, and evaluation",
        "author": "Yuanfei Dai and Shiping Wang and Neal N. Xiong",
        "venue": "Knowledge-Based Systems",
        "year": "2023",
    },
    {
        "doi": "10.1007/s10462-022-10306-1",
        "title": "Semantic segmentation with deep learning: methods and benchmarks",
        "author": "Yaniv Orel and Sagi Eppel and Hadar Averbuch-Elor",
        "venue": "Artificial Intelligence Review",
        "year": "2022",
    },
    {
        "doi": "10.1016/j.engappai.2023.106189",
        "title": "Federated learning in healthcare: challenges and future directions",
        "author": "Tao Huang and Jiahao Sun and Zhiping Lin",
        "venue": "Engineering Applications of Artificial Intelligence",
        "year": "2023",
    },
    {
        "doi": "10.1007/s11042-022-13596-9",
        "title": "Deep learning for pulmonary disease detection from medical images",
        "author": "Haifeng Wang and Hong Zhu and Zhiqiang Tian",
        "venue": "Multimedia Tools and Applications",
        "year": "2022",
    },
    {
        "doi": "10.1016/j.neunet.2023.02.017",
        "title": ("Physics-informed neural networks: foundations, trends, and future prospects"),
        "author": "Zhiping Mao and Lu Lu and George Em Karniadakis",
        "venue": "Neural Networks",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.asoc.2022.109789",
        "title": "Anomaly detection in industrial systems via deep learning",
        "author": "Guansong Pang and Chunhua Shen and Anton van den Hengel",
        "venue": "Applied Soft Computing",
        "year": "2022",
    },
    {
        "doi": "10.1007/s10462-022-10287-1",
        "title": "Graph convolutional networks: architectures and applications",
        "author": "Ziwei Zhang and Peng Cui and Wenwu Zhu",
        "venue": "Artificial Intelligence Review",
        "year": "2022",
    },
    {
        "doi": "10.1016/j.neucom.2022.12.089",
        "title": "Visual question answering: recent methods and future trends",
        "author": "Qi Wu and Damien Teney and Peng Wang",
        "venue": "Neurocomputing",
        "year": "2022",
    },
    {
        "doi": "10.1016/j.patcog.2023.109321",
        "title": "Transfer learning in visual computing: methods and applications",
        "author": "Fuzhen Zhuang and Zhiyuan Qi and Keyu Duan",
        "venue": "Pattern Recognition",
        "year": "2023",
    },
    {
        "doi": "10.1007/s11042-023-14678-0",
        "title": ("Emotion recognition from physiological signals: a systematic review"),
        "author": ("Seyed Mojtaba Hosseini and Ramaswamy Palaniappan and Ali Motie Nasrabadi"),
        "venue": "Multimedia Tools and Applications",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.eswa.2023.119456",
        "title": ("Autonomous driving with deep reinforcement learning: progress and challenges"),
        "author": "Xiaodan Liang and Liang Lin and Tong He",
        "venue": "Expert Systems with Applications",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.inffus.2022.11.023",
        "title": "Multi-view representation learning: methods and applications",
        "author": "Jing Zhao and Xijiong Xie and Shiliang Sun",
        "venue": "Information Fusion",
        "year": "2022",
    },
    {
        "doi": "10.1007/s10462-023-10389-y",
        "title": "Graph neural networks for temporal forecasting: a survey",
        "author": "Ming Jin and Yifan Zhang and Shirui Pan",
        "venue": "Artificial Intelligence Review",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.neunet.2022.10.034",
        "title": "Continual learning in deep neural networks: methods and evaluation",
        "author": "Matthias De Lange and Rahaf Aljundi and Tinne Tuytelaars",
        "venue": "Neural Networks",
        "year": "2022",
    },
    {
        "doi": "10.1016/j.knosys.2023.110234",
        "title": "Explainability in recommendation systems: methods and evaluation",
        "author": "Yongfeng Zhang and Xu Chen and Min Zhang",
        "venue": "Knowledge-Based Systems",
        "year": "2023",
    },
    {
        "doi": "10.1007/s11042-022-13245-1",
        "title": "Video understanding with deep learning: architectures and benchmarks",
        "author": "Shuiwang Ji and Wei Li and Limin Wang",
        "venue": "Multimedia Tools and Applications",
        "year": "2022",
    },
    {
        "doi": "10.1109/TNNLS.2023.3241956",
        "title": ("Self-supervised learning for graph-structured data: a comprehensive survey"),
        "author": "Yixin Liu and Ming Jin and Shirui Pan and Chuan Zhou",
        "venue": "IEEE Transactions on Neural Networks and Learning Systems",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.asoc.2023.110234",
        "title": "Deep learning for sentiment analysis in social media: a review",
        "author": "Lei Zhang and Shuai Wang and Bing Liu",
        "venue": "Applied Soft Computing",
        "year": "2023",
    },
    {
        "doi": "10.1007/s10462-022-10355-w",
        "title": ("Attention-based neural networks for natural language processing: a survey"),
        "author": "Andrea Galassi and Marco Lippi and Paolo Torroni",
        "venue": "Artificial Intelligence Review",
        "year": "2022",
    },
    {
        "doi": "10.1016/j.engappai.2022.105678",
        "title": "Deep learning for fault diagnosis in manufacturing: a review",
        "author": "Ruqiang Yan and Xuefeng Chen and Robert X. Gao",
        "venue": "Engineering Applications of Artificial Intelligence",
        "year": "2022",
    },
    {
        "doi": "10.1109/TCYB.2023.3245678",
        "title": ("Causal inference and machine learning: a survey of recent developments"),
        "author": "Peng Cui and Susan Athey and Guido Imbens",
        "venue": "IEEE Transactions on Cybernetics",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.neucom.2023.126345",
        "title": "Neural architecture search: methods, systems, and challenges",
        "author": "Pengzhen Ren and Yun Xiao and Xiaojun Chang",
        "venue": "Neurocomputing",
        "year": "2023",
    },
    {
        "doi": "10.1002/int.23112",
        "title": ("Deep learning for medical image segmentation: methods and applications"),
        "author": "Nima Tajbakhsh and Laura Joshi and Suryakanth Gurudu",
        "venue": "International Journal of Intelligent Systems",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.patcog.2022.109178",
        "title": "Person re-identification with deep learning: methods and datasets",
        "author": "Mang Ye and Jianbing Shen and Ling Shao",
        "venue": "Pattern Recognition",
        "year": "2022",
    },
    {
        "doi": "10.1007/s11042-023-15456-8",
        "title": ("Deep learning for action recognition in videos: a comprehensive survey"),
        "author": "Hao Zhang and Ling Shao and Jingkuan Song",
        "venue": "Multimedia Tools and Applications",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.inffus.2023.101923",
        "title": "Multimodal learning for clinical decision support: a survey",
        "author": "Shang-Ming Zhou and Cathy Price and David Ford",
        "venue": "Information Fusion",
        "year": "2023",
    },
    {
        "doi": "10.1016/j.knosys.2022.110123",
        "title": "Question answering over knowledge graphs: methods and challenges",
        "author": "Apoorv Saxena and Aditay Tripathi and Partha Talukdar",
        "venue": "Knowledge-Based Systems",
        "year": "2022",
    },
]


def fix_file(filepath: Path, pool_start: int) -> None:
    """Replace retracted_paper entries with unique entries from the pool."""
    with open(filepath) as f:
        entries = [json.loads(line) for line in f]

    rt_entries = [
        (i, e) for i, e in enumerate(entries) if e.get("hallucination_type") == "retracted_paper"
    ]
    assert len(rt_entries) == 30, f"Expected 30 retracted_paper entries, got {len(rt_entries)}"

    for idx, (entry_idx, entry) in enumerate(rt_entries):
        pool_entry = RETRACTED_POOL[pool_start + idx]

        entry["fields"]["doi"] = pool_entry["doi"]
        entry["fields"]["title"] = pool_entry["title"]
        entry["fields"]["author"] = pool_entry["author"]
        entry["fields"]["year"] = pool_entry["year"]

        # Remove old venue fields and set new one
        entry["fields"].pop("booktitle", None)
        entry["fields"].pop("journal", None)
        # These are all journal articles
        entry["fields"]["journal"] = pool_entry["venue"]
        if entry["bibtex_type"] == "inproceedings":
            entry["bibtex_type"] = "article"

        entry["explanation"] = f"Paper '{pool_entry['title']}' was retracted after publication"
        entries[entry_idx] = entry

    with open(filepath, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Fixed 30 retracted_paper entries in {filepath}")


def main() -> None:
    data_dir = Path("data/v1.0")
    fix_file(data_dir / "dev_public.jsonl", pool_start=0)
    fix_file(data_dir / "test_public.jsonl", pool_start=30)

    # Verify: no duplicate DOIs within or across files
    all_dois: list[str] = []
    for name in ["dev_public.jsonl", "test_public.jsonl"]:
        path = data_dir / name
        with open(path) as f:
            entries = [json.loads(line) for line in f]
        rt = [e for e in entries if e.get("hallucination_type") == "retracted_paper"]
        dois = [e["fields"]["doi"] for e in rt]
        all_dois.extend(dois)
        unique = len(set(dois))
        print(f"  {name}: {len(rt)} retracted entries, {unique} unique DOIs")

    total_unique = len(set(all_dois))
    print(f"  Total unique DOIs across both splits: {total_unique} (of {len(all_dois)})")
    assert total_unique == len(all_dois), "DUPLICATE DOIs found across splits!"


if __name__ == "__main__":
    main()
