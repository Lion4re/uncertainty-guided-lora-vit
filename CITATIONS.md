# Citations

This repository builds on several existing methods. Please cite the original papers that correspond
to the components you use.

## Method Map

| Component in this repository | Primary citation |
| --- | --- |
| PACE, `LoRAmul_VPTadd`, consistency regularization | Ni et al., 2024 |
| LoRA adapters | Hu et al., 2022 |
| Visual prompt tuning / VPT-style additive prompt branch | Jia et al., 2022 |
| ViT-B/16 backbone | Dosovitskiy et al., 2021 |
| PAC-Bayes / PAC-tuning-style perturbation | Liu et al., 2023 |
| IVON-LoRA | Cong et al., 2024 |
| Bayesian-LoRA diagnostics | Lin et al., 2026 |
| Temperature scaling and ECE calibration metrics | Guo et al., 2017 |
| Inference-time probability ensembles | Lakshminarayanan et al., 2017 |
| Shift/OOD uncertainty evaluation framing | Ovadia et al., 2019 |

## BibTeX

```bibtex
@inproceedings{ni2024pace,
  title={PACE: Marrying Generalization in PArameter-Efficient Fine-Tuning with Consistency rEgularization},
  author={Ni, Yao and Zhang, Shan and Koniusz, Piotr},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}

@inproceedings{hu2022lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J. and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  booktitle={International Conference on Learning Representations},
  year={2022}
}

@inproceedings{jia2022visual,
  title={Visual Prompt Tuning},
  author={Jia, Menglin and Tang, Luming and Chen, Bor-Chun and Cardie, Claire and Belongie, Serge and Hariharan, Bharath and Lim, Ser-Nam},
  booktitle={European Conference on Computer Vision},
  year={2022}
}

@inproceedings{dosovitskiy2021image,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  booktitle={International Conference on Learning Representations},
  year={2021}
}

@inproceedings{liu2023pactuning,
  title={PAC-tuning: Fine-tuning Pretrained Language Models with PAC-driven Perturbed Gradient Descent},
  author={Liu, Guangliang and Xue, Zhiyu and Zhang, Xitong and Johnson, Kristen Marie and Wang, Rongrong},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  year={2023}
}

@inproceedings{cong2024variational,
  title={Variational Low-Rank Adaptation Using IVON},
  author={Cong, Bai and Daheim, Nico and Shen, Yuesong and Cremers, Daniel and Yokota, Rio and Khan, Mohammad Emtiyaz and M{\"o}llenhoff, Thomas},
  booktitle={NeurIPS 2024 Workshop on Fine-Tuning in Modern Machine Learning: Principles and Scalability},
  year={2024}
}

@misc{lin2026bayesianlora,
  title={Bayesian-LoRA: Probabilistic Low-Rank Adaptation of Large Language Models},
  author={Lin, Moule and Guan, Shuhao and Patane, Andrea and Gregg, David and Botterweck, Goetz},
  year={2026},
  eprint={2601.21003},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}

@inproceedings{guo2017calibration,
  title={On Calibration of Modern Neural Networks},
  author={Guo, Chuan and Pleiss, Geoff and Sun, Yu and Weinberger, Kilian Q.},
  booktitle={International Conference on Machine Learning},
  year={2017}
}

@inproceedings{lakshminarayanan2017simple,
  title={Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles},
  author={Lakshminarayanan, Balaji and Pritzel, Alexander and Blundell, Charles},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
}

@inproceedings{ovadia2019trust,
  title={Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift},
  author={Ovadia, Yaniv and Fertig, Emily and Ren, Jie and Nado, Zachary and Sculley, D. and Nowozin, Sebastian and Dillon, Joshua and Lakshminarayanan, Balaji and Snoek, Jasper},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
```

## Notes

- The PAC-Bayes branch in this repository is an adaptation inspired by PAC-tuning-style learned
  perturbation/noise training; it is not an official implementation of the EMNLP 2023 PAC-tuning
  repository.
- The IVON-LoRA branch uses IVON-style variational optimization for LoRA parameters in the vision
  setting; the cited IVON-LoRA paper studies language-model fine-tuning.
- The Bayesian-LoRA code is included as a diagnostic/experimental branch and should be cited only
  when that branch is used.