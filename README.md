# [NeurIPS 2024] Official code for "Variational Distillation of Diffusion Policies into Mixture of Experts"
# (Under construction)
# TODO: Installation instructions
# TODO: Usage instructions
# TODO: Citation
# TODO: License
# TODO: Nice Figures


```bash
conda create --name <NAME>
conda activate <NAME>

conda install black isort pre-commit -c conda-forge

pre-commit install
pre-commit run
```
### Acknowledgements

This repo relies on the following existing codebases:

- The goal-conditioned variants of the environments are based on [play-to-policy](https://github.com/jeffacce/play-to-policy).
- The inital environments are adapted from [Relay Policy Learning](https://github.com/google-research/relay-policy-learning), [IBC](https://github.com/google-research/ibc) and [BET](https://github.com/notmahi/bet).
- The continuous time diffusion model is adapted from [k-diffusion](https://github.com/crowsonkb/k-diffusion) together with all sampler implementations. 
- the ```score_gpt``` class is adapted from [miniGPT](https://github.com/karpathy/minGPT).
- A few samplers are have been imported from [dpm-solver](https://github.com/LuChengTHU/dpm-solver)

---

## Citation

```bibtex
@article{zhou2024variational,
  title={Variational Distillation of Diffusion Policies into Mixture of Experts},
  author={Zhou, Hongyi and Blessing, Denis and Li, Ge and Celik, Onur and Jia, Xiaogang and Neumann, Gerhard and Lioutikov, Rudolf},
  journal={arXiv preprint arXiv:2406.12538},
  year={2024}
}
```

---