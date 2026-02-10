# Compose by Focus — Official Code Release

Official code release for:

Compose by Focus: Scene Graph-based Atomic Skills
Paper: https://arxiv.org/abs/2509.16053

This repository implements an object-centric 3D diffusion policy pipeline that is effective for skill composition and long-horizon, multi-step robotic tasks.

---
## Quick Start


Train:
```
python train.py
```

Evaluate (test suites):
```
python eval_test_together.py
```

---
## Overview


Compose by Focus leverages scene graph-based object-centric representations to learn reusable atomic skills and compose them into complex behaviors. The policy is trained using a 3D diffusion model, enabling expressive and robust action generation under diverse scene configurations.

---
## Benchmark


The compositional benchmark used in this work is available at:
https://github.com/han20192019/skill-composition-benchmark

---
## Notes

More detailed usage instructions is coming soon.

---
## Citation

If you find this benchmark useful, please cite:

```
@article{qi25arxiv-compose,
  title={Compose by Focus: Scene Graph-based Atomic Skills},
  author={Qi, Han and Chen, Changhe and Yang, Heng},
  journal={arXiv preprint arXiv:2509.16053},
  year={2025},
  note={\linkToWeb{https://computationalrobotics.seas.harvard.edu/SkillComposition/}}
}
```

