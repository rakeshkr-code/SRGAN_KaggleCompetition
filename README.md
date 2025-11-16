# SRGAN_KaggleCompetition

## Directory Structure
```
srgan_lowlight/
├── config/
│   ├── __init__.py
│   └── config.yaml
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── generator.py
│   │   ├── discriminator.py
│   │   └── srgan.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── transforms.py
│   ├── losses/
│   │   ├── __init__.py
│   │   └── perceptual_loss.py
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py
│       ├── visualization.py
│       └── logger.py
├── experiments/
│   ├── __init__.py
│   └── run_experiment.py
├── outputs/
│   ├── models/
│   ├── logs/
│   └── results/
├── main.py
├── sweep.py
└── requirements.txt
```

