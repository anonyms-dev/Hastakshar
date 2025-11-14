## Installation

Install the required dependencies using:

```bash
pip install -r requirements.txt

## Performance of EER (80:20 vs 1:4)

| Database            | No. of Signatures | Train (80:20) | EER (80:20)<br>Stokes+Sigma log+HHT | Train (1:4) | EER (1:4)<br>Stokes+Sigma log+HHT |
|--------------------|-------------------|---------------|--------------------------------------|-------------|-----------------------------------|
| SigComp11 Chinese  | 130               | 104           | 1.2615                               | 1 / 1       | 1.633                             |
| SigComp11 Dutch    | 120               | 95            | 0.2518                               | 1 / 1       | 1.9089                            |
| DeepSign DB        | 1526              | 1220          | 1.5986                               | 1 / 4       | 1.6035                            |
| Hindi              | 2500              | 200           | 1.4827                               | 3 / 2       | 1.8817                            |
