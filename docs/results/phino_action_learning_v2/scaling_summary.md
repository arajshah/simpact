| representation | split | target | n=0 | n=1 | n=2 | n=3 | n=4 |
|---|---|---|---|---|---|---|---|
| frozen DINOv2 | logo | mass | 1.00 | 0.88 | 0.85 | 0.80 | 0.76 |
| frozen DINOv2 | logo | friction | 0.99 | 0.95 | 0.85 | 0.88 | 0.86 |
| frozen DINOv2 | within | mass | 1.00 | 0.80 | 0.64 | 0.48 | 0.35 |
| frozen DINOv2 | within | friction | 1.00 | 0.92 | 0.77 | 0.72 | 0.69 |
| PHINO-ft clips (seed 0) | logo | mass | 1.00 | 0.87 | 0.74 | 0.67 | 0.61 |
| PHINO-ft clips (seed 0) | logo | friction | 0.97 | 0.85 | 0.73 | 0.66 | 0.60 |
| PHINO-ft clips (seed 0) | within | mass | 1.00 | 0.80 | 0.63 | 0.49 | 0.34 |
| PHINO-ft clips (seed 0) | within | friction | 1.00 | 0.86 | 0.72 | 0.66 | 0.62 |
| PHINO-ft clips (seed 1) | logo | mass | 1.00 | 0.89 | 0.73 | 0.67 | 0.60 |
| PHINO-ft clips (seed 1) | logo | friction | 0.97 | 0.84 | 0.72 | 0.65 | 0.61 |
| PHINO-ft clips (seed 1) | within | mass | 1.00 | 0.79 | 0.62 | 0.48 | 0.33 |
| PHINO-ft clips (seed 1) | within | friction | 1.00 | 0.86 | 0.72 | 0.65 | 0.61 |
| PHINO-ft static control | logo | mass | 1.00 | 0.89 | 0.88 | 0.87 | 0.86 |
| PHINO-ft static control | logo | friction | 0.99 | 1.00 | 0.83 | 0.84 | 0.83 |
| PHINO-ft static control | within | mass | 1.00 | 0.81 | 0.64 | 0.48 | 0.35 |
| PHINO-ft static control | within | friction | 1.00 | 0.91 | 0.79 | 0.73 | 0.70 |
| image-position tracker | logo | mass | – | 0.47 | 0.25 | 0.07 | 0.04 |
| image-position tracker | logo | friction | – | 0.58 | 0.29 | 0.13 | 0.12 |
| image-position tracker | within | mass | – | 0.54 | 0.22 | 0.09 | 0.08 |
| image-position tracker | within | friction | – | 0.72 | 0.32 | 0.22 | 0.19 |
| privileged state (physics summary) | logo | mass | – | 0.56 | 0.34 | 0.21 | 0.20 |
| privileged state (physics summary) | logo | friction | – | 0.63 | 0.39 | 0.23 | 0.18 |
| privileged state (physics summary) | within | mass | – | 0.59 | 0.31 | 0.19 | 0.15 |
| privileged state (physics summary) | within | friction | – | 0.51 | 0.26 | 0.20 | 0.18 |
| privileged state (observed frames) | logo | mass | – | 0.82 | 0.77 | 0.70 | 0.72 |
| privileged state (observed frames) | logo | friction | – | 0.76 | 0.87 | 0.90 | 0.96 |
| privileged state (observed frames) | within | mass | – | 0.50 | 0.37 | 0.35 | 0.34 |
| privileged state (observed frames) | within | friction | – | 0.65 | 0.42 | 0.37 | 0.33 |
| privileged state (full trajectory) | logo | mass | – | 0.85 | 0.95 | 0.93 | 0.91 |
| privileged state (full trajectory) | logo | friction | – | 0.88 | 0.95 | 0.97 | 0.98 |
| privileged state (full trajectory) | within | mass | – | 0.80 | 0.88 | 0.91 | 0.93 |
| privileged state (full trajectory) | within | friction | – | 0.87 | 0.87 | 0.91 | 1.02 |

| fine-tuned vs frozen | split | target | mean delta nMAE | subsets improved | fold cells improved |
|---|---|---|---|---|---|
| PHINO-ft clips (seed 0) | logo | mass | -0.094 | 14/15 | 61/75 |
| PHINO-ft clips (seed 0) | logo | friction | -0.154 | 14/15 | 59/75 |
| PHINO-ft clips (seed 0) | within | mass | -0.003 | 8/15 | 35/60 |
| PHINO-ft clips (seed 0) | within | friction | -0.056 | 13/15 | 43/60 |
| PHINO-ft clips (seed 1) | logo | mass | -0.093 | 14/15 | 61/75 |
| PHINO-ft clips (seed 1) | logo | friction | -0.163 | 15/15 | 59/75 |
| PHINO-ft clips (seed 1) | within | mass | -0.014 | 11/15 | 38/60 |
| PHINO-ft clips (seed 1) | within | friction | -0.058 | 13/15 | 43/60 |
| PHINO-ft static control | logo | mass | +0.035 | 4/15 | 30/75 |
| PHINO-ft static control | logo | friction | -0.006 | 13/15 | 34/75 |
| PHINO-ft static control | within | mass | +0.001 | 8/15 | 34/60 |
| PHINO-ft static control | within | friction | +0.011 | 2/15 | 16/60 |
