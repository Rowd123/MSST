# MSST

Traduction Python des fonctions MATLAB `SST_Y` et `MSST_Y`, qui calculent les
transformées synchrosqueezées SST et multi-SST d'un signal.

## Installation

```bash
python -m pip install numpy
```

## Utilisation

```python
import numpy as np

from sst_y import sst_y

x = np.sin(2 * np.pi * np.arange(100) / 20)
Ts = sst_y(x, hlength=21)
print(Ts.shape)  # (50, 100)
```

Le signal peut être un tableau unidimensionnel ou un vecteur colonne de forme
`(n, 1)`. Si `hlength` est omis, sa valeur par défaut est `round(n / 5)`. La
fonction force ensuite la longueur de fenêtre à être impaire.

## Multi-SST

```python
from msst_y import msst_y

Ts, stft = msst_y(x, hlength=21, num=2)
```

`num` indique le nombre de réallocations successives. La fonction renvoie la
MSST normalisée et la STFT originale.
