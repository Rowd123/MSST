# MSST

Traduction Python de la fonction MATLAB `SST_Y`, qui calcule la transformée
synchrosqueezée (SST) d'un signal.

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
fonction force ensuite la longueur de fenêtre à être impaire, comme le code
MATLAB d'origine. L'alias `SST_Y` est également disponible pour faciliter le
portage de code existant.
