# MSST

Traduction Python des fonctions MATLAB `SST_Y` et `MSST_Y`, qui calculent les
transformées synchrosqueezées SST et multi-SST d'un signal.
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
fonction force ensuite la longueur de fenêtre à être impaire.

## Multi-SST

```python
from msst_y import msst_y

Ts, stft = msst_y(x, hlength=21, num=2)
```

`num` indique le nombre de réallocations successives. La fonction renvoie la
MSST normalisée et la STFT originale.

## Visualiser une mesure réelle avec Plotly

Installez les dépendances du graphique :

```bash
python -m pip install -e ".[plot]"
```

À partir d'un CSV dont la première ligne est un en-tête et dont le signal est
dans la deuxième colonne :

```bash
python -m examples.plot_instantaneous_frequency mesure.csv \
  --sample-rate 1000 --column 1 --skip-rows 1 \
  --hlength 31 --iterations 1 2 3 --show
```

Le graphique compare le signal mesuré, la crête fréquentielle de la STFT
(itération 0) et celles obtenues après chaque nombre d'itérations demandé. Une
version interactive autonome est aussi écrite dans `frequence_instantanee.html`.
fonction force ensuite la longueur de fenêtre à être impaire, comme le code
MATLAB d'origine. L'alias `SST_Y` est également disponible pour faciliter le
portage de code existant.
