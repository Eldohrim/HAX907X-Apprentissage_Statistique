# TP1 : $k$-plus proches voisins

## Rappels de Classification
### Génération aritificielle de données

`rand_bi_gauss` : produit un échantillon de réponse qui vaut 1 pour la première variable gausienne et -1 pour la deuxième.

`rand_tri_gauss`: même chose que la première avec trois variables et les classes sont 1,2 et 3.

`rand_clown`: va renvoyer des matrices du plan qui appartiennent à deux groupes
$x_0$ est un vecteur normal centrée en 0
$x_1$ va renvoyer une parabole à partir de $x_0$ (avec une erreur gaussienne)
$x_2$ 
`rand_checkers` : 

La dernière colonne correspond à la classe de la realisation $i$

## La méthode

### Approche formelle

Rajouter une dimension 
```
import numpy
X = np.random.rand(100,2)
X[:,np.newaxis,:]
```