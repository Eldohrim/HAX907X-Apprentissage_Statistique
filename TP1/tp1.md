# TP1 : $k$-plus proches voisins

## Rappels de Classification
### Génération aritificielle de données

Dans le fichier `tp_knn_source.py`, nous avons à notre dispositions des fonctions qui vont nous permettre de générer des dataset différents. Par exemple : 

- `rand_tri_gauss`: produit un échantillon issue de trois vecteurs gaussien (de paramètres différents) appartenant à trois classes distinctes définies par les nombres 1,2 et 3. 

- `rand_clown`: produit un échantillon de deux variables différentes : `x_1` est un vecteur aléatoire formant une parabole à une erreur gaussienne près(groupe du sourire du clown=1) et `x_2` est la réalisation d'un vecteur gaussien (groupe du nez du clown=-1).

La dernière colonne correspond à la classe de la realisation $i$.

## La méthode

### Approche intuitive 

Faire la moyenne.


### Approche formelle

Rajouter une dimension 
```
import numpy
X = np.random.rand(100,2)
X[:,np.newaxis,:]
```