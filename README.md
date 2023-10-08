# Compte rendu des TP d'apprentissage statistique (HAX907X)

Ce dêpot contiendra tous les comptes rendus effectués par Alexandre CAPEL dans le cadre du cours HAX907X d'apprentissage statistiques. 

Pour pouvoir bien compiler les codes des fichiers `.py`, il est nécessaire d'avoir installé au préalable les packages inscrits dans le fichier `requirements.txt` avec la commande suivante :

```bash
$ pip install -r requirements.txt
```

## Compilation du fichier Quarto (`.qmd`)

Pour les comptes rendus effectués dans un markdown sous Quarto (les fichiers `.qmd`), on peut compiler ces derniers pour produire un fichier `.pdf` contenant ces derniers. Pour ce faire, assurez-vous tout d'abord d'avoir bien installé Quarto (avec une commande `pip` par exemple) dans votre environnement, puis dans le terminal, placez vous dans le bon dossier et tapez la commande :

```bash
$ quarto render file.qmd
```

## Contenu des comptes rendus

### TP1 - k plus proches voisins

Le sujet de ce TP porte sur l'étude et l'implémentation d'un algorithme basique de classification : les $k$ plus proches voisins (ou $k$ nearest neighbors). Nous en profiterons pour construire nos premières simulations et pour prendre en main le package `scikit-learn`).

Le compte rendu est en format `.md`.

### TP2 - Arbres

Le sujet de ce TP porte sur l'étude des arbres de décisions (ou decision tree). Nous allons apprendre à générer nos premiers arbres à partir de données simulées et enregistrées (du package `scikit-learn`) et à faire de la sélection de modèle.

**Installation de graphviz requis**

Il est nécessaire d'avoir bien téléchargé correctement le package `graphviz` pour pouvoir enregistrer vos arbres. Une fois placé dans votre environnement python, exécutez la commande :

```bash
$  conda install python-graphviz
```

Le compte rendu est en format `.qmd`.


### TP3 - SVM

Dans ce TP, nous allons utiliser un classifieur SVM avec noyau pour de la classfication binaire (même si on peut l'étendre au cadre multi-classe). Nous avons pu construire nos premiers modèles et étudier l'effet du paramètre de régularisation dans le calcul du classifieur pour différents exemples, et voir les potentiels problèmes qu'entraînent la trop grand présence de variables de bruits.


Le compte rendu est en format `.md`.


### TP4 - ???