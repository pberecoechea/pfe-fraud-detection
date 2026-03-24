# PFE : Système de détection de fraude bancaire en temps réél 💳

*Par Pablo BERECOECHEA et Rubens TUEUX*

![Photo montagne dossier data](https://media.istockphoto.com/id/2214860342/fr/vectoriel/pirate-informatique-volant-carte-de-paiement-illustration-vectorielle.jpg?s=2048x2048&w=is&k=20&c=jyrA1mrry68HSD8A72W6My0Wp9vYW4kRn535kReUgKU=)

> [!NOTE]
> Sujet de notre projet de fin d'études pour notre dernière année du cycle ingénieur à Cy Tech PAU.

## SOMMAIRE

1. [**Objectif du projet**](#objectif-du-projet)
2. [**Organisation du dossier**](#organisation-du-dossier)
3. [**Prérequis**](#prérequis)
4. [**Installation**](#installation)
    1. [Git](#git)
    2. [Préparation de l'environnement virtuel](#préparation-de-lenvironnement-virtuel)
        1. [*Sur un environnement Linux*](#sur-un-environnement-linux)
        2. [*Sur un environnement Windows*](#sur-un-environnement-windows)
    3. [Installation des dépendences](#installation-des-dépendences)
        1. [*Sur un environnement Linux*](#sur-un-environnement-linux-1)
        2. [*Sur un environnement Windows*](#sur-un-environnement-windows-1)
5. [**Exécution**](#exécution)
    1. [Exécution dans un environnement virtuel](#exécution-dans-un-environnement-virtuel)
        1. [*Sur un environnement Linux*](#sur-un-environnement-linux-2)
        2. [*Sur un environnement Windows*](#sur-un-environnement-windows-2)
6. [**Résultats**](#résultats)


## Objectif du projet

L'objectif du projet est de simuler un système de transactions bancaire et de pouvoir repérer y les fraudes.

## Organisation du dossier

```
pfe-fraud-detection/ 
├── data/
├── docker/
├── models/
├── notebooks/
├── src/
│   ├── api/
│   ├── dashboard/
│   ├── processing/
│   └── producer/
├── docker-compose.yml
├── README.md
└── requirements.txt
```

## Jeu de données

Le jeu données utilisé nous provient de [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Prérequis

* Avoir installé Python 3.10 (+).
* `git`. 
* `docker`.

## Installation

### Git 

Si le dossier du projet n'est pas déjà installé, vous pouvez le cloner depuis le dépôt [`🗃️ Github`](https://github.com/pberecoechea/pfe-fraud-detection.git). Pour cela, ouvrez un terminal, déplacez vous dans le dossier où vous souhaitez placer le projet et exécutez :

```bash
git clone https://github.com/pberecoechea/pfe-fraud-detection.git
```

### Préparation des outils

```bash
docker compose up
```

### Préparation de l'environnement virtuel

Pour commencer à préparer l'environnement virtuel, il faut se déplacer dans le dossier racine `📁 ti-compressed-sensing` et exécuter :

```bash
python -m venv venv
```

Une fois l'environnement virtuel installé, il faudra l'activer grâce à la commande suivante :

#### Sur un environnement Linux

```bash
source venv/bin/activate
```

#### Sur un environnement Windows

```shell
venv/Scripts/activate
```

Par la suite et pour l'exécution du projet, il faudra toujours activer l'environnement virtuel au préalable.

### Installation des dépendences

Le nouvel environnement virtuel ne possède pas encore les dépendences du projets. Il faudra donc les installer. Pour cela, exécutez les commandes suivantes :

#### Sur un environnement Linux
```bash
venv/bin/pip install -r requirements.txt
```

#### Sur un environnement Windows

```shell
venv/Scripts/pip install -r requirements.txt
```

## Exécution

Pour exécuter le programme, lancez un terminal dans le fichier racine du projet. Ouvrez un terminal, lancez l'environnement virtuel et exécutez :

### Exécution dans un environnement virtuel
 
#### Sur un environnement Linux

```bash
venv/bin/python main.py
```

#### Sur un environnement Windows

```shell
venv/Scripts/python main.py
```

## Résultats

Pour le premier rendu, le résultat sera l'image initiale recréée après être transformée en vecteur puis de nouveau en image. Ce résultat sera dans le dossier `📁 results`, dossier créé après exécution du programme.