# Openclassromm projet 7 : Implémenter un modéle de scoring
## Description

## Installation

Créer un environnement virtuel avec une version python 3.8+

> python -m venv .venv

Activer l'envornnement virtuel

>(windows) ./.venv/Scripts/activate

>(ubuntu) source .venv/bin/activate 


Installer les dépendances

> python -m pip install -r requirements.txt
 

### Construction des fichiers
Github limite la taille des fichiers archivable à 100 Mb, ce qui n'est pas suffisant pour hebergé les jeu de données. 
Ainsi, il est necessaire de telecharger les fichiers de données brutes et rejouer les scripts de transformation pour regénerer les models.

1- telecharger les fichier de données [ici](https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Impl%C3%A9menter+un+mod%C3%A8le+de+scoring/Projet+Mise+en+prod+-+home-credit-default-risk.zip
) et les déposer dans le répertoire "datasets/raw"

2- executer le scripts de construction du projet
> python build_project.py

Ce script prend en charge le preprocessing et la sauvegarde des fichiers de données necessaire a la modélisation, il permet également de générer une pipeline de preprocessing et un modéle LGBM préentrainé selon les hyperparamétres identifiés comme optimaux.


### Déploiement de l'API

> uvicorn src.api.main:app --host 0.0.0.0 --port 8000


### Déploiement du dashboard

> streamlit run src/dashboard/dash.py


### Configuration
le fichier settings.py contient les variables prinicpales du projet. Il est necessaire de modifier l'url de l'API pour l'adapter a l'environement en cours

