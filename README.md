# classification_auto_consommation

CONTEXTE
l’entreprise "Place de marché” souhaite lancer une marketplace e-commerce
vendeurs proposent des articles à des acheteurs en postant une photo et une description.
l'attribution de la catégorie d'un article est effectuée manuellement par les vendeurs, et est donc peu fiable. De plus, le volume des articles est pour l’instant très petit.
nécessaire d'automatiser cette tâche

BUT
étudier la faisabilité d'un moteur de classification d'articles, basé sur une image et une description, pour l'automatisation de l'attribution de la catégorie de l'article, avec un niveau de précision suffisant

TO DO
1- Un ou des notebooks (ou des fichiers .py) contenant les fonctions permettant le prétraitement et la feature extraction des données textes et images ainsi que les résultats de l’étude de faisabilité (graphiques, mesure de similarité) 
2 - Un notebook de classification supervisée des images
3 - Un script Python (notebook ou fichier .py) de test de l’API et le fichier au format “csv” contenant les produits extraits
4 - Un support de présentation pour la soutenance, détaillant le travail réalisé (Powerpoint ou équivalent, sauvegardé en pdf, 30 slides maximum) : 
-> l’étude de faisabilité
-> la classification supervisée
-> test de l’API


# Notebooks
NB -> dossier data hors du dossier code pour ne pas l'importer dans github

1) Data_exploration_EDA.ipynb
Analyse exploratoire des données, gestion des données manquantes et des outliers, extraction de la target (catégorie des produits) et export des données "nettoyées" sous forme de fichier .csv

2) Data_exploration_text_analysis.ipynb
- prétraitement
- analyse de features
- réduction 2D
- calcul de similarité

Méthodes testées :
- BOW
- TF-IDF
- Word embedding : GloVe, Word2Vec, Fastext(principe évoqué mais méthode non testée)
- LDA
- BERT
- USE

3) Data_explo_prep_images.ipynb


4) Classification_supervisee_images.ipynb


5) Api_request.ipynb