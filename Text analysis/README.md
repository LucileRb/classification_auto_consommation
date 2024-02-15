## Analyse des descriptions textuelles des produits

But = Démontrer la faisabilité de regrouper automatiquement des produits en utilisant leurs descriptions
    -> approche non supervisée - clustering

Faire :
- 1) un prétraitement des données texte
- 2) une extraction de features
- 3) une réduction en 2 dimensions afin de projeter les produits sur un graphique 2D, sous la forme de points dont la couleur correspondra à la catégorie réelle
- 4) Analyse du graphique afin d’en déduire ou pas, à l’aide des descriptions ou des images, la faisabilité de regrouper automatiquement des produits de même catégorie
- 5) Réalisation d’une mesure pour confirmer ton analyse visuelle, en calculant la similarité entre les catégories réelles et les catégories issues d’une segmentation en clusters

Contraintes => mettre en œuvre :
- deux approches de type “bag-of-words”, comptage simple de mots et Tf-idf
- une approche de type word/sentence embedding classique avec Word2Vec (ou Glove ou FastText)
- une approche de type word/sentence embedding avec BERT
- une approche de type word/sentence embedding avec USE (Universal Sentence Encoder)

# A faire -> comparer la performance des différentes approches (avec/sans bow/tfidf, avec/sans word embedding et comparer les 3 word embedding differents)


Classification supervisée -> non demandée dans ce projet