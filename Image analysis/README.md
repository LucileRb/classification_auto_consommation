## Analyser les images des produits

1) Démontrer la faisabilité de regrouper automatiquement des produits en utilisant leurs images
    -> approche non supervisée - clustering

Faire :
- un prétraitement des données images
- une extraction de features
- une réduction en 2 dimensions afin de projeter les produits sur un graphique 2D, sous la forme de points dont la couleur correspondra à la catégorie réelle
- Analyse du graphique afin d’en déduire ou pas, à l’aide des descriptions ou des images, la faisabilité de regrouper automatiquement des produits de même catégorie
- Réalisation d’une mesure pour confirmer ton analyse visuelle, en calculant la similarité entre les catégories réelles et les catégories issues d’une segmentation en clusters

Contraintes => mettre en œuvre :
- un algorithme de type SIFT / ORB / SURF
- un algorithme de type CNN Transfer Learning

2) Réaliser une classification supervisée à partir des images

- classification supervisée
- data augmentation