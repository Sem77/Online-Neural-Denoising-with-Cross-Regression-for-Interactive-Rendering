# Online Neural Denoising with Cross-Regression for Interactive Rendering

Ce dépôt contient l'implémentation du papier de recherche *"Online Neural Denoising with Cross-Regression for Interactive Rendering"* (Choi et al., 2024). Ce projet a été réalisé dans le cadre du cours IG3DA du Master 2 IMA (Informatique) de Sorbonne Université.

## 🌍 Contexte et Motivation

Le rendu par lancer de rayons Monte Carlo (Monte Carlo ray tracing) permet de générer des images photoréalistes en simulant le comportement de la lumière. Cependant, pour des applications interactives ou en temps réel, le faible nombre d'échantillons par pixel génère un bruit visuel important. 

Les méthodes de débruitage classiques basées sur la régression (qui utilisent les G-buffers comme les normales ou l'albédo) parviennent à préserver les arêtes géométriques, mais ont tendance à flouter les détails non géométriques tels que les ombres complexes. 

Ce projet implémente une approche hybride innovante qui combine la **régression croisée (cross-regression)** pour générer des estimations pilotes, et un **réseau de neurones** pour un filtrage spatio-temporel adaptatif. Le modèle est entraîné **en ligne et de manière auto-supervisée** directement sur la séquence d'images en cours de rendu.

## 🧠 Méthodologie

L'architecture du projet repose sur plusieurs piliers techniques :
1. **Régression Croisée (Cross-Regression)** : Utilisation des G-buffers pour extraire des caractéristiques locales et préserver les détails haute fréquence sans perdre les ombres douces.
2. **Apprentissage Auto-Supervisé en Ligne** : Le rendu d'entrée est divisé en deux buffers distincts (A et B) avec des bruits indépendants. Une fonction de perte spatio-temporelle permet au réseau d'apprendre à extraire le signal propre sans nécessiter d'images de référence (Ground Truth).
3. **Filtrage Spatio-Temporel** : Une reprojection temporelle couplée à un filtrage spatial adaptatif pour assurer la stabilité temporelle entre les frames.
4. **Optimisation** : Traitement par tuiles (tiling) pour gérer les contraintes de VRAM du GPU lors du calcul des opérations complexes (comme la décomposition de Cholesky).

## 📂 Structure du Dépôt

Le dépôt contient les fichiers de code source en PyTorch ainsi que la documentation du projet :

* **`3687938.pdf`** : L'article de recherche original de Choi et al. (ACM SIGGRAPH 2024) définissant la méthode.
* **`Report_IG3DA.pdf`** : Le rapport technique de ce projet. Il détaille l'état de l'art, l'approche technique, les détails d'implémentation, ainsi qu'une analyse des résultats et des limites rencontrées.
* **`start.ipynb`** : Le point d'entrée principal (Jupyter Notebook). Il charge les séquences d'images, exécute la boucle d'entraînement auto-supervisée (calcul des pertes spatiales et temporelles), met à jour les poids du réseau et sauvegarde les images débruitées.
* **`ops.py`** : Script contenant les opérations mathématiques fondamentales en PyTorch, notamment l'estimation de la variance (`estimate_sigma`) et le calcul de la régression croisée (`compute_alpha_beta`).
* **`utils.py`** : Script utilitaire gérant le chargement des séquences d'images au format `.exr` (via `OpenEXR`), le calcul des poids pour le filtrage, et les transformations dans le domaine logarithmique pour stabiliser l'apprentissage.

## ⚙️ Installation et Prérequis

Ce projet est développé en Python et exploite l'accélération GPU avec **PyTorch**. Pour exécuter le code, vous aurez besoin des bibliothèques suivantes :

```bash
pip install torch torchvision numpy matplotlib OpenEXR Imath
```
