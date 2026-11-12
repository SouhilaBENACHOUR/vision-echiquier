# Vision embarquée — Transcription d'une partie d'échecs par vidéo

Projet du module **Vision embarquée et IA** (Licence Informatique, La Rochelle Université, 2024/2025).

Ce projet utilise la vision par ordinateur (**OpenCV**) pour analyser la vidéo d'une partie d'échecs filmée du dessus, suivre les coups joués et les retranscrire en **notation algébrique**. Aucun réseau de neurones : tout repose sur du traitement d'image classique (contours, transformation de perspective, seuillage, flux optique).

## Principe

Le traitement se fait en plusieurs étapes, réparties sur les scripts :

1. **Détection et redressement de l'échiquier** (`detect_echq.py`)
   Détection des contours (Canny) pour repérer le plus grand quadrilatère de l'image, puis **transformation de perspective** pour obtenir une vue de dessus bien à plat de l'échiquier (800×800). La grille des cases est ensuite retrouvée par détection de lignes (Hough).

2. **Détection des mouvements de main** (`detect_mouv.py`)
   Détection de la peau dans l'espace colorimétrique **YCrCb** (plus robuste à l'éclairage) pour repérer les mains des joueurs. Le script distingue le joueur de gauche et celui de droite et signale qu'un coup est en train d'être joué.

3. **Détection des pièces et génération des coups** (`detect_piece.py`)
   À partir d'un échiquier de référence vide, chaque case est analysée pour savoir si une pièce l'occupe. Le **flux optique** (Farnebäck) détecte quand le plateau bouge ; en comparant l'état des cases avant/après, le script déduit la case de départ et d'arrivée et traduit le coup en notation algébrique (ex. `Nf3`, `exd5`).

4. **Transcription** (`transcription.py`)
   Assemble les coups détectés pour produire la transcription de la partie.

## Données de test

- `echiquier_vide_1.mp4` : plateau vide, sert de référence.
- `mat_du_berger_1.mp4`, `mat_du_lion.mp4` : parties filmées pour tester la détection.
- `echiquier.png` : image de l'échiquier redressé, produite par `detect_echq.py`.

## Installation

```bash
pip install opencv-python numpy
```

## Utilisation

Chaque script se lance seul et affiche le résultat dans des fenêtres OpenCV (`q` pour quitter, `p` pour mettre en pause) :

```bash
# Redresse l'échiquier et enregistre echiquier.png
python detect_echq.py

# Visualise la détection des mains
python detect_mouv.py

# Détecte les pièces et les coups joués
python detect_piece.py
```

Le nom de la vidéo analysée est défini en haut de chaque script : modifie-le pour tester une autre partie.

## Limites

- Les coordonnées de l'échiquier (points de calibration) sont réglées à la main pour les vidéos de test ; une nouvelle vidéo filmée sous un autre angle demande de réajuster ces valeurs.
- La position de départ des pièces est supposée connue (échiquier classique).

## Auteures

- Lina HADDADOU
- Souhila BENACHOUR

Détails complets dans `Rapport_VE_HADDADOU&BENACHOUR.pdf`.
