# AutoEditor Video - Highlight Generator

Ce projet est une interface graphique (Tkinter) pour l'édition vidéo automatique, utilisant une **fusion** de `FFmpeg` (pour le volume) et `openai-whisper` (pour l'analyse de parole) afin de générer intelligemment des highlights et des clips Tiktok.

## Structure du Projet

* `main.py`: L'application principale. Contient l'interface graphique (GUI) `CutGUI`.
* `processor.py`: Le **cerveau** du projet. Contient la classe `VideoProcessor` qui gère les analyses et la création des vidéos.
* `requirements.txt`: Les dépendances Python (`openai-whisper`).
* `README.md`: Ce fichier.

## Fonctionnalités (Logique Hybride)

Ce n'est plus l'un *ou* l'autre, l'outil fait tout :

1.  **Double Analyse** : Le script lance deux analyses en parallèle :
    * **Analyse de Volume** : Découpe la vidéo en "bouts" et les note pour trouver les moments "intenses" (votre voix, action du jeu, cris, etc.).
    * **Analyse Sémantique (Whisper)** : Transcrit l'intégralité de la vidéo pour comprendre où commencent et finissent les **phrases**.

2.  **Cerveau "Intelligent"** : Le script **fusionne** les deux analyses. Il prend un moment "intense" (volume) et vérifie s'il y a de la parole dedans (Whisper). Si oui, il **étend le clip pour correspondre au début et à la fin de la phrase**, garantissant qu'aucune phrase n'est coupée au milieu.

3.  **Sortie 1: Highlight + Dérushage Pro** :
    * L'outil sélectionne les meilleurs "moments intelligents" selon votre profil (**Court, Moyen, Longue**).
    * Il sauvegarde chaque clip individuellement dans un dossier `_clips` (votre "dérushage pro").
    * Il assemble ensuite tous ces clips pour créer la vidéo **Highlight** finale (ex: `highlight.mp4`).

4.  **Sortie 2: Tiktoks (Optionnel)** :
    * Si la case est cochée, le script **compile** de nouveaux Tiktoks.
    * **Logique de Compilation** : Chaque Tiktok est une **compilation dynamique** de 1 minute maximum, assemblée en utilisant les **meilleurs "moments intelligents"** (parole+hype) disponibles, pour créer un "best-of" dynamique.
    * Il les extrait et les **redimensionne automatiquement** au format 9:16 (Tiktok) dans un dossier `_tiktoks`.

## Prérequis

* Python 3.x
* **FFmpeg**: Les exécutables `ffmpeg` et `ffprobe` doivent être installés et accessibles via le PATH.
* **`openai-whisper`**: La dépendance principale pour l'analyse.

## Installation

1.  Clonez ce dépôt (ou téléchargez les fichiers `main.py`, `processor.py`, `requirements.txt`).

2.  Installez les dépendances requises (principalement `openai-whisper`) :

    ```bash
    pip install -r requirements.txt
    ```

    *(Note: La première fois que vous l'utilisez, le script téléchargera le modèle de langue).*

## Utilisation

1.  Lancez l'interface graphique :

    ```bash
    python main.py
    ```

2.  Sélectionnez un fichier vidéo (un long stream, VOD, etc.).
3.  Choisissez un nom pour votre fichier Highlight (ex: `MaVideo_highlight.mp4`).
4.  Sélectionnez un profil (Court, Moyen, Longue).
5.  (Optionnel) Cochez la case "Générer aussi les clips Tiktok".
6.  Cliquez sur "Démarrer" pour lancer le traitement.

## Structure des Fiers de Sortie

Si votre fichier de sortie est `MaVideo_highlight.mp4`:

* `MaVideo_highlight.mp4` (La vidéo Highlight assemblée)
* `MaVideo_highlight_clips/` (Dossier de Dérushage)
    * `clip_001_10m42s.mp4`
    * `clip_002_25m11s.mp4`
    * `...`
* `MaVideo_highlight_tiktoks/` (Si l'option est cochée)
    * `tiktok_001_15m20s.mp4` (Clip 1, au format 9:16, compilation)
    * `...`
