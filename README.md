# Détection du Sarcasme en Arabe : Déséquilibre, Multi-tâche et Généralisation
### Projet Académique — Module : *Transformers & Large Language Models (LLMs)*
**Enseignant :** Prof. Yassir MATRANE  
**Auteur :** Mouad KARMA (*Master IASD / Ingénierie IA*)  
**Corpus :** [ArSarcasm-v2 (WANLP 2021)](https://github.com/iabufarha/ArSarcasm-v2) — 15 548 tweets annotés (Sarcasme, Sentiment, Dialecte)

---

##  Aperçu du Projet

Ce projet aborde la problématique complexe de la **détection du sarcasme dans les réseaux sociaux en langue arabe**, caractérisée par un fort déséquilibre de classes (~17% de sarcasme dans le train), une grande diversité dialectale (Égypte, Maghreb, Golfe, Levant, MSA) et une ambiguïté pragmatique élevée.

5 familles de modélisation ont été implémentées, comparées et soumises à des tests de robustesse industrielle :
1. **Baseline classique :** TF-IDF (mots 1–2 + caractères 2–5) + Régression Logistique avec pondération analytique.
2. **PLM Zero-Shot & Heuristiques :** Inférence textuelle (NLI) avec `mDeBERTa-v3-base-xnli` (Arabe, FR, EN) et transfert de polarité avec `CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment`.
3. **Embeddings & Réseaux Profonds :** BiLSTM avec Max-Pooling (C1) et BiLSTM avec pooling attentionnel auto-appris (C2), combiné au sur-échantillonnage de la classe minoritaire.
4. **PLM Gelé comme Extracteur :** `aubmindlab/bert-base-arabertv02-twitter` gelé (`[CLS]` vs *Mean Pooling*), et tête BiLSTM + Attention sur représentations profondes (D2).
5. **Fine-Tuning Transformer :** Fine-tuning complet mono-tâche (E1) et apprentissage **Multi-tâche conjointe Sarcasme + Sentiment** (E2) avec encodeur partagé.

---

## 🏆 Résultats Consolidés & Tableau Comparatif

> **Métrique principale :** F1-score de la classe sarcastique (optimisé sur l'ensemble de validation via recherche de seuil $\tau$).

| Famille / Modèle | Approche | F1-Score (Val) | F2 (Dialecte Inédit) | Taille | Paramètres Entraînés | Latence CPU |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | TF-IDF + LogReg (`balanced`) | 0.5085 | N/A | < 1 MB | ~100 K | < 1 ms |
| **Deep Learning** | BiLSTM + Attention (C2) | 0.4131 | N/A | ~2 MB | ~500 K | ~5 ms |
| **PLM Gelé** | AraBERT-twitter [CLS] (D1) | 0.6112 | N/A | ~135 MB | 0 (Extracteur) | ~50 ms |
| **PLM Gelé** | AraBERT-twitter Moyenne (D1) | 0.6184 | N/A | ~135 MB | 0 (Extracteur) | ~50 ms |
| **PLM Gelé + Tête**| AraBERT-twitter + BiLSTM-Attn (D2) | **0.6726** | N/A | ~137 MB | ~500 K | ~55 ms |
| **Fine-Tuned Mono**| AraBERT-twitter Mono (E1) | 0.6667 | **0.7803** | ~135 MB | 135 Millions | ~50 ms |
| **Fine-Tuned Multi**| AraBERT Multi-tâche (E2, $\lambda=0.5$) | **0.6681** | N/A | ~135 MB | 135 Millions | ~52 ms |

*Le score officiel record de la compétition internationale WANLP 2021 s'établit à **0.6225** sur le test officiel.*

---

## 🔬 Faits Marquants & Analyse Scientifique

1. **Prior Shift massif (F1) :**
   * Prévalence du sarcasme : **17.29%** en validation $\to$ **27.37%** sur le test officiel (+58% relatif).
   * Le seuil optimisé sur validation ($\tau = 0.20$) doit être recalibré vers [0.35 - 0.45] en production pour éviter une prolifération de faux positifs.
2. **Robustesse dialectale Zero-Shot (F2) :**
   * Entraîné sur 8 854 tweets **sans aucun exemple égyptien**, AraBERT conserve un F1 de **0.7803** sur les 306 tweets égyptiens du test, grâce à son pré-entraînement massif sur les tweets arabes.
   * **Biais géographique critique :** Le dialecte maghrébin (`magreb`) ne compte que **38 tweets en train** et **2 tweets en test**, nécessitant un benchmark dédié pour la Darija marocaine.
3. **Dépendance aux raccourcis (F3) :**
   * Retirer les tokens d'URLs et de mentions fait baisser le F1 de 0.6667 à **0.6316**, prouvant que le modèle exploitait partiellement ces biais de surface.
4. **Significativité statistique (F5) :**
   * Le test de McNemar confirme que la supériorité d'AraBERT sur la baseline est statistiquement significative avec $p = 2.55 \times 10^{-12} \ll 0.05$.

---

## 📂 Structure du Répertoire

```text
├── arabic_sarcasm_detection_lab.ipynb            # Notebook Jupyter complet avec code et sorties d'exécution
├── TP2_VersionB_Sarcasme_ArSarcasm-v2.docx # Compte-rendu officiel entièrement rédigé et rempli
├── requirements.txt                    # Dépendances Python figées
├── Dockerfile                          # Conteneurisation pour reproductibilité totale
├── README.md                           # Documentation du projet
├── extracted_img_38_1.png              # Matrice de confusion sur le test officiel
├── extracted_img_38_3.png              # Courbe F1 = f(seuil) illustrant le prior shift
└── figure_attention_d2.png             # Visualisation des poids d'attention sur 5 tweets sarcastiques
```

---

## 🚀 Guide d'Exécution & Reproductibilité

### Option 1 : Exécution locale (Python 3.10+)

```bash
# 1. Cloner le dépôt
git clone https://github.com/Mouad-Karma/arabic-sarcasm-detection-lab.git
cd ArSarcasm-v2-Detection

# 2. Créer et activer un environnement virtuel
python3 -m venv venv
source venv/bin/activate

# 3. Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt

# 4. Lancer Jupyter Notebook
jupyter notebook arabic_sarcasm_detection_lab.ipynb
```

### Option 2 : Exécution via Docker (Recommandé pour isolation totale)

```bash
# Construire l'image Docker
docker build -t arsarcasm-llm-tp2 .

# Lancer le conteneur avec GPU (si NVIDIA Container Toolkit installé)
docker run --gpus all -p 8888:8888 arsarcasm-llm-tp2

# Ou sur CPU standard :
docker run -p 8888:8888 arsarcasm-llm-tp2
```
Ouvrez ensuite votre navigateur sur `http://localhost:8888`.

### Option 3 : Google Colab (Gratuit avec GPU T4)
1. Téléversez le fichier `arabic_sarcasm_detection_lab.ipynb` sur [Google Colab](https://colab.research.google.com/).
2. Activez l'accélérateur GPU : **Exécution > Modifier le type d'exécution > T4 GPU**.
3. Exécutez les cellules séquentiellement.

---

## 📄 Livrable Officiel
Le compte-rendu complété avec toutes les justifications théoriques, les tableaux comparatifs et les figures intégrées se trouve dans :  
👉 **`TP2_VersionB_Sarcasme_ArSarcasm-v2.docx`**
