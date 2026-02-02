"""
Utilitaires pour la documentation de l'interface utilisateur.

Ce module contient les fonctions qui génèrent le contenu des onglets de documentation
de l'application Streamlit. Chaque fonction correspond à un onglet de documentation
et retourne le contenu markdown/HTML à afficher.
"""

import streamlit as st


def render_doc_irrigation_intelligente():
    """
    Affiche le contenu de l'onglet de documentation : Irrigation Intelligente.
    """
    st.markdown('<h2 class="section-header">💧 Pourquoi s\'intéresser à l\'irrigation intelligente ?</h2>', unsafe_allow_html=True)
    
    #st.image("images/logo_uttop.jpg", width=200)
        
    st.markdown("""
    Dans un contexte de changement climatique, l'eau devient une ressource rare, coûteuse et de plus en plus incertaine. 
    Les agriculteurs doivent arbitrer entre :
    
    - **Maintenir un niveau d'humidité adapté à la culture** : assurer une croissance optimale et un rendement satisfaisant
    - **Économiser l'eau** : respecter les quotas et réduire les coûts
    - **Éviter la lixiviation des nutriments** : prévenir les pertes en cas d'arrosage excessif
    
    ### Pratiques actuelles
    
    Aujourd'hui, de nombreux irrigants prennent leurs décisions à partir de :
    - **Seuils simples de tension** : ex. "si la tension dépasse 80 cbar, j'irrigue"
    - **Calendriers d'irrigation** : programmes fixes basés sur l'expérience
    - **Expérience personnelle** : intuition et connaissance du terrain
    """)
    
    st.markdown("### Opportunités technologiques")
    
    st.markdown("""
    <div style="background-color: #E3F2FD; padding: 20px; border-radius: 10px; border-left: 4px solid #2196F3; margin: 15px 0;">
    <p style="margin: 0 0 10px 0;"><strong>Avec l'essor de :</strong></p>
    <ul style="margin: 0 0 15px 0; padding-left: 20px;">
    <li><strong>Tensiomètres</strong> : mesure de la tension matricielle du sol</li>
    <li><strong>Prévisions météo</strong> : données de pluie et d'évapotranspiration</li>
    <li><strong>Techniques d'IA</strong> : apprentissage automatique et optimisation</li>
    </ul>
    <p style="margin: 10px 0 0 0;"><strong>La question devient :</strong></p>
    <blockquote style="margin: 15px 0 0 0; padding: 10px 15px; background-color: #BBDEFB; border-left: 3px solid #2196F3; border-radius: 5px; font-style: italic;">
    <strong>"Peut-on apprendre automatiquement une politique d'irrigation qui utilise les tensions mesurées, 
    respecte la physique du sol, et s'adapte à la parcelle réelle ?"</strong>
    </blockquote>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### Approche proposée
    
    Le travail présenté ici combine :
    - **Modèles physiques** : simulation du bilan hydrique du sol
    - **Modèles neuronaux** : Neural ODE / Neural CDE pour la correction
    - **Apprentissage par renforcement (RL)** : pour piloter l'irrigation à partir des séries temporelles de tension de l'eau ($\\psi_t$)
    """)


def render_doc_variables_etat():
    """
    Affiche le contenu de l'onglet de documentation : Variables d'État.
    """
    st.markdown('<h2 class="section-header">📊 Les variables clés : ce que l\'on mesure vraiment</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Une saison culturale est décrite jour par jour : $t = 0,1,\\dots,T$
    
    ### Variable effectivement mesurée : la tension de l'eau
    
    Le tensiomètre donne directement :
    
    $$\\psi_t \\quad (\\text{cbar})$$
    
    La tension est la force que la plante doit exercer pour extraire l'eau.
    C'est la **variable observée**, et c'est aussi ce que "ressent" réellement la culture.
    
    ### Variable interne non observée : la réserve en eau du sol
    
    Le modèle physique du bilan hydrique travaille pourtant avec une variable interne :
    
    $$S_t \\quad (\\text{mm})$$
    
    C'est la quantité d'eau stockée dans la zone racinaire.
    Mais **nous ne l'observons jamais directement**.
    
    ### Comment relier les deux ?
    
    On utilise une **courbe de rétention** propre au sol :
    
    $$\\psi_t = f_{\\text{retention}}(S_t) \\quad\\text{et idéalement}\\quad S_t = f_{\\text{retention}}^{-1}(\\psi_t)$$
    
    - Le tensiomètre mesure $\\psi_t$.
    - Le modèle reconstruit une estimation de $S_t$.
    - Le bilan hydrique agit sur $S_t$.
    - Puis on reconvertit en $\\psi_{t+1}$ pour comparaison.
    
    C'est une architecture à **états cachés**, courante en agro-hydrologie.
    """)
    
    st.markdown("### Paramètres de sol (pédophysique)")
    st.markdown("""
    - $Z_r$ : profondeur de la zone racinaire (mm)
    - $\\theta_s, \\ \\theta_{fc}, \\ \\theta_{wp}$ : saturation, capacité au champ, point de flétrissement
    - $S_{\\max} = \\theta_s \\cdot Z_r$, $S_{fc} = \\theta_{fc} \\cdot Z_r$, $S_{wp} = \\theta_{wp} \\cdot Z_r$
    """)
    
    st.markdown("### Flux entrants")
    st.markdown("""
    - $I_t$ : dose d'irrigation (mm)
    - $\\eta_I$ : efficacité d'irrigation (0–1)
    - $R_t$ : pluie (mm)
    - $G_t$ : remontée capillaire (optionnelle, mm)
    """)
    
    st.markdown("### Flux sortants")
    st.markdown("""
    - $ET0_t$ : évapotranspiration de référence (mm/j)
    - $Kc_t$ : coefficient cultural (adimensionnel) - représente la demande en eau de la culture selon son stade de développement
    - $f_{ET}(\\psi_t)$ : facteur de stress hydrique (0–1)
    - $ETc_t = Kc_t \\cdot ET0_t \\cdot f_{ET}(\\psi_t)$
    - $D(S_t)$ : drainage/percolation (mm)
    - $Q_t$ : ruissellement (optionnel, mm)
    """)
    
    st.markdown("### Dynamique physique (bilan hydrique)")
    st.markdown("""
    $$S_{t+1} = S_t + \\eta_I I_t + R_t + G_t - ETc_t - D(S_t) - Q_t$$
    
    $$\\psi_{t+1} = f_{\\text{retention}}(S_{t+1})$$
    
    Cette équation décrit l'évolution temporelle de la réserve en eau du sol en fonction des flux entrants 
    (irrigation $I_t$, pluie $R_t$, remontée capillaire $G_t$) et des flux sortants 
    (évapotranspiration $ETc_t$, drainage $D(S_t)$, ruissellement $Q_t$).
    """)
    
    st.markdown("### Contraintes opérationnelles")
    st.markdown("""
    - $I_{\\max}$ : dose journalière max
    - Quotas d'eau saisonniers
    - Fenêtres d'irrigation (heures/jours)
    - Pas de temps (journalier ; infra-journalier possible avec CDE)
    """)
    
    st.markdown("### Unités (rappel)")
    st.markdown("""
    - $\\psi$ : cbar
    - $S, I, R, G, D, Q$ : mm
    - $ET0$ : mm/j, $Kc$ : adimensionnel
    """)


def render_doc_apprentissage_renforcement():
    """
    Affiche le contenu de l'onglet de documentation : Apprentissage par Renforcement.
    """
    st.markdown('<h2 class="section-header">🤖 Rappel : qu\'est-ce que l\'Apprentissage par Renforcement (RL) ?</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Le RL modélise un problème de décision séquentielle sous forme de MDP (Markov Decision Process).
    
    - **État**: $s_t \\in \\mathcal{S}$ (ici, observables liées à la tension $\\psi_t$ et à la météo)
    - **Action**: $a_t \\in \\mathcal{A}$ (ici, dose d'irrigation $I_t$)
    - **Transition**: $p(s_{t+1}\\mid s_t, a_t)$ (dynamique du sol + météo)
    - **Récompense**: $r_t = r(s_t, a_t)$ (ex. stress hydrique, eau utilisée, drainage)
    - **Politique**: $\\pi_\\theta(a\\mid s)$, paramétrée (réseau de neurones)
    - **Objectif**: maximiser le retour $J(\\theta)=\\mathbb{E}_\\pi\\!\\left[\\sum_{t=0}^{T}\\gamma^t r_t\\right]$
    """)
    
    st.markdown("### Ce que voit l'agent (exemple d'observation)")
    st.markdown("""
    L'observation $o_t$ que reçoit l'agent RL combine les variables d'état physiques décrites précédemment 
    avec des informations contextuelles :
    
    $$o_t = (\\psi_t,\\ t/T,\\ R_{t-k:t},\\ ET0_t,\\ \\hat{R}_{t+1:t+h},\\ \\widehat{ET0}_{t+1:t+h},\\ \\text{évent. } \\psi_{t-k:t-1})$$
    
    Où :
    - **$\\psi_t$** : tension matricielle actuelle (variable mesurée, voir onglet "Variables d'état")
    - **$t/T$** : progression temporelle dans la saison (normalisée entre 0 et 1)
    - **$R_{t-k:t}$** : historique de pluie sur les $k$ jours précédents
    - **$ET0_t$** : évapotranspiration de référence actuelle
    - **$\\hat{R}_{t+1:t+h}$** : prévisions de pluie pour les $h$ prochains jours
    - **$\\widehat{ET0}_{t+1:t+h}$** : prévisions d'ET0 pour les $h$ prochains jours
    - **$\\psi_{t-k:t-1}$** (optionnel) : historique de tension pour capturer les tendances
    
    **Normalisation** : Les observations sont généralement standardisées ou clippées pour stabiliser l'apprentissage.
    """)
    
    st.markdown("### Espace d'actions")
    st.markdown("""
    L'action $a_t$ de l'agent correspond à la dose d'irrigation $I_t$ à appliquer :
    
    - **Continu** (Box): $I_t \\in [0, I_{\\max}]$ (mm) — cas le plus réaliste
      - Permet des doses précises et adaptatives
      - $I_{\\max}$ est une contrainte opérationnelle (débit maximal du système)
    
    - **Discret**: choix parmi des doses pré-définies (ex. $I_t \\in \\{0, 5, 10, 15, 20\\}$ mm)
      - Plus simple à implémenter mais moins flexible
    
    Le choix entre continu et discret est influencé par les contraintes opérationnelles et l'implémentation RL.
    """)
    
    st.markdown("### Conception de la récompense (exemples)")
    st.markdown("""
    - **Stress hydrique**: pénaliser les $\\psi_t$ hors zone de confort
      - $r^{stress}_t = -\\alpha\\ \\text{stress}(\\psi_t)$
    - **Eau utilisée**: pénaliser la quantité d'irrigation
      - $r^{eau}_t = -\\beta\\, I_t$
    - **Drainage/pertes**: pénaliser $D(S_t)$
      - $r^{drain}_t = -\\gamma\\, D(S_t)$
    - **Terminaison**: bonus de rendement – pénalité eau
      - $R_{final} = Y - \\lambda \\sum_t I_t$, avec $Y = Y_{\\max} \\exp(-k_{CS}\\sum_t \\text{stress}(\\psi_t))$
    
    Récompense totale typique: $r_t = r^{stress}_t + r^{eau}_t + r^{drain}_t$, puis ajout de $R_{final}$ en fin d'épisode.
    """)
    
    st.markdown("### PPO en bref (Proximal Policy Optimization)")
    st.markdown("""
    - **Type**: on-policy, gradient de politique
    - **Idée clé**: mise à jour "proche" de la politique courante via un objectif avec **clipping**
    - **Avantage**: variance réduite via l'estimation d'**avantage** $\\hat{A}_t$
    - **Compatibilité**: actions continues (Box) et discrètes
    - **Stabilité**: bonnes propriétés empiriques, tuning modéré
    
    Objectif PPO (schématique):
    - Maximiser $\\mathbb{E}_t\\left[\\min\\left(r_t(\\theta)\\hat{A}_t,\\ \\text{clip}(r_t(\\theta),1-\\epsilon,1+\\epsilon)\\hat{A}_t\\right)\\right]$
    - Avec $r_t(\\theta)=\\frac{\\pi_\\theta(a_t\\mid s_t)}{\\pi_{\\theta_{old}}(a_t\\mid s_t)}$, et une perte valeur + entropie
    """)
    
    st.markdown("### Boucle d'apprentissage (schéma)")
    st.markdown("""
    1. **Collecte**: rouler la politique $\\pi_\\theta$ dans l'environnement, stocker $(s_t,a_t,r_t,s_{t+1})$
    2. **Calcul**: retours, avantages (ex. GAE-$\\lambda$)
    3. **Update**: optimiser politique et critique (réseau valeur) via PPO
    4. **Évaluer/valider**: sur seeds/épisodes non vus
    5. **Répéter** jusqu'à convergence ou budget
    """)
    
    st.markdown("### Bonnes pratiques pour l'irrigation")
    st.markdown("""
    - **Exploration vs exploitation**: contrôler la stochasticité (entropie), garder des doses plausibles
    - **Contraintes**: intégrer $I_{\\max}$, quotas, fenêtres temporelles (via clipping, masques, pénalités)
    - **Robustesse**: randomiser météo/sols (domain randomization), gérer données manquantes
    - **Observations**: inclure prévisions météo, historiques, et indicateurs de confiance
    - **Échelle temporelle**: choisir $\\Delta t$ (journalier vs infra-journalier), cohérent avec le modèle (CDE si irrégulier)
    - **Évaluation**: multi-saisons, scénarios secs/humides, métriques eau, stress, rendement
    """)
    
    st.markdown("### Exemple d'état et de politique (illustratifs)")
    st.markdown("""
    - État: $s_t = (\\psi_t,\\ t/T,\\ R_{t-2:t},\\ ET0_t,\\ \\hat{R}_{t+1:t+2})$
    - Heuristiques utiles que PPO peut apprendre:
      - **Pluie prévue** → réduire $I_t$
      - **$\\psi_t$ élevé (sol sec)** → irriguer
      - **$\\psi_t$ modéré mais ET0 fort** → irrigation préventive
    
    En résumé, le RL (et en particulier PPO) offre un cadre pour apprendre automatiquement une stratégie d'irrigation 
    tenant compte de la dynamique du sol, des prévisions et des coûts, tout en gérant des espaces d'actions continus 
    et des observations riches.
    """)

# ========================================================================
# ONGLET DOCUMENTATION 4 : SCÉNARIO 2 (RL SUR MODÈLE PHYSIQUE)
# ========================================================================

def render_doc_scenario2():
    """
    Affiche le contenu de l'onglet de documentation : Scénario 2 (RL sur modèle physique).
    """
    st.markdown('<h2 class="section-header">🎓 RL sur modèle physique (avec tension observée)</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### ❓ Qu'est-ce que RL sur modèle physique ?
    
    Le **RL sur modèle physique** implémente un agent d'apprentissage par renforcement (RL) qui apprend une politique d'irrigation optimale
    en interagissant directement avec un environnement simulé par le modèle physique FAO. Contrairement au Scénario 1 qui utilise
    des règles fixes, ce Scénario apprend automatiquement à optimiser l'irrigation en minimisant le stress hydrique tout en
    économisant l'eau.
    
    **Principe fondamental** : Un agent RL observe la tension matricielle $\\psi_t$ (et le contexte météorologique) et choisit
    une dose d'irrigation $I_t$ dans un environnement simulé par le modèle physique. L'agent apprend à partir de ses interactions
    avec l'environnement pour améliorer progressivement sa stratégie.
    """)
    
    with st.expander("🔬 Architecture MDP (Markov Decision Process)", expanded=False):
        st.markdown("""
        Le Scénario 2 modélise le problème d'irrigation comme un MDP :
        
        **1. Espace d'observation** ($\\mathcal{S}$) :
        - Observation standard : $o_t = [\\psi_t, S_t, R_t, ET0_t] \\in \\mathbb{R}^4$
          - $\\psi_t$ : Tension matricielle actuelle (cbar) - variable clé mesurée par tensiomètre
          - $S_t$ : Réserve en eau du sol (mm)
          - $R_t$ : Pluie du jour (mm)
          - $ET0_t$ : Évapotranspiration de référence (mm/j)
        - Observation enrichie (optionnelle) : $o_t = (\\psi_t, t/T, R_{t-k:t}, ET0_t, \\hat{R}_{t+1:t+h}, \\widehat{ET0}_{t+1:t+h})$
          - $t/T$ : Progression temporelle dans la saison (normalisée)
          - $R_{t-k:t}$ : Historique de pluie sur $k$ jours précédents
          - $\\hat{R}_{t+1:t+h}$ : Prévisions de pluie pour les $h$ prochains jours
          - $\\widehat{ET0}_{t+1:t+h}$ : Prévisions d'ET0 pour les $h$ prochains jours
        
        **2. Espace d'actions** ($\\mathcal{A}$) :
        - Action continue : $a_t = I_t \\in [0, I_{\\max}]$ (mm)
          - $I_{\\max}$ : Irrigation maximale par jour (contrainte opérationnelle)
          - Permet des doses précises et adaptatives
        
        **3. Fonction de transition** ($p(s_{t+1} | s_t, a_t)$) :
        - Modèle physique FAO : $S_{t+1} = f_{\\text{FAO}}(S_t, I_t, R_t, ET0_t, Kc_t)$
        - Conversion : $\\psi_{t+1} = f_{\\text{retention}}(S_{t+1})$
        - Le modèle physique garantit la cohérence des prédictions
        
        **4. Fonction de récompense** ($r_t = r(s_t, a_t)$) :
        - Récompense journalière : $r_t = -\\alpha \\cdot \\text{stress}(\\psi_t) - \\beta \\cdot I_t - \\gamma \\cdot D(S_t)$
          - $\\alpha$ : Poids de la pénalité de stress hydrique
          - $\\beta$ : Poids de la pénalité d'irrigation (coût de l'eau)
          - $\\gamma$ : Poids de la pénalité de drainage (pertes d'eau)
        - Récompense terminale : $R_{\\text{final}} = Y(\\text{cum\_stress}) - \\lambda_{\\text{water}} \\cdot \\sum_t I_t$
          - $Y$ : Rendement de la culture (décroît avec le stress cumulé)
          - $\\lambda_{\\text{water}}$ : Poids de la pénalité d'eau totale
        
        **5. Algorithme RL** :
        - **PPO (Proximal Policy Optimization)** : Algorithme on-policy, gradient de politique
        - Objectif : Maximiser $J(\\theta) = \\mathbb{E}_\\pi\\left[\\sum_{t=0}^{T} \\gamma^t r_t\\right]$
        - $\\gamma$ : Discount factor (0.99 recommandé pour planification long terme)
        """)
    
    with st.expander("🎯 Application dans notre projet : Pipeline d'entraînement", expanded=False):
        st.markdown("""
        ### Processus d'entraînement du Scénario 2
        
        **1. Génération de l'environnement** :
        - Création de l'environnement Gymnasium avec le modèle physique FAO
        - Configuration des paramètres (sol, météo, récompenses)
        - Génération de séries météorologiques avec seed pour reproductibilité
        
        **2. Initialisation de l'agent PPO** :
        - Réseau de politique (policy network) : MLP qui mappe observation → distribution d'actions
        - Réseau de valeur (value network) : MLP qui estime $V(s_t)$ pour réduire la variance
        - Hyperparamètres PPO (learning rate, gamma, GAE lambda, etc.)
        
        **3. Boucle d'entraînement** :
        - **Collecte de données** :
          - Rollout de la politique actuelle $\\pi_\\theta$ dans l'environnement
          - Stockage des trajectoires : $(s_t, a_t, r_t, s_{t+1})$
          - Calcul des retours et avantages (GAE-$\\lambda$)
        - **Mise à jour de la politique** :
          - Optimisation de l'objectif PPO avec clipping
          - Mise à jour du réseau de valeur
          - Contrôle de l'exploration via coefficient d'entropie
        - **Évaluation** :
          - Test sur épisodes de validation (seeds différents)
          - Calcul de métriques (récompense moyenne, longueur d'épisode, etc.)
        
        **4. Utilisation du modèle entraîné** :
        - Chargement du modèle PPO sauvegardé
        - Évaluation sur nouvelles saisons
        - Déploiement pour prise de décision en temps réel
        """)
    
    with st.expander("📐 Architecture de l'agent PPO", expanded=False):
        st.markdown("""
        ### Réseau de politique (Policy Network)
        
        **Architecture** :
        - **Type** : MLP (Multi-Layer Perceptron)
        - **Entrée** : Observation $o_t \\in \\mathbb{R}^d$ où $d$ est la dimension de l'observation (4 par défaut)
        - **Couches cachées** : 2-3 couches avec 64-256 neurones chacune
        - **Activation** : Tanh ou ReLU
        - **Sortie** : Paramètres d'une distribution d'actions
          - Pour actions continues : Moyenne $\\mu(o_t)$ et écart-type $\\sigma(o_t)$ d'une distribution normale
          - Action échantillonnée : $a_t \\sim \\mathcal{N}(\\mu(o_t), \\sigma(o_t))$
        
        **Formule** :
        $$
        \\begin{aligned}
        \\mathbf{h}_1 &= \\text{ReLU}(\\text{Linear}_1(o_t)) \\\\
        \\mathbf{h}_2 &= \\text{ReLU}(\\text{Linear}_2(\\mathbf{h}_1)) \\\\
        \\mu_t &= \\text{Linear}_\\mu(\\mathbf{h}_2) \\\\
        \\sigma_t &= \\text{softplus}(\\text{Linear}_\\sigma(\\mathbf{h}_2)) \\\\
        a_t &\\sim \\mathcal{N}(\\mu_t, \\sigma_t)
        \\end{aligned}
        $$
        
        ### Réseau de valeur (Value Network)
        
        **Architecture** :
        - **Type** : MLP similaire au réseau de politique
        - **Entrée** : Observation $o_t$
        - **Sortie** : Estimation de la valeur $V(o_t) = \\mathbb{E}_\\pi\\left[\\sum_{k=0}^{T-t} \\gamma^k r_{t+k} | o_t\\right]$
        
        **Rôle** :
        - Réduit la variance de l'estimation du gradient
        - Utilisé dans le calcul de l'avantage : $\\hat{A}_t = \\delta_t + (\\gamma \\lambda) \\delta_{t+1} + \\ldots$
        - Où $\\delta_t = r_t + \\gamma V(o_{t+1}) - V(o_t)$ est le TD-error
        
        ### Objectif PPO
        
        L'objectif PPO combine plusieurs termes :
        
        $$
        L^{\\text{PPO}}(\\theta) = \\mathbb{E}_t\\left[L^{\\text{CLIP}}(\\theta) - c_v L^V(\\theta) + c_e H[\\pi_\\theta](o_t)\\right]
        $$
        
        où :
        - $L^{\\text{CLIP}}(\\theta) = \\min\\left(r_t(\\theta) \\hat{A}_t, \\text{clip}(r_t(\\theta), 1-\\epsilon, 1+\\epsilon) \\hat{A}_t\\right)$
        - $r_t(\\theta) = \\frac{\\pi_\\theta(a_t | o_t)}{\\pi_{\\theta_{\\text{old}}}(a_t | o_t)}$ est le ratio de probabilité
        - $L^V(\\theta) = (V_\\theta(o_t) - \\hat{V}_t)^2$ est la perte de valeur
        - $H[\\pi_\\theta](o_t)$ est l'entropie de la politique (encourage l'exploration)
        - $c_v$ et $c_e$ sont des coefficients de pondération
        """)
    
    with st.expander("⚖️ Avantages et limitations", expanded=False):
        st.markdown("""
        ### ✅ Avantages du Scénario 2
        
        **1. Apprentissage d'une politique optimale** :
        - ✅ **Optimisation automatique** : L'agent RL apprend à minimiser le stress hydrique tout en économisant l'eau
        - ✅ **Compromis optimal** : Trouve automatiquement le meilleur équilibre entre performance agronomique et coût de l'eau
        - ✅ **Stratégie adaptative** : S'ajuste selon les conditions météorologiques et l'état du sol
        
        **2. Adaptabilité aux conditions** :
        - ✅ **Prévisions météo** : Utilise les prévisions de pluie et d'ET0 pour anticiper et ajuster l'irrigation
        - ✅ **Conditions variables** : S'adapte aux variations saisonnières et aux événements météorologiques
        - ✅ **Historique contextuel** : Prend en compte l'historique récent (pluie, tension) pour des décisions informées
        
        **3. Respect de la physique** :
        - ✅ **Modèle physique fiable** : Utilise un modèle bucket validé pour simuler la dynamique du sol
        - ✅ **Courbe de rétention** : Respecte la relation $S \\leftrightarrow \\psi$ basée sur les propriétés pédophysiques
        - ✅ **Bilan hydrique cohérent** : Les équations physiques garantissent la cohérence des prédictions
        
        **4. Flexibilité des actions** :
        - ✅ **Actions continues** : Permet des doses d'irrigation précises et graduées (pas seulement 0 ou dose fixe)
        - ✅ **Doses adaptatives** : Ajuste la quantité d'eau selon l'intensité du stress et les conditions
        - ✅ **Stratégie préventive** : Peut irriguer préventivement avant que le stress ne devienne critique
        
        **5. Performance supérieure** :
        - ✅ **Efficacité de l'eau** : Généralement meilleure que les règles fixes en termes de consommation d'eau
        - ✅ **Réduction du stress** : Maintient mieux la tension dans la zone de confort
        - ✅ **Minimisation du drainage** : Apprend à éviter les pertes d'eau par drainage excessif
        
        **6. Réutilisabilité** :
        - ✅ **Modèle entraîné** : Une fois entraîné, le modèle peut être utilisé sur différentes saisons
        - ✅ **Transfert possible** : Peut être adapté à d'autres parcelles avec ré-entraînement
        - ✅ **Amélioration continue** : Peut être ré-entraîné avec de nouvelles données pour s'améliorer
        
        ### ⚠️ Limitations du Scénario 2
        
        **1. Dépendance à la qualité du modèle physique** :
        - ⚠️ **Biais du modèle** : Si le modèle bucket a des biais (paramètres mal calibrés, processus négligés), 
          la politique apprise sera biaisée
        - ⚠️ **Erreurs de paramétrisation** : Des erreurs dans les paramètres du sol ($S_{fc}$, $\\psi_{fc}$, $k_d$) 
          se propagent dans les décisions
        - ⚠️ **Processus non modélisés** : Phénomènes non capturés par le modèle (hétérogénéité spatiale, 
          interactions complexes) ne sont pas pris en compte
        
        **2. Phase d'entraînement nécessaire** :
        - ⚠️ **Temps d'entraînement** : Nécessite une phase d'apprentissage (plusieurs milliers de timesteps) 
          avant d'être utilisable
        - ⚠️ **Ressources computationnelles** : Entraînement PPO nécessite des ressources CPU/GPU
        - ⚠️ **Expertise technique** : Nécessite des compétences en RL pour l'entraînement et le réglage
        
        **3. Données d'entraînement** :
        - ⚠️ **Simulation requise** : Besoin de générer des données de simulation pour l'entraînement
        - ⚠️ **Qualité de la simulation** : La qualité de l'entraînement dépend de la qualité de la simulation météo
        - ⚠️ **Robustesse** : Nécessite d'entraîner sur plusieurs saisons/scénarios pour être robuste
        
        **4. Complexité de déploiement** :
        - ⚠️ **Infrastructure** : Nécessite une infrastructure pour exécuter le modèle entraîné
        - ⚠️ **Maintenance** : Le modèle peut nécessiter un ré-entraînement périodique
        - ⚠️ **Interprétabilité réduite** : Moins interprétable que les règles simples (boîte noire)
        
        **5. Hyperparamètres à régler** :
        - ⚠️ **Tuning nécessaire** : Nombreux hyperparamètres à ajuster (learning rate, gamma, GAE-$\\lambda$, etc.)
        - ⚠️ **Sensibilité** : La performance peut être sensible aux choix d'hyperparamètres
        - ⚠️ **Expertise requise** : Nécessite une compréhension du RL pour optimiser les hyperparamètres
        
        **6. Stabilité de l'apprentissage** :
        - ⚠️ **Convergence** : L'entraînement peut ne pas converger ou converger vers un optimum local
        - ⚠️ **Variabilité** : La performance peut varier entre différentes exécutions d'entraînement
        - ⚠️ **Normalisation** : Nécessite une normalisation soigneuse des observations et récompenses
        
        **7. Observations cohérentes** :
        - ⚠️ **Alignement temporel** : Nécessite que les observations soient alignées temporellement
        - ⚠️ **Données manquantes** : Doit gérer les cas de données manquantes ou irrégulières
        - ⚠️ **Prévisions météo** : Dépend de la qualité des prévisions météorologiques disponibles
        """)
    
    with st.expander("🔧 Paramètres recommandés et tuning", expanded=False):
        st.markdown("""
        ### Hyperparamètres PPO
        
        **Learning rate** : $3 \\times 10^{-4}$ (recommandé)
        - Trop élevé (> $10^{-3}$) : Instabilité, oscillations
        - Trop faible (< $10^{-5}$) : Apprentissage trop lent
        - Tuning : Réduire si loss oscille, augmenter si convergence lente
        
        **Gamma (discount factor)** : 0.99 (recommandé)
        - Contrôle l'importance des récompenses futures
        - Élevé (0.99) : Planification à long terme
        - Faible (0.95) : Focus sur court terme
        
        **GAE lambda** : 0.95 (recommandé)
        - Contrôle le biais/variance de l'estimation de la valeur
        - Élevé (0.95-0.99) : Moins de variance, plus de biais
        - Faible (0.8-0.9) : Plus de variance, moins de biais
        
        **Entropy coefficient** : 0.01-0.05
        - Encourage l'exploration
        - Élevé : Plus d'exploration, convergence plus lente
        - Faible : Moins d'exploration, risque de sous-optimum local
        
        **Clip range** : 0.2 (standard PPO)
        - Limite les changements de politique
        - Élevé : Permet plus de changements, moins stable
        - Faible : Changements limités, plus stable
        
        **Batch size** : 64-256
        - Plus grand : Gradients plus stables mais plus de mémoire
        - Plus petit : Moins de mémoire mais gradients plus variables
        
        **Number of steps per rollout** : 2048
        - Plus grand : Meilleure estimation mais plus de mémoire
        - Plus petit : Moins de mémoire mais estimation moins précise
        
        ### Hyperparamètres de l'environnement
        
        **Paramètres de récompense** :
        - $\\alpha$ (pénalité stress) : 1.0 (recommandé)
        - $\\beta$ (pénalité irrigation) : 0.05 (recommandé)
        - $\\gamma$ (pénalité drainage) : 0.01 (recommandé)
        - Tuning : Ajuster selon priorités (eau vs stress)
        
        **Paramètres du sol** :
        - Utiliser les valeurs par défaut sauf si données spécifiques disponibles
        - Calibrer $S_{fc}$, $\\psi_{fc}$ selon mesures réelles si possible
        
        ### Stratégie de tuning
        
        **1. Commencer avec valeurs par défaut** :
        - Utiliser les valeurs recommandées ci-dessus
        - Entraîner sur 50,000-100,000 timesteps
        
        **2. Observer les métriques** :
        - Récompense moyenne : Doit augmenter
        - Longueur d'épisode : Doit être stable
        - Variance des actions : Ne doit pas exploser
        
        **3. Ajuster si nécessaire** :
        - Si instabilité : Réduire learning rate, augmenter clip range
        - Si convergence lente : Augmenter learning rate, réduire entropy
        - Si sous-optimum : Augmenter entropy, réduire clip range
        """)
    
    with st.expander("🧭 Quand utiliser le Scénario 2 ?", expanded=False):
        st.markdown("""
        ### ✅ Choisir le Scénario 2 si :
        
        - **Optimisation recherchée** :
          - Besoin de minimiser la consommation d'eau
          - Recherche du compromis optimal stress/coût
          - Performance supérieure aux règles simples
        
        - **Données disponibles** :
          - Possibilité de générer des simulations pour l'entraînement
          - Modèle physique fiable et bien calibré
          - Conditions météorologiques variées pour robustesse
        
        - **Ressources computationnelles** :
          - Infrastructure disponible pour l'entraînement PPO
          - Temps d'entraînement acceptable (quelques heures)
          - Expertise en RL disponible
        
        - **Adaptabilité nécessaire** :
          - Conditions variables nécessitant adaptation
          - Besoin de stratégie préventive
          - Optimisation selon objectifs multiples
        
        - **Point de départ pour approches avancées** :
          - Baseline pour comparer avec Scénarios 3-6
          - Validation de l'approche RL avant complexification
        
        ### ❌ Ne pas choisir le Scénario 2 si :
        
        - **Simplicité prioritaire** :
          - Besoin de solution simple et rapide
          - Pas d'infrastructure d'entraînement
          - Règles simples suffisent
        
        - **Modèle physique incertain** :
          - Paramètres du sol mal connus
          - Modèle physique non validé
          - Données de simulation de mauvaise qualité
        
        - **Données limitées** :
          - Pas de possibilité de générer des simulations
          - Conditions trop spécifiques pour généraliser
        
        - **Besoin de correction physique** :
          - Modèle physique a des biais connus
          - Nécessité de corriger les prédictions physiques
          - → Préférer Scénarios 3-4
        """)
    
    with st.expander("🛠️ Conseils pratiques", expanded=False):
        st.markdown("""
        ### Workflow recommandé
        
        **1. Préparation** :
        - Vérifier la cohérence météo (mêmes seeds/params que Scénario 1)
        - Valider le modèle physique sur quelques épisodes
        - Configurer les hyperparamètres avec valeurs par défaut
        
        **2. Entraînement initial** :
        - Commencer avec 50,000 timesteps
        - Observer les métriques (récompense, longueur d'épisode)
        - Vérifier la convergence
        
        **3. Tuning itératif** :
        - Ajuster les hyperparamètres si nécessaire
        - Ré-entraîner avec nouveaux paramètres
        - Comparer les performances
        
        **4. Évaluation** :
        - Tester sur nouvelles saisons (seeds différents)
        - Comparer avec Scénario 1 (baseline)
        - Analyser les décisions prises
        
        ### Troubleshooting
        
        **Problème : Instabilité de l'entraînement**
        - **Symptôme** : Loss oscille, récompense ne converge pas
        - **Solutions** :
          - Réduire learning rate (ex: $3 \\times 10^{-4} \\to 10^{-4}$)
          - Augmenter clip range (ex: 0.2 → 0.3)
          - Normaliser les observations et récompenses
        
        **Problème : Convergence lente**
        - **Symptôme** : Récompense augmente très lentement
        - **Solutions** :
          - Augmenter learning rate (avec prudence)
          - Augmenter entropy coefficient pour plus d'exploration
          - Vérifier la normalisation des récompenses
        
        **Problème : Sous-optimum local**
        - **Symptôme** : Performance plafonne à un niveau sous-optimal
        - **Solutions** :
          - Augmenter entropy coefficient
          - Réduire clip range pour permettre plus de changements
          - Augmenter le nombre de timesteps d'entraînement
        
        **Problème : Politique trop conservatrice**
        - **Symptôme** : Irrigation insuffisante, stress hydrique
        - **Solutions** :
          - Ajuster les poids de récompense ($\\alpha$ vs $\\beta$)
          - Augmenter la pénalité de stress ($\\alpha$)
          - Réduire la pénalité d'irrigation ($\\beta$)
        
        ### Métriques à surveiller
        
        - **Récompense moyenne** : Doit augmenter avec l'entraînement
        - **Longueur d'épisode** : Doit être stable (≈ longueur de saison)
        - **Variance des actions** : Ne doit pas exploser (signe d'instabilité)
        - **Policy loss** : Doit décroître et converger
        - **Value loss** : Doit décroître (estimation de la valeur)
        """)
    
    with st.expander("🔗 Comparaison avec les autres scénarios", expanded=False):
        st.markdown("""
        ### Scénario 2 vs Scénario 1 (Règles simples)
        
        **Scénario 1** :
        - Règles fixes, pas d'apprentissage
        - Simple et rapide
        - Performance sous-optimale
        
        **Scénario 2** :
        - Apprentissage automatique
        - Plus complexe mais meilleure performance
        - Nécessite entraînement
        
        **Quand choisir Scénario 2** : Optimisation et adaptabilité recherchées
        
        ### Scénario 2 vs Scénarios 3-4 (Neural ODE/CDE)
        
        **Scénario 2** :
        - RL direct sur modèle physique
        - Pas de correction du modèle physique
        - Plus simple
        
        **Scénarios 3-4** :
        - Correction résiduelle du modèle physique
        - Améliore la prédiction physique
        - Plus complexe
        
        **Quand choisir Scénarios 3-4** : Modèle physique a des biais connus
        
        ### Scénario 2 vs Scénario 5 (PatchTST)
        
        **Scénario 2** :
        - Observation standard (4 dimensions)
        - Pas de mémoire temporelle explicite
        
        **Scénario 5** :
        - Observation enrichie avec features temporelles
        - Mémoire longue via PatchTST
        
        **Quand choisir Scénario 5** : Besoin de comprendre tendances et saisonnalité
        
        ### Scénario 2 vs Scénario 6 (World Model)
        
        **Scénario 2** :
        - Model-free RL
        - Pas de planification explicite
        
        **Scénario 6** :
        - Model-based RL avec planification
        - Rollouts d'imagination
        
        **Quand choisir Scénario 6** : Besoin de planification et sample efficiency
        """)
    
    with st.expander("📊 Variables et notations", expanded=False):
        st.markdown("""
        ### Variables principales
        
        **Observations et états** :
        - $o_t = [\\psi_t, S_t, R_t, ET0_t] \\in \\mathbb{R}^4$ : Observation au temps $t$
        - $s_t$ : État du MDP (peut être identique à $o_t$ ou enrichi)
        - $a_t = I_t \\in [0, I_{\\max}]$ : Action (irrigation) au temps $t$
        - $r_t$ : Récompense au temps $t$
        
        **Modèle physique** :
        - $S_t$ : Réserve en eau du sol (mm)
        - $\\psi_t$ : Tension matricielle (cbar)
        - $R_t$ : Pluie (mm)
        - $ET0_t$ : Évapotranspiration de référence (mm/j)
        - $Kc_t$ : Coefficient cultural
        - $ETc_t = Kc_t \\times ET0_t \\times f_{ET}(\\psi_t)$ : Évapotranspiration culturelle
        - $D(S_t)$ : Drainage (mm)
        
        **Fonctions et modèles** :
        - $\\pi_\\theta(a_t | o_t)$ : Politique PPO (distribution d'actions)
        - $V_\\theta(o_t)$ : Fonction de valeur (estimation du retour futur)
        - $g(\\psi_t; \\theta_{\\text{règle}})$ : Règle d'irrigation déterministe du Scénario 1 (seuil, bande de confort, proportionnelle) qui renvoie $I_t$
        - $f_{\\text{FAO}}(\\cdot)$ : Modèle physique FAO
        - $f_{\\text{retention}}(\\cdot)$ : Courbe de rétention $S \\leftrightarrow \\psi$
        
        **Entraînement** :
        - $\\theta$ : Paramètres de la politique et de la fonction de valeur
        - $\\gamma$ : Discount factor (0.99)
        - $\\lambda$ : Paramètre GAE (0.95)
        - $\\epsilon$ : Clip range (0.2)
        - $\\hat{A}_t$ : Estimation de l'avantage (GAE)
        """)

# ========================================================================
# ONGLET DOCUMENTATION 5 : NEURAL ODE
# ========================================================================

def render_doc_neural_ode():
    """
    Affiche le contenu de l'onglet de documentation : Neural ODE.
    """
    st.markdown('<h2 class="section-header">🧠 Neural ODE : Modèle hybride physique-neuronal</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### ❓ Qu'est-ce qu'un Neural ODE ?
    
    Un **Neural ODE** (Neural Ordinary Differential Equation) est un modèle qui combine des équations différentielles ordinaires (ODE) avec des réseaux de neurones. 
    Dans notre contexte, il s'agit d'un **modèle résiduel** qui apprend à corriger les prédictions d'un modèle physique.
    """)
    
    with st.expander("🔬 Principe général des Neural ODE", expanded=False):
        st.markdown("""
        Les Neural ODE modélisent la dynamique d'un système par une équation différentielle où la dérivée est apprise par un réseau de neurones :
        
        $$
        \\frac{d\\mathbf{z}(t)}{dt} = f_\\theta(\\mathbf{z}(t), t)
        $$
        
        où :
        - $\\mathbf{z}(t)$ est l'état du système au temps $t$
        - $f_\\theta$ est un réseau de neurones paramétré par $\\theta$
        - La solution est obtenue par intégration numérique (méthode d'Euler, Runge-Kutta, etc.)
        
        **Avantages** :
        - **Continuité** : Modélise des processus continus naturellement
        - **Efficacité mémoire** : Pas besoin de stocker tous les états intermédiaires
        - **Flexibilité** : Le réseau apprend la dynamique à partir des données
        """)
    
    with st.expander("🎯 Application dans notre projet : Modèle hybride", expanded=False):
        st.markdown("""
        Dans notre projet d'irrigation intelligente, le Neural ODE est utilisé comme **correction résiduelle** sur le modèle physique FAO.
        L'idée est de combiner :
        
        - **Modèle physique** : Fournit une prédiction de base basée sur les lois de la physique (bilan hydrique)
        - **Neural ODE** : Apprend les écarts systématiques et les phénomènes non modélisés par le modèle physique
        
        Cette approche hybride permet de :
        - ✅ Bénéficier de la robustesse et de l'interprétabilité du modèle physique
        - ✅ Capturer les biais systématiques et les phénomènes complexes non modélisés
        - ✅ S'adapter aux spécificités locales (type de sol, conditions météo, etc.)
        
        ### 📋 Paramètres de configuration du pré-entraînement
        
        **Nombre de trajectoires** :
        - **Signification** : Nombre de simulations indépendantes utilisées pour générer les données d'entraînement du Neural ODE
        - **Valeur usuelle** : 32 trajectoires (par défaut)
        - **Impact** : Plus de trajectoires = plus de diversité dans les données (conditions météo variées, stratégies d'irrigation différentes)
        - **Tuning** : Augmenter (50-100) si le modèle ne généralise pas bien, réduire (10-20) pour accélérer l'entraînement
        - **Note** : Chaque trajectoire simule une saison complète (120 jours par défaut)
        
        **Nombre d'epochs** :
        - **Signification** : Nombre de passages complets sur l'ensemble des données d'entraînement
        - **Valeur usuelle** : 10 epochs (par défaut)
        - **Impact** : Plus d'epochs = meilleur apprentissage mais risque de surapprentissage
        - **Tuning** : Augmenter (20-50) si la loss continue à diminuer, réduire si la loss stagne ou augmente
        - **Note** : Surveiller la loss de validation pour détecter le surapprentissage
        
        **Taille des batches** :
        - **Signification** : Nombre d'échantillons traités simultanément lors de chaque mise à jour des paramètres
        - **Valeur usuelle** : 256 (par défaut)
        - **Impact** : Batch plus grand = gradients plus stables mais plus de mémoire requise
        - **Tuning** : Réduire (32-128) si mémoire limitée, augmenter (512+) si disponible et pour plus de stabilité
        - **Note** : Doit être adapté à la taille du dataset (nombre de trajectoires × longueur de saison)
        
        **Taux d'apprentissage (Learning Rate)** :
        - **Signification** : Vitesse à laquelle le modèle ajuste ses paramètres lors de l'optimisation
        - **Valeur usuelle** : $10^{-3}$ (0.001) (par défaut)
        - **Impact** : LR trop élevé = instabilité, LR trop faible = apprentissage lent
        - **Tuning** : Réduire (10^{-4} - 10^{-5}) si la loss oscille, augmenter (10^{-2}) si l'apprentissage est trop lent
        - **Note** : Utilise l'optimiseur Adam qui adapte le LR par paramètre
        """)


def render_doc_neural_ode_cont():
    """
    Affiche le contenu de l'onglet de documentation : Neural ODE continu (Scénario 3b).
    """
    st.markdown('<h2 class="section-header">🧠 Neural ODE continu : correction résiduelle lissée</h2>', unsafe_allow_html=True)

    st.markdown("""
    ### ❓ Qu'est-ce qu'un Neural ODE continu ?

    Variante continue du Neural ODE : le réseau apprend directement la dérivée $d\\psi/dt$
    et l'intègre sur un pas (1 jour) avec `torchdiffeq` (Runge-Kutta) ou un Euler explicite fallback.
    Résultat : corrections $\\Delta \\psi$ plus lisses que la version discrète.
    """)

    with st.expander("🔬 Principe général", expanded=False):
        st.markdown("""
        $$
        \\frac{d\\psi}{dt} = f_\\theta(\\psi, I, R, ET0) \\quad\\Rightarrow\\quad
        \\Delta\\psi = \\int_{t}^{t+\\Delta t} f_\\theta(\\psi, I, R, ET0)\\, dt
        $$

        - **Intégration** : `torchdiffeq.odeint` (rk4) si dispo, sinon Euler multi-sous-pas.
        - **Avantage** : correction temporellement plus régulière (moins d'oscillations sur ψ et I).
        """)

    with st.expander("🎯 Application hybride (Scénario 3b)", expanded=False):
        st.markdown("""
        - **Base physique** : même bucket FAO que le scénario 3.
        - **Correction continue** : $\\psi_{t+1} = \\psi_{t+1}^{phys} + \\Delta\\psi_{cont}$.
        - **Impact attendu** : transitions plus douces, politique PPO moins sujette aux à-coups.
        """)

    with st.expander("🧰 Hyperparamètres clés", expanded=False):
        st.markdown("""
        - **Trajectoires (N_traj)** : 32 par défaut. Augmenter (50-100) si la loss continue de baisser.
        - **Epochs** : 10 par défaut. Monter à 20-50 si sous-apprentissage.
        - **Batch size** : 256 par défaut. Réduire (64-128) si mémoire limitée.
        - **Learning rate** : $10^{-3}$ par défaut. Baisser (1e-4) si la loss oscille.
        - **Solver** : rk4 via `torchdiffeq` si installé ; sinon Euler avec sous-pas (paramètre `substeps`).
        """)

    with st.expander("📐 Architecture", expanded=False):
        st.markdown("""
        - **Réseau f_θ (dψ/dt)** : MLP 4→64→64→1 avec Tanh.
        - **Entrée** : [ψ, I, R, ET0].
        - **Sortie** : dψ/dt, intégré sur Δt=1 jour pour produire Δψ.
        """)

    with st.expander("🚀 Intégration RL", expanded=False):
        st.markdown("""
        - **Pré-entraîner** le modèle continu sur données simulées (Étape 1 onglet 3b).
        - **Entraîner PPO** sur l'environnement hybride (Étape 2 onglet 3b) : mêmes récompenses que scénarios 2/3.
        - **Évaluation** : onglet Évaluation → choisir "Scénario 3b", puis Visualisation/Comparaison.
        """)
    
    with st.expander("📐 Architecture du modèle hybride", expanded=False):
        st.markdown("""
        ### Architecture du modèle hybride
    
    Le modèle hybride combine deux composantes :
    
    **a) Prédiction physique** :
    
    Le modèle physique calcule d'abord la réserve en eau $S_{t+1}^{\\text{phys}}$ selon le bilan hydrique :
    
    $$
    S_{t+1}^{\\text{phys}} = S_t + \\eta_I I_t + R_t - ETc_t - D(S_t)
    $$
    
    où :
    - $S_t$ : Réserve en eau au jour $t$ (mm)
    - $\\eta_I$ : Efficacité d'irrigation (fraction de l'eau d'irrigation effectivement disponible)
    - $I_t$ : Dose d'irrigation appliquée au jour $t$ (mm)
    - $R_t$ : Pluie au jour $t$ (mm)
    - $ETc_t$ : Évapotranspiration culture au jour $t$ (mm), calculée comme $ETc_t = Kc_t \\cdot ET0_t \\cdot f_{ET}(\\psi_t)$
    - $D(S_t)$ : Drainage (perte d'eau par percolation) au jour $t$ (mm)
    
    La tension matricielle prédite par le modèle physique est ensuite obtenue via la courbe de rétention :
    
    $$
    \\psi_{t+1}^{\\text{phys}} = f_{\\text{retention}}(S_{t+1}^{\\text{phys}})
    $$
    
    où $f_{\\text{retention}}$ est la fonction de rétention d'eau du sol (relation $S \\leftrightarrow \\psi$).
    
    **b) Correction résiduelle par Neural ODE** :
    
    Le Neural ODE apprend une correction $\\Delta \\psi_t$ basée sur l'état actuel :
    
    $$
    \\Delta \\psi_t = f_\\theta(\\psi_t, I_t, R_t, ET0_t)
    $$
    
    où :
    - $\\psi_t$ : Tension matricielle actuelle (cbar)
    - $I_t$ : Irrigation appliquée (mm)
    - $R_t$ : Pluie (mm)
    - $ET0_t$ : Évapotranspiration de référence (mm/jour)
    - $f_\\theta$ : Réseau de neurones (MLP) paramétré par $\\theta$
    
    **c) Prédiction finale hybride** :
    
    La prédiction finale combine les deux composantes :
    
    $$
    \\psi_{t+1} = \\psi_{t+1}^{\\text{phys}} + \\Delta \\psi_t
    $$
    
    La réserve en eau corrigée est ensuite obtenue par inversion de la courbe de rétention :
    
    $$
    S_{t+1} = f_{\\text{retention}}^{-1}(\\psi_{t+1})
    $$
        """)
        
        with st.expander("🏗️ Architecture du réseau de neurones $f_\\theta$", expanded=False):
            st.markdown("""
            Le réseau de neurones $f_\\theta$ est un **MLP (Multi-Layer Perceptron)** avec :
            
            - **Couche d'entrée** : 4 neurones (pour $\\psi_t$, $I_t$, $R_t$, $ET0_t$)
            - **Couches cachées** : 2 couches de 64 neurones chacune avec activation $\\tanh$
            - **Couche de sortie** : 1 neurone (pour $\\Delta \\psi_t$)
            
            **Équations du réseau** :
            
            $$
            \\mathbf{h}_1 = \\tanh(\\mathbf{W}_1 \\mathbf{x} + \\mathbf{b}_1)
            $$
            
            $$
            \\mathbf{h}_2 = \\tanh(\\mathbf{W}_2 \\mathbf{h}_1 + \\mathbf{b}_2)
            $$
            
            $$
            \\Delta \\psi_t = \\mathbf{W}_3 \\mathbf{h}_2 + b_3
            $$
            
            où :
            - $\\mathbf{x} = [\\psi_t, I_t, R_t, ET0_t]^T$ est le vecteur d'entrée
            - $\\mathbf{W}_1, \\mathbf{W}_2, \\mathbf{W}_3$ sont les matrices de poids
            - $\\mathbf{b}_1, \\mathbf{b}_2, b_3$ sont les biais
            - $\\mathbf{h}_1, \\mathbf{h}_2$ sont les activations des couches cachées
            """)
    
    with st.expander("📚 Processus d'entraînement", expanded=False):
        st.markdown("""
        ### Processus d'entraînement
        
        Le Neural ODE est entraîné de manière **supervisée** sur des données de simulation ou réelles :
        
        **a) Génération des données d'entraînement** :
        
        Pour chaque pas de temps $t$ d'une simulation, on collecte :
        - **Entrées** : $X_t = [\\psi_t, I_t, R_t, ET0_t]$
        - **Cible** : $y_t = \\psi_{t+1}^{\\text{réel}} - \\psi_{t+1}^{\\text{phys}}$
        
        où $\\psi_{t+1}^{\\text{réel}}$ peut être :
        - Une mesure réelle de tension (si disponible)
        - Une simulation avec un modèle plus sophistiqué (HYDRUS, Aquacrop)
        - Une simulation physique avec biais artificiel pour tester la capacité de correction
        
        **b) Fonction de perte** :
        
        Le modèle est entraîné pour minimiser l'erreur quadratique moyenne :
        
        $$
        \\mathcal{L}(\\theta) = \\frac{1}{N} \\sum_{i=1}^{N} \\left( \\Delta \\psi_t^{(i)} - y_t^{(i)} \\right)^2
        $$
        
        où $N$ est le nombre d'échantillons d'entraînement.
        
        **c) Optimisation** :
        
        Les paramètres $\\theta$ sont optimisés via l'algorithme d'Adam avec un learning rate typiquement de $10^{-3}$ à $10^{-4}$.
        """)
    
    with st.expander("🔢 Méthode d'intégration : Discrétisation temporelle", expanded=False):
        st.markdown("""
        ### Méthode d'intégration : Approche discrète
        
        **Important** : Dans notre implémentation, le Neural ODE utilise une **approche discrète** plutôt qu'une intégration continue.
        
        #### Principe de discrétisation
        
        Le modèle prédit directement la correction résiduelle $\\Delta \\psi_t$ sur un **pas de temps fixe de 1 jour** :
        
        $$
        \\Delta \\psi_t = f_\\theta(\\psi_t, I_t, R_t, ET0_t)
        $$
        
        où $f_\\theta$ est un réseau de neurones (MLP) qui apprend directement la variation de tension sur un jour.
        
        #### Pourquoi une approche discrète ?
        
        - **Simplicité** : Pas besoin de solveurs d'ODE complexes (Euler, Runge-Kutta, etc.)
        - **Efficacité** : Un seul forward pass du réseau par pas de temps
        - **Adéquation au problème** : Les données météorologiques et les décisions d'irrigation sont disponibles à l'échelle journalière
        - **Stabilité** : Évite les problèmes numériques liés aux solveurs d'ODE adaptatifs
        
        #### Comparaison avec une approche continue
        
        Dans un Neural ODE "classique" continu, on modéliserait :
        
        $$
        \\frac{d\\psi(t)}{dt} = f_\\theta(\\psi(t), I(t), R(t), ET0(t))
        $$
        
        et on intégrerait cette équation différentielle avec un solveur (Euler, Runge-Kutta d'ordre 2 ou 4, etc.) :
        
        $$
        \\psi_{t+1} = \\psi_t + \\int_t^{t+1} f_\\theta(\\psi(\\tau), I(\\tau), R(\\tau), ET0(\\tau)) \\, d\\tau
        $$
        
        **Dans notre cas** : Le réseau apprend directement la solution discrétisée :
        
        $$
        \\Delta \\psi_t = \\psi_{t+1} - \\psi_t \\approx f_\\theta(\\psi_t, I_t, R_t, ET0_t)
        $$
        
        où l'intégration est implicite dans l'apprentissage du réseau.
        
        #### Implémentation dans le code
        
        Dans l'environnement RL (`utils_env_modeles.py`), l'inférence est très simple :
        
        ```python
        # Calcul direct de la correction (pas d'intégration)
        x = [psi_t, I_t, R_t, ET0_t]
        delta_psi = residual_ode(x)  # Un seul forward pass
        psi_next = psi_next_phys + delta_psi
        ```
        
        **Note** : Certaines variantes expérimentales dans les notebooks utilisent une méthode du trapèze (Runge-Kutta d'ordre 2) avec deux évaluations du réseau :
        
        $$
        k_1 = f_\\theta(\\psi_t, I_t, R_t, ET0_t)
        $$
        
        $$
        k_2 = f_\\theta(\\psi_t + 0.5 \\cdot k_1, I_t, R_t, ET0_t)
        $$
        
        $$
        \\Delta \\psi_t = \\frac{k_1 + k_2}{2}
        $$
        
        Cependant, l'implémentation principale dans l'environnement RL utilise la version simple à un seul pas.
        
        #### Avantages et limites
        
        **Avantages de l'approche discrète** :
        - ✅ **Rapidité** : Calcul instantané, pas de solveur itératif
        - ✅ **Simplicité** : Facile à implémenter et déboguer
        - ✅ **Stabilité** : Pas de problèmes de convergence des solveurs
        - ✅ **Adéquation** : Correspond à la granularité temporelle des données disponibles
        
        **Limites** :
        - ⚠️ **Pas de résolution infra-journalière** : Ne peut pas modéliser des phénomènes à l'échelle horaire
        - ⚠️ **Pas de pas de temps adaptatif** : Le pas de temps est fixe (1 jour)
        - ⚠️ **Moins précis pour des dynamiques rapides** : Si des changements importants se produisent en moins d'un jour, ils peuvent être mal capturés
        
        #### Alternative : Neural CDE (Scénario 4)
        
        Pour des besoins de modélisation plus sophistiqués avec dépendances temporelles, le projet implémente également un **Neural CDE** (Controlled Differential Equation) qui utilise un schéma d'Euler discretisé sur une séquence d'états passés :
        
        $$
        Z_{k+1} = Z_k + f_\\theta(Z_k, X_k) \\cdot \\Delta X_k
        $$
        
        où $\\Delta X_k = X_{k+1} - X_k$ et $X_k = [\\psi_k, I_k, R_k, ET0_k]$.
        
        Cette approche capture mieux les dépendances temporelles longues mais nécessite de maintenir un historique des états.
        """)
    
    with st.expander("🤖 Utilisation dans l'environnement RL", expanded=False):
        st.markdown("""
        ### Utilisation dans l'environnement RL
        
        Lors de l'exécution dans l'environnement Gymnasium pour l'apprentissage par renforcement :
        
        **a) Inférence** :
        
        À chaque pas de temps $t$ :
        1. Le modèle physique calcule $\\psi_{t+1}^{\\text{phys}}$ à partir de $S_t$, $I_t$, $R_t$, $ETc_t$, $D_t$
        2. Le Neural ODE calcule $\\Delta \\psi_t = f_\\theta(\\psi_t, I_t, R_t, ET0_t)$ (mode évaluation, sans gradient)
        3. La prédiction finale est $\\psi_{t+1} = \\psi_{t+1}^{\\text{phys}} + \\Delta \\psi_t$
        4. La réserve en eau est mise à jour : $S_{t+1} = f_{\\text{retention}}^{-1}(\\psi_{t+1})$
        
        **b) Avantages pour le RL** :
        
        - **Meilleure précision** : Le modèle hybride capture mieux la dynamique réelle du système
        - **Apprentissage plus efficace** : L'agent RL apprend sur un modèle plus fidèle à la réalité
        - **Robustesse** : Le modèle physique garantit des prédictions dans des plages physiquement réalistes
        - **Adaptabilité** : Le Neural ODE peut être ré-entraîné avec de nouvelles données pour s'adapter aux conditions locales
        """)
    
    with st.expander("📊 Variables et notations complètes", expanded=False):
        st.markdown("""
        **Variables d'état** :
        - $S_t$ : Réserve en eau du sol au jour $t$ (mm)
        - $\\psi_t$ : Tension matricielle de l'eau au jour $t$ (cbar)
        
        **Variables d'action** :
        - $I_t$ : Dose d'irrigation appliquée au jour $t$ (mm)
        
        **Variables météorologiques** :
        - $R_t$ : Pluie au jour $t$ (mm)
        - $ET0_t$ : Évapotranspiration de référence au jour $t$ (mm/jour)
        - $Kc_t$ : Coefficient cultural au jour $t$ (dimensionless)
        - $ETc_t = Kc_t \\cdot ET0_t \\cdot f_{ET}(\\psi_t)$ : Évapotranspiration culture (mm)
        
        **Variables de perte** :
        - $D(S_t)$ : Drainage (perte par percolation) au jour $t$ (mm)
        - $Q_t$ : Ruissellement au jour $t$ (mm, généralement négligé dans notre modèle)
        
        **Paramètres du sol** :
        - $\\eta_I$ : Efficacité d'irrigation (fraction, typiquement 0.8)
        - $S_{\\max}$ : Capacité maximale de stockage (mm)
        - $S_{fc}$ : Réserve à la capacité au champ (mm)
        - $S_{wp}$ : Réserve au point de flétrissement (mm)
        - $\\psi_{sat}$ : Tension à saturation (cbar, typiquement ~10 cbar)
        - $\\psi_{wp}$ : Tension au point de flétrissement (cbar, typiquement ~150 cbar)
        
        **Fonctions** :
        - $f_{\\text{retention}}(S)$ : Courbe de rétention (relation $S \\to \\psi$)
        - $f_{\\text{retention}}^{-1}(\\psi)$ : Inversion de la courbe de rétention (relation $\\psi \\to S$)
        - $f_{ET}(\\psi)$ : Fonction de réduction de l'évapotranspiration selon la tension
        - $f_\\theta(\\psi, I, R, ET0)$ : Réseau de neurones du Neural ODE
        
        **Corrections résiduelles** :
        - $\\Delta \\psi_t$ : Correction résiduelle apprise par le Neural ODE (cbar)
        - $\\psi_{t+1}^{\\text{phys}}$ : Prédiction du modèle physique (cbar)
        - $\\psi_{t+1}$ : Prédiction finale hybride (cbar)
        """)
    
    with st.expander("🔄 Différence avec Neural CDE", expanded=False):
        st.markdown("""
        Le projet implémente également un **Neural CDE** (Controlled Differential Equation) qui est une extension plus sophistiquée :
        
        - **Neural ODE** : Utilise uniquement l'état actuel $[\\psi_t, I_t, R_t, ET0_t]$ pour prédire $\\Delta \\psi_t$
        - **Neural CDE** : Utilise une séquence d'états passés $\\{X_{t-L+1}, \\ldots, X_t\\}$ où $X_k = [\\psi_k, I_k, R_k, ET0_k]$
        
        Le Neural CDE capture des dépendances temporelles plus longues mais nécessite de maintenir un historique des états.
        """)
    
    with st.expander("⚖️ Avantages et limitations", expanded=False):
        st.markdown("""
        ### ✅ Avantages du modèle hybride Neural ODE
        
        - **Précision améliorée** : Capture les biais systématiques du modèle physique
        - **Interprétabilité préservée** : Le modèle physique reste la base, la correction est additive
        - **Efficacité computationnelle** : Inférence rapide (un seul forward pass du réseau)
        - **Flexibilité** : Peut être ré-entraîné avec de nouvelles données
        - **Robustesse** : Le modèle physique garantit des prédictions dans des plages réalistes
        
        ### ⚠️ Limitations
        
        - **Données d'entraînement** : Nécessite des données pour apprendre la correction
        - **Généralisation** : Peut ne pas généraliser à des conditions très différentes de l'entraînement
        - **Complexité** : Ajoute une couche de complexité par rapport au modèle physique pur
        """)
    
    with st.expander("🔧 Paramètres recommandés et tuning", expanded=False):
        st.markdown("""
        ### Hyperparamètres du Neural ODE
        
        **Architecture du réseau** :
        - **Nombre de couches cachées** : 2-3 (recommandé)
        - **Dimension cachée** : 64-128
        - **Activation** : ReLU ou Tanh
        - **Dimension d'entrée** : 4 ($[\\psi_t, I_t, R_t, ET0_t]$)
        - **Dimension de sortie** : 1 ($\\Delta \\psi_t$)
        
        **Pré-entraînement** :
        - **Nombre de trajectoires** : 32-50 (recommandé)
        - **Nombre d'epochs** : 20-50 selon convergence
        - **Batch size** : 64-128
        - **Learning rate** : $10^{-3}$ (recommandé)
        - **Optimiseur** : Adam
        
        **Hyperparamètres PPO** :
        - Identiques au Scénario 2
        - Learning rate : $3 \\times 10^{-4}$
        - Gamma : 0.99
        - GAE lambda : 0.95
        
        ### Stratégie de tuning
        
        **1. Pré-entraînement du Neural ODE** :
        - Générer des trajectoires avec le modèle physique
        - Entraîner le Neural ODE à prédire $\\Delta \\psi$
        - Vérifier que la loss converge (< 0.01 idéalement)
        
        **2. Entraînement PPO** :
        - Utiliser le Neural ODE pré-entraîné
        - Entraîner PPO comme Scénario 2
        - Observer si la correction améliore les performances
        
        **3. Ajustements** :
        - Si correction trop faible : Augmenter la capacité du réseau
        - Si correction instable : Réduire learning rate, ajouter régularisation
        - Si pas d'amélioration : Vérifier qualité des données d'entraînement
        """)
    
    with st.expander("🧭 Quand utiliser le Scénario 3 ?", expanded=False):
        st.markdown("""
        ### ✅ Choisir le Scénario 3 si :
        
        - **Biais du modèle physique connu** :
          - Paramètres du sol mal calibrés
          - Phénomènes non modélisés (hétérogénéité, structure du sol)
          - Erreurs systématiques dans les prédictions
        
        - **Données réelles disponibles** :
          - Mesures de tension réelles pour entraîner la correction
          - Données historiques suffisantes
          - Qualité des données acceptable
        
        - **Besoin de précision améliorée** :
          - Scénario 2 ne suffit pas
          - Besoin de corriger les biais du modèle physique
          - Performance supérieure recherchée
        
        - **Interprétabilité importante** :
          - Besoin de séparer physique et correction
          - Analyse des biais du modèle physique
          - Compréhension des phénomènes non modélisés
        
        ### ❌ Ne pas choisir le Scénario 3 si :
        
        - **Modèle physique fiable** :
          - Pas de biais connus
          - Paramètres bien calibrés
          - Scénario 2 suffit
        
        - **Pas de données réelles** :
          - Impossible d'entraîner la correction
          - Seulement des simulations disponibles
          - → Préférer Scénario 2 ou 5
        
        - **Simplicité recherchée** :
          - Pas besoin de correction
          - Approche simple suffit
          - → Préférer Scénario 1 ou 2
        
        - **Dépendances temporelles longues** :
          - Besoin de mémoire temporelle
          - → Préférer Scénario 4 (Neural CDE)
        """)
    
    with st.expander("🛠️ Conseils pratiques", expanded=False):
        st.markdown("""
        ### Workflow recommandé
        
        **1. Validation du modèle physique** :
        - Comparer prédictions physiques avec données réelles
        - Identifier les biais systématiques
        - Quantifier l'erreur à corriger
        
        **2. Pré-entraînement Neural ODE** :
        - Générer trajectoires avec modèle physique
        - Calculer $\\Delta \\psi = \\psi_{\\text{réel}} - \\psi_{\\text{physique}}$
        - Entraîner Neural ODE à prédire $\\Delta \\psi$
        - Vérifier convergence (loss < 0.01)
        
        **3. Entraînement PPO** :
        - Intégrer Neural ODE dans l'environnement
        - Entraîner PPO comme Scénario 2
        - Comparer performances avec Scénario 2
        
        **4. Analyse** :
        - Analyser la correction apprise
        - Identifier quels phénomènes sont corrigés
        - Vérifier la cohérence physique
        
        ### Troubleshooting
        
        **Problème : Correction trop faible**
        - **Symptôme** : $\\Delta \\psi$ toujours proche de 0
        - **Solutions** :
          - Augmenter capacité du réseau (plus de couches/neurones)
          - Vérifier qualité des données d'entraînement
          - Augmenter nombre d'epochs
        
        **Problème : Correction instable**
        - **Symptôme** : $\\Delta \\psi$ oscille, prédictions erratiques
        - **Solutions** :
          - Réduire learning rate
          - Ajouter régularisation (L2, dropout)
          - Réduire capacité du réseau
        
        **Problème : Pas d'amélioration vs Scénario 2**
        - **Symptôme** : Performance similaire ou pire
        - **Solutions** :
          - Vérifier que le Neural ODE est bien pré-entraîné
          - Vérifier qualité des données
          - Augmenter nombre de trajectoires d'entraînement
        """)
    
    with st.expander("🔗 Comparaison avec les autres scénarios", expanded=False):
        st.markdown("""
        ### Scénario 3 vs Scénario 2 (RL basique)
        
        **Scénario 2** :
        - RL direct sur modèle physique
        - Pas de correction du modèle
        
        **Scénario 3** :
        - Correction résiduelle du modèle physique
        - Améliore la prédiction
        
        **Quand choisir Scénario 3** : Biais du modèle physique connus
        
        ### Scénario 3 vs Scénario 4 (Neural CDE)
        
        **Scénario 3** :
        - Neural ODE : Pas de mémoire temporelle
        - Correction locale basée sur état actuel
        
        **Scénario 4** :
        - Neural CDE : Mémoire temporelle
        - Correction basée sur historique
        
        **Quand choisir Scénario 4** : Besoin de dépendances temporelles
        
        ### Scénario 3 vs Scénarios 5-6
        
        **Scénario 3** :
        - Correction du modèle physique
        - Améliore prédiction
        
        **Scénarios 5-6** :
        - Enrichissement observation (5) ou planification (6)
        - Rôle différent (pas de correction physique)
        
        **Complémentarité** : Peut être combiné avec Scénario 5
        """)
    
    # ========================================================================
    # ONGLET DOCUMENTATION 5 : NEURAL CDE
    # ========================================================================

def render_doc_neural_cde():
    """
    Affiche le contenu de l'onglet de documentation : Neural CDE.
    """
    st.markdown('<h2 class="section-header">🌀 Neural CDE : Modèle hybride avec dépendances temporelles</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### ❓ Qu'est-ce qu'un Neural CDE ?
    
    Un **Neural CDE** (Neural Controlled Differential Equation) est une extension des Neural ODE qui utilise une séquence d'états passés pour capturer des dépendances temporelles plus longues. 
    Dans notre contexte, il s'agit d'un **modèle résiduel** qui apprend à corriger les prédictions d'un modèle physique en exploitant l'historique des états.
    """)
    
    with st.expander("🔬 Principe général des Neural CDE", expanded=False):
        st.markdown("""
        Les Neural CDE modélisent la dynamique d'un système par une équation différentielle contrôlée où la dérivée dépend d'une séquence d'états passés :
        
        $$
        dZ_t = f_\\theta(Z_t, X_t) \\, dX_t
        $$
        
        où :
        - $Z_t$ est l'état latent du système au temps $t$
        - $X_t$ est le processus de contrôle (séquence d'observations)
        - $f_\\theta$ est un réseau de neurones paramétré par $\\theta$
        - L'intégration se fait le long du chemin de $X_t$
        
        **Différence avec Neural ODE** :
        - **Neural ODE** : $\\frac{dZ(t)}{dt} = f_\\theta(Z(t), t)$ dépend uniquement de l'état actuel
        - **Neural CDE** : $dZ_t = f_\\theta(Z_t, X_t) \\, dX_t$ dépend de la séquence complète $\\{X_s : s \\leq t\\}$
        
        **Avantages** :
        - **Dépendances temporelles** : Capture des effets à long terme et des dynamiques complexes
        - **Robustesse** : Gère mieux les données irrégulières ou manquantes
        - **Flexibilité** : Peut modéliser des processus avec mémoire longue
        """)
    
    with st.expander("🎯 Application dans notre projet : Modèle hybride avec mémoire", expanded=False):
        st.markdown("""
        Dans notre projet d'irrigation intelligente, le Neural CDE est utilisé comme **correction résiduelle** sur le modèle physique FAO, 
        mais avec une **mémoire temporelle** pour capturer des effets à long terme :
        
        - **Modèle physique** : Fournit une prédiction de base basée sur les lois de la physique (bilan hydrique)
        - **Neural CDE** : Apprend les écarts systématiques en utilisant une séquence d'états passés
        
        Cette approche hybride permet de :
        - ✅ Bénéficier de la robustesse et de l'interprétabilité du modèle physique
        - ✅ Capturer les biais systématiques et les phénomènes complexes non modélisés
        - ✅ Modéliser des dépendances temporelles longues (effets cumulatifs de l'irrigation, sécheresses prolongées, etc.)
        - ✅ S'adapter aux spécificités locales avec une meilleure compréhension de l'historique
        
        ### 📋 Paramètres de configuration du pré-entraînement
        
        **Nombre de trajectoires** :
        - **Signification** : Nombre de simulations indépendantes utilisées pour générer les données d'entraînement du Neural CDE
        - **Valeur usuelle** : 32 trajectoires (par défaut)
        - **Impact** : Plus de trajectoires = plus de diversité dans les séquences temporelles
        - **Tuning** : Augmenter (50-100) pour capturer plus de variabilité, réduire (10-20) pour accélérer
        - **Note** : Important pour avoir suffisamment de séquences variées pour l'apprentissage séquentiel
        
        **Nombre d'epochs** :
        - **Signification** : Nombre de passages complets sur l'ensemble des données d'entraînement
        - **Valeur usuelle** : 10 epochs (par défaut)
        - **Impact** : Plus d'epochs = meilleur apprentissage des dépendances temporelles
        - **Tuning** : Augmenter (20-50) si le modèle continue à apprendre, réduire si surapprentissage
        - **Note** : Les modèles séquentiels peuvent nécessiter plus d'epochs que les modèles sans mémoire
        
        **Taille des batches** :
        - **Signification** : Nombre de séquences traitées simultanément lors de chaque mise à jour
        - **Valeur usuelle** : 256 (par défaut)
        - **Impact** : Batch plus grand = gradients plus stables pour les séquences
        - **Tuning** : Réduire (32-128) si mémoire limitée, augmenter (512+) pour plus de stabilité
        - **Note** : Doit tenir compte de la longueur de séquence (plus de mémoire requise)
        
        **Taux d'apprentissage (Learning Rate)** :
        - **Signification** : Vitesse d'ajustement des paramètres lors de l'optimisation
        - **Valeur usuelle** : $10^{-3}$ (0.001) (par défaut)
        - **Impact** : LR trop élevé = instabilité dans l'apprentissage séquentiel
        - **Tuning** : Réduire (10^{-4} - 10^{-5}) si la loss oscille, augmenter (10^{-2}) si trop lent
        - **Note** : Les modèles séquentiels sont souvent plus sensibles au LR que les modèles sans mémoire
        
        **Longueur de séquence** :
        - **Signification** : Nombre de pas de temps dans l'historique utilisé par le Neural CDE
        - **Valeur usuelle** : 5 pas de temps (par défaut)
        - **Impact** : Séquence plus longue = capture de dépendances plus longues mais plus de complexité
        - **Tuning** : Augmenter (7-10) pour effets cumulatifs longs, réduire (3-4) pour dépendances courtes
        - **Note** : Doit être adapté à la nature des dépendances temporelles du problème (effets d'irrigation sur plusieurs jours)
        """)
    
    with st.expander("📐 Architecture du modèle hybride Neural CDE", expanded=False):
        st.markdown("""
        ### Architecture du modèle hybride Neural CDE
    
    Le modèle hybride combine deux composantes, avec une approche séquentielle pour la correction :
    
    **a) Prédiction physique** :
    
    Le modèle physique calcule d'abord la réserve en eau $S_{t+1}^{\\text{phys}}$ selon le bilan hydrique :
    
    $$
    S_{t+1}^{\\text{phys}} = S_t + \\eta_I I_t + R_t - ETc_t - D(S_t)
    $$
    
    où :
    - $S_t$ : Réserve en eau au jour $t$ (mm)
    - $\\eta_I$ : Efficacité d'irrigation (fraction de l'eau d'irrigation effectivement disponible)
    - $I_t$ : Dose d'irrigation appliquée au jour $t$ (mm)
    - $R_t$ : Pluie au jour $t$ (mm)
    - $ETc_t$ : Évapotranspiration culture au jour $t$ (mm), calculée comme $ETc_t = Kc_t \\cdot ET0_t \\cdot f_{ET}(\\psi_t)$
    - $D(S_t)$ : Drainage (perte d'eau par percolation) au jour $t$ (mm)
    
    La tension matricielle prédite par le modèle physique est ensuite obtenue via la courbe de rétention :
    
    $$
    \\psi_{t+1}^{\\text{phys}} = f_{\\text{retention}}(S_{t+1}^{\\text{phys}})
    $$
    
    où $f_{\\text{retention}}$ est la fonction de rétention d'eau du sol (relation $S \\leftrightarrow \\psi$).
    
    **b) Correction résiduelle par Neural CDE** :
    
    Le Neural CDE apprend une correction $\\Delta \\psi_t$ basée sur une **séquence d'états passés** :
    
    $$
    X_k = [\\psi_k, I_k, R_k, ET0_k] \\quad \\text{pour } k \\in \\{t-L+1, \\ldots, t\\}
    $$
    
    où $L$ est la longueur de la séquence (typiquement $L = 5$ jours).
    
    Le modèle utilise un schéma d'Euler discretisé pour intégrer l'équation différentielle contrôlée :
    
    $$
    Z_{k+1} = Z_k + f_\\theta(Z_k, X_k) \\cdot \\Delta X_k
    $$
    
    où :
    - $Z_k$ est l'état latent au pas $k$
    - $\\Delta X_k = X_{k+1} - X_k$ est l'incrément du processus de contrôle
    - $f_\\theta$ est un réseau de neurones (MLP) paramétré par $\\theta$
    
    La correction finale est obtenue à partir de l'état latent final :
    
    $$
    \\Delta \\psi_t = g_\\phi(Z_t)
    $$
    
    où $g_\\phi$ est une couche de sortie qui mappe l'état latent vers la correction résiduelle.
    
    **c) Prédiction finale hybride** :
    
    La prédiction finale combine les deux composantes :
    
    $$
    \\psi_{t+1} = \\psi_{t+1}^{\\text{phys}} + \\Delta \\psi_t
    $$
    
    La réserve en eau corrigée est ensuite obtenue par inversion de la courbe de rétention :
    
    $$
    S_{t+1} = f_{\\text{retention}}^{-1}(\\psi_{t+1})
    $$
        """)
        
        with st.expander("🏗️ Architecture du réseau de neurones $f_\\theta$ et $g_\\phi$", expanded=False):
            st.markdown("""
            Le réseau de neurones $f_\\theta$ est un **MLP (Multi-Layer Perceptron)** avec :
            
            - **Couche d'entrée** : $\\dim(Z_k) + 4$ neurones (pour l'état latent $Z_k$ et $X_k = [\\psi_k, I_k, R_k, ET0_k]$)
            - **Couches cachées** : 2 couches de 64 neurones chacune avec activation $\\tanh$
            - **Couche de sortie** : $\\dim(Z_k)$ neurones (pour la mise à jour de l'état latent)
            
            La couche de sortie $g_\\phi$ est un MLP simple :
            
            - **Couche d'entrée** : $\\dim(Z_t)$ neurones (pour l'état latent final $Z_t$)
            - **Couche cachée** : 1 couche de 32 neurones avec activation $\\tanh$
            - **Couche de sortie** : 1 neurone (pour $\\Delta \\psi_t$)
            
            **Équations du réseau** :
            
            Pour $f_\\theta$ :
            $$
            \\mathbf{h}_1 = \\tanh(\\mathbf{W}_1 [Z_k, X_k] + \\mathbf{b}_1)
            $$
            
            $$
            \\mathbf{h}_2 = \\tanh(\\mathbf{W}_2 \\mathbf{h}_1 + \\mathbf{b}_2)
            $$
            
            $$
            \\Delta Z_k = \\mathbf{W}_3 \\mathbf{h}_2 + \\mathbf{b}_3
            $$
            
            Pour $g_\\phi$ :
            $$
            \\mathbf{h}_{out} = \\tanh(\\mathbf{W}_{out} Z_t + \\mathbf{b}_{out})
            $$
            
            $$
            \\Delta \\psi_t = \\mathbf{W}_{final} \\mathbf{h}_{out} + b_{final}
            $$
            
            où :
            - $[Z_k, X_k]$ est la concaténation de l'état latent et de l'observation
            - $\\mathbf{W}_1, \\mathbf{W}_2, \\mathbf{W}_3, \\mathbf{W}_{out}, \\mathbf{W}_{final}$ sont les matrices de poids
            - $\\mathbf{b}_1, \\mathbf{b}_2, \\mathbf{b}_3, \\mathbf{b}_{out}, b_{final}$ sont les biais
            """)
    
    with st.expander("📚 Processus d'entraînement", expanded=False):
        st.markdown("""
        ### Processus d'entraînement
        
        Le Neural CDE est entraîné de manière **supervisée** sur des séquences de données de simulation ou réelles :
        
        **a) Génération des données d'entraînement** :
        
        Pour chaque pas de temps $t$ d'une simulation, on collecte :
        - **Entrées** : Séquence $\\{X_{t-L+1}, \\ldots, X_t\\}$ où $X_k = [\\psi_k, I_k, R_k, ET0_k]$
        - **Cible** : $y_t = \\psi_{t+1}^{\\text{réel}} - \\psi_{t+1}^{\\text{phys}}$
        
        où $\\psi_{t+1}^{\\text{réel}}$ peut être :
        - Une mesure réelle de tension (si disponible)
        - Une simulation avec un modèle plus sophistiqué (HYDRUS, Aquacrop)
        - Une simulation physique avec biais artificiel pour tester la capacité de correction
        
        **b) Fonction de perte** :
        
        Le modèle est entraîné pour minimiser l'erreur quadratique moyenne sur les séquences :
        
        $$
        \\mathcal{L}(\\theta, \\phi) = \\frac{1}{N} \\sum_{i=1}^{N} \\left( \\Delta \\psi_t^{(i)} - y_t^{(i)} \\right)^2
        $$
        
        où $N$ est le nombre d'échantillons d'entraînement (séquences).
        
        **c) Optimisation** :
        
        Les paramètres $\\theta$ et $\\phi$ sont optimisés via l'algorithme d'Adam avec un learning rate typiquement de $10^{-3}$ à $10^{-4}$.
        L'entraînement se fait par mini-batches de séquences.
        """)
    
    with st.expander("🔢 Méthode d'intégration : Schéma d'Euler discretisé", expanded=False):
        st.markdown("""
        ### Méthode d'intégration : Approche discrète avec séquence
        
        **Important** : Dans notre implémentation, le Neural CDE utilise un **schéma d'Euler discretisé** sur une séquence d'états passés.
        
        #### Principe d'intégration séquentielle
        
        Le modèle intègre l'équation différentielle contrôlée sur une séquence de $L$ pas de temps :
        
        $$
        Z_{k+1} = Z_k + f_\\theta(Z_k, X_k) \\cdot \\Delta X_k \\quad \\text{pour } k \\in \\{t-L+1, \\ldots, t-1\\}
        $$
        
        où :
        - $Z_{t-L+1} = 0$ (état latent initialisé à zéro)
        - $\\Delta X_k = X_{k+1} - X_k$ est l'incrément du processus de contrôle
        - $f_\\theta(Z_k, X_k)$ est évalué à chaque pas $k$
        
        La correction finale est obtenue à partir de l'état latent final :
        
        $$
        \\Delta \\psi_t = g_\\phi(Z_t)
        $$
        
        #### Pourquoi une approche séquentielle ?
        
        - **Dépendances temporelles** : Capture les effets cumulatifs et les dynamiques à long terme
        - **Mémoire** : Maintient un historique des états pour mieux prédire les corrections
        - **Robustesse** : Gère mieux les variations temporelles complexes
        - **Adéquation au problème** : Les effets de l'irrigation et de la météo peuvent avoir des impacts sur plusieurs jours
        
        #### Comparaison avec Neural ODE
        
        **Neural ODE** :
        - Utilise uniquement l'état actuel : $\\Delta \\psi_t = f_\\theta(\\psi_t, I_t, R_t, ET0_t)$
        - Pas de mémoire temporelle
        - Plus simple et plus rapide
        
        **Neural CDE** :
        - Utilise une séquence d'états : $\\{X_{t-L+1}, \\ldots, X_t\\}$
        - Capture des dépendances temporelles longues
        - Plus complexe mais plus expressif
        
        #### Implémentation dans le code
        
        Dans l'environnement RL (`utils_env_modeles.py`), l'inférence se fait séquentiellement :
        
        ```python
        # Initialisation
        Z = torch.zeros(hidden_dim)
        X_prev = X_sequence[0]  # Premier état de la séquence
        
        # Intégration séquentielle
        for k in range(1, seq_len):
            X_k = X_sequence[k]
            dX = X_k - X_prev
            dZ = f_theta(Z, X_k) * dX
            Z = Z + dZ
            X_prev = X_k
        
        # Correction finale
        delta_psi = g_phi(Z)
        psi_next = psi_next_phys + delta_psi
        ```
        
        #### Avantages et limites
        
        **Avantages de l'approche séquentielle** :
        - ✅ **Dépendances temporelles** : Capture des effets à long terme
        - ✅ **Mémoire** : Exploite l'historique des états
        - ✅ **Expressivité** : Modélise des dynamiques complexes
        - ✅ **Robustesse** : Gère mieux les variations temporelles
        
        **Limites** :
        - ⚠️ **Complexité computationnelle** : Nécessite $L$ évaluations du réseau par prédiction
        - ⚠️ **Mémoire** : Doit maintenir un historique de $L$ états
        - ⚠️ **Temps d'entraînement** : Plus long que Neural ODE
        - ⚠️ **Hyperparamètres** : Nécessite de choisir $L$ (longueur de séquence)
        """)
    
    with st.expander("🤖 Utilisation dans l'environnement RL", expanded=False):
        st.markdown("""
        ### Utilisation dans l'environnement RL
        
        Lors de l'exécution dans l'environnement Gymnasium pour l'apprentissage par renforcement :
        
        **a) Inférence** :
        
        À chaque pas de temps $t$ :
        1. Le modèle physique calcule $\\psi_{t+1}^{\\text{phys}}$ à partir de $S_t$, $I_t$, $R_t$, $ETc_t$, $D_t$
        2. Le Neural CDE utilise la séquence $\\{X_{t-L+1}, \\ldots, X_t\\}$ pour calculer $\\Delta \\psi_t$ (mode évaluation, sans gradient)
        3. La prédiction finale est $\\psi_{t+1} = \\psi_{t+1}^{\\text{phys}} + \\Delta \\psi_t$
        4. La réserve en eau est mise à jour : $S_{t+1} = f_{\\text{retention}}^{-1}(\\psi_{t+1})$
        5. L'historique est mis à jour : $X_{t+1} = [\\psi_{t+1}, I_{t+1}, R_{t+1}, ET0_{t+1}]$
        
        **b) Avantages pour le RL** :
        
        - **Meilleure précision** : Le modèle hybride capture mieux la dynamique réelle avec mémoire temporelle
        - **Apprentissage plus efficace** : L'agent RL apprend sur un modèle plus fidèle à la réalité
        - **Robustesse** : Le modèle physique garantit des prédictions dans des plages physiquement réalistes
        - **Adaptabilité** : Le Neural CDE peut être ré-entraîné avec de nouvelles données pour s'adapter aux conditions locales
        - **Dépendances temporelles** : Capture mieux les effets cumulatifs de l'irrigation et de la météo
        """)
    
    with st.expander("📊 Variables et notations complètes", expanded=False):
        st.markdown("""
        **Variables d'état** :
        - $S_t$ : Réserve en eau du sol au jour $t$ (mm)
        - $\\psi_t$ : Tension matricielle de l'eau au jour $t$ (cbar)
        - $Z_t$ : État latent du Neural CDE au jour $t$ (vecteur de dimension $d$)
        
        **Variables d'action** :
        - $I_t$ : Dose d'irrigation appliquée au jour $t$ (mm)
        
        **Variables météorologiques** :
        - $R_t$ : Pluie au jour $t$ (mm)
        - $ET0_t$ : Évapotranspiration de référence au jour $t$ (mm/jour)
        - $Kc_t$ : Coefficient cultural au jour $t$ (dimensionless)
        - $ETc_t = Kc_t \\cdot ET0_t \\cdot f_{ET}(\\psi_t)$ : Évapotranspiration culture (mm)
        
        **Variables de perte** :
        - $D(S_t)$ : Drainage (perte par percolation) au jour $t$ (mm)
        - $Q_t$ : Ruissellement au jour $t$ (mm, généralement négligé dans notre modèle)
        
        **Paramètres du sol** :
        - $\\eta_I$ : Efficacité d'irrigation (fraction, typiquement 0.8)
        - $S_{\\max}$ : Capacité maximale de stockage (mm)
        - $S_{fc}$ : Réserve à la capacité au champ (mm)
        - $S_{wp}$ : Réserve au point de flétrissement (mm)
        - $\\psi_{sat}$ : Tension à saturation (cbar, typiquement ~10 cbar)
        - $\\psi_{wp}$ : Tension au point de flétrissement (cbar, typiquement ~150 cbar)
        
        **Paramètres du Neural CDE** :
        - $L$ : Longueur de la séquence d'états (typiquement $L = 5$ jours)
        - $d$ : Dimension de l'état latent $Z_t$ (typiquement $d = 32$)
        
        **Fonctions** :
        - $f_{\\text{retention}}(S)$ : Courbe de rétention (relation $S \\to \\psi$)
        - $f_{\\text{retention}}^{-1}(\\psi)$ : Inversion de la courbe de rétention (relation $\\psi \\to S$)
        - $f_{ET}(\\psi)$ : Fonction de réduction de l'évapotranspiration selon la tension
        - $f_\\theta(Z, X)$ : Réseau de neurones du Neural CDE pour la dynamique de l'état latent
        - $g_\\phi(Z)$ : Réseau de neurones du Neural CDE pour la correction résiduelle
        
        **Séquences** :
        - $X_k = [\\psi_k, I_k, R_k, ET0_k]$ : Vecteur d'observation au pas $k$
        - $\\{X_{t-L+1}, \\ldots, X_t\\}$ : Séquence d'observations utilisée pour la prédiction
        - $\\Delta X_k = X_{k+1} - X_k$ : Incrément du processus de contrôle
        
        **Corrections résiduelles** :
        - $\\Delta \\psi_t$ : Correction résiduelle apprise par le Neural CDE (cbar)
        - $\\psi_{t+1}^{\\text{phys}}$ : Prédiction du modèle physique (cbar)
        - $\\psi_{t+1}$ : Prédiction finale hybride (cbar)
        """)
    
    with st.expander("🔄 Différence avec Neural ODE", expanded=False):
        st.markdown("""
        **Neural ODE** :
        - Utilise uniquement l'état actuel : $\\Delta \\psi_t = f_\\theta(\\psi_t, I_t, R_t, ET0_t)$
        - Pas de mémoire temporelle
        - Plus simple et plus rapide
        - Adéquat pour des dynamiques à court terme
        
        **Neural CDE** :
        - Utilise une séquence d'états : $\\{X_{t-L+1}, \\ldots, X_t\\}$ où $X_k = [\\psi_k, I_k, R_k, ET0_k]$
        - Capture des dépendances temporelles longues
        - Plus complexe mais plus expressif
        - Adéquat pour des dynamiques à long terme et des effets cumulatifs
        
        **Quand utiliser Neural CDE ?**
        - Effets cumulatifs de l'irrigation sur plusieurs jours
        - Sécheresses prolongées avec impacts retardés
        - Dynamiques complexes nécessitant une mémoire temporelle
        - Données avec dépendances temporelles importantes
        """)
    
    with st.expander("🔬 Relation avec les Physics-Informed Neural Networks (PINN)", expanded=False):
        st.markdown("""
        ### Neural CDE comme modèle Physics-Informed
        
        Le Neural CDE utilisé dans ce projet peut être considéré comme un **modèle hybride physics-informed avec mémoire temporelle** :
        
        **Définition des PINN** :
        Les Physics-Informed Neural Networks (PINN) sont des réseaux de neurones qui intègrent explicitement les lois de la physique dans leur architecture ou leur fonction de perte.
        
        **Notre approche hybride** :
        - **Modèle physique** : Fournit la structure et les contraintes physiques (bilan hydrique FAO)
        - **Neural CDE** : Apprend une correction résiduelle basée sur une séquence d'états passés, respectant la structure physique
        
        **Caractéristiques physics-informed** :
        - ✅ **Intégration explicite de la physique** : Le modèle physique FAO est intégré directement dans l'architecture
        - ✅ **Respect des contraintes physiques** : La correction $\\Delta \\psi$ est appliquée de manière cohérente avec le modèle physique
        - ✅ **Mémoire temporelle guidée par la physique** : Le Neural CDE utilise l'historique des états physiques pour produire une correction cohérente
        - ✅ **Apprentissage guidé par la physique** : Le modèle apprend à partir de données mais dans le contexte d'un modèle physique avec dépendances temporelles
        
        **Avantage par rapport aux PINN classiques** :
        - **PINN classiques** : Généralement sans mémoire, modélisent des processus instantanés
        - **Notre Neural CDE** : Capture des dépendances temporelles longues tout en respectant la physique, permettant de modéliser des effets cumulatifs (séchage progressif, pluies répétées, etc.)
        
        **Conclusion** :
        Notre modèle Neural CDE est un **modèle hybride physics-informed avec mémoire temporelle** qui combine :
        - La robustesse et l'interprétabilité du modèle physique
        - La flexibilité d'apprentissage des réseaux de neurones
        - La capacité de capturer des dépendances temporelles longues
        """)
    
    with st.expander("⚖️ Avantages et limitations", expanded=False):
        st.markdown("""
        ### ✅ Avantages du modèle hybride Neural CDE
        
        - **Précision améliorée** : Capture les biais systématiques du modèle physique avec mémoire temporelle
        - **Dépendances temporelles** : Modélise des effets à long terme et des dynamiques complexes
        - **Interprétabilité préservée** : Le modèle physique reste la base, la correction est additive
        - **Efficacité computationnelle** : Inférence rapide (séquentielle mais parallélisable)
        - **Flexibilité** : Peut être ré-entraîné avec de nouvelles données
        - **Robustesse** : Le modèle physique garantit des prédictions dans des plages réalistes
        - **Mémoire** : Exploite l'historique des états pour de meilleures prédictions
        
        ### ⚠️ Limitations
        
        - **Données d'entraînement** : Nécessite des données séquentielles pour apprendre la correction
        - **Complexité computationnelle** : Nécessite $L$ évaluations du réseau par prédiction
        - **Mémoire** : Doit maintenir un historique de $L$ états
        - **Hyperparamètres** : Nécessite de choisir $L$ (longueur de séquence) et $d$ (dimension de l'état latent)
        - **Généralisation** : Peut ne pas généraliser à des conditions très différentes de l'entraînement
        - **Temps d'entraînement** : Plus long que Neural ODE
        """)
    
    with st.expander("🔧 Paramètres recommandés et tuning", expanded=False):
        st.markdown("""
        ### Hyperparamètres du Neural CDE
        
        **Longueur de séquence** ($L$) : 5-10 (recommandé)
        - Trop court (< 5) : Pas assez de mémoire temporelle
        - Trop long (> 15) : Complexité computationnelle élevée, risque de surapprentissage
        - Tuning : Commencer à 8, ajuster selon résultats
        
        **Dimension de l'état latent** ($d$) : 16-32 (recommandé)
        - Plus grande : Plus de capacité mais plus complexe
        - Plus petite : Plus simple mais moins de capacité
        - Tuning : Commencer à 16, augmenter si nécessaire
        
        **Architecture du réseau** :
        - **Nombre de couches cachées** : 2-3
        - **Dimension cachée** : 64-128
        - **Activation** : ReLU ou Tanh
        
        **Pré-entraînement** :
        - **Nombre de trajectoires** : 50-100 (plus que Neural ODE)
        - **Nombre d'epochs** : 30-60 selon convergence
        - **Batch size** : 32-64 (plus petit que Neural ODE)
        - **Learning rate** : $10^{-3}$ (recommandé)
        
        **Hyperparamètres PPO** :
        - Identiques au Scénario 2
        - Learning rate : $3 \\times 10^{-4}$
        - Gamma : 0.99
        
        ### Stratégie de tuning
        
        **1. Choix de $L$** :
        - Commencer avec $L = 8$
        - Augmenter si besoin de mémoire plus longue
        - Réduire si complexité trop élevée
        
        **2. Pré-entraînement** :
        - Plus de trajectoires que Neural ODE (mémoire nécessite plus de données)
        - Vérifier convergence de la loss
        - Analyser la qualité de la correction
        
        **3. Entraînement PPO** :
        - Comme Scénario 3
        - Comparer avec Scénario 3 pour évaluer gain
        """)
    
    with st.expander("🧭 Quand utiliser le Scénario 4 ?", expanded=False):
        st.markdown("""
        ### ✅ Choisir le Scénario 4 si :
        
        - **Dépendances temporelles importantes** :
          - Effets à long terme des décisions d'irrigation
          - Dynamiques complexes nécessitant mémoire
          - Scénario 3 (Neural ODE) insuffisant
        
        - **Données séquentielles disponibles** :
          - Historique de mesures disponible
          - Données temporelles de qualité
          - Séries temporelles complètes
        
        - **Biais temporels du modèle** :
          - Modèle physique ne capture pas bien les effets retardés
          - Nécessité de corriger avec mémoire temporelle
          - Phénomènes avec inertie (drainage, remontée capillaire)
        
        - **Performance maximale recherchée** :
          - Scénario 3 ne suffit pas
          - Besoin de meilleure précision
          - Ressources computationnelles disponibles
        
        ### ❌ Ne pas choisir le Scénario 4 si :
        
        - **Pas de dépendances temporelles** :
          - Effets locaux uniquement
          - Scénario 3 (Neural ODE) suffit
          - Pas besoin de mémoire
        
        - **Ressources limitées** :
          - Complexité computationnelle trop élevée
          - Temps d'entraînement trop long
          - Mémoire insuffisante
        
        - **Données insuffisantes** :
          - Pas assez de données séquentielles
          - Qualité des données insuffisante
          - → Préférer Scénario 3
        
        - **Simplicité recherchée** :
          - Approche simple suffit
          - → Préférer Scénarios 1-3
        """)
    
    with st.expander("🛠️ Conseils pratiques", expanded=False):
        st.markdown("""
        ### Workflow recommandé
        
        **1. Évaluer besoin de mémoire** :
        - Analyser si Scénario 3 suffit
        - Identifier dépendances temporelles
        - Quantifier gain potentiel
        
        **2. Choix de $L$** :
        - Commencer avec $L = 8$
        - Tester avec $L = 5, 10, 12$
        - Choisir selon performance/complexité
        
        **3. Pré-entraînement** :
        - Générer plus de trajectoires que Neural ODE
        - Entraîner Neural CDE
        - Vérifier qualité de correction
        
        **4. Entraînement PPO** :
        - Comme Scénario 3
        - Comparer performances
        
        ### Troubleshooting
        
        **Problème : Complexité computationnelle trop élevée**
        - **Symptôme** : Entraînement très lent
        - **Solutions** :
          - Réduire $L$ (ex: 8 → 5)
          - Réduire dimension latente $d$
          - Réduire batch size
        
        **Problème : Pas d'amélioration vs Scénario 3**
        - **Symptôme** : Performance similaire
        - **Solutions** :
          - Vérifier que $L$ est adapté
          - Augmenter nombre de trajectoires
          - Vérifier qualité des données séquentielles
        
        **Problème : Surapprentissage**
        - **Symptôme** : Bonne performance entraînement, mauvaise généralisation
        - **Solutions** :
          - Réduire $L$
          - Ajouter régularisation
          - Augmenter nombre de trajectoires d'entraînement
        """)
    
    with st.expander("🔗 Comparaison avec les autres scénarios", expanded=False):
        st.markdown("""
        ### Scénario 4 vs Scénario 3 (Neural ODE)
        
        **Scénario 3** :
        - Neural ODE : Pas de mémoire
        - Correction locale
        
        **Scénario 4** :
        - Neural CDE : Mémoire temporelle
        - Correction avec historique
        
        **Quand choisir Scénario 4** : Besoin de dépendances temporelles
        
        ### Scénario 4 vs Scénario 5 (PatchTST)
        
        **Scénario 4** :
        - Correction du modèle physique
        - Mémoire courte (5-10 pas)
        
        **Scénario 5** :
        - Enrichissement observation
        - Mémoire longue (30+ pas)
        
        **Différence** : Scénario 4 corrige physique, Scénario 5 enrichit observation
        
        ### Scénario 4 vs Scénarios 1-2
        
        **Scénario 4** :
        - Correction résiduelle avec mémoire
        - Plus complexe
        
        **Scénarios 1-2** :
        - Pas de correction
        - Plus simple
        
        **Quand choisir Scénario 4** : Biais temporels du modèle physique
        """)
    
    # ========================================================================
    # ONGLET DOCUMENTATION 6 : PATCHTST
    # ========================================================================

def render_doc_patchtst():
    """
    Affiche le contenu de l'onglet de documentation : PatchTST.
    """
    st.markdown('<h2 class="section-header">🔮 PatchTST : Extracteur de features temporelles pour l\'apprentissage par renforcement</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### ❓ Qu'est-ce que PatchTST ?
    
    **PatchTST** (Patch-based Time Series Transformer) est un modèle Transformer spécialement conçu pour la prévision de séries temporelles. 
    Contrairement aux approches traditionnelles qui traitent chaque point de temps individuellement, PatchTST segmente la série temporelle en **"patches"** 
    (segments temporels) pour capturer des motifs à différentes échelles temporelles.
    
    Dans notre contexte d'irrigation intelligente, PatchTST est utilisé comme **extracteur de features temporelles** pour enrichir l'observation de l'agent RL, 
    lui permettant de mieux comprendre les dynamiques temporelles longues (tendances, saisonnalité, motifs récurrents).
    """)
    
    with st.expander("🔬 Principe général de PatchTST", expanded=False):
        st.markdown("""
        PatchTST transforme une série temporelle en une séquence de patches qui sont ensuite traités par un Transformer :
        
        **1. Patchification** :
        - La série temporelle $X = [x_1, x_2, \\ldots, x_T]$ est divisée en patches de longueur $P$ avec un stride $S$
        - Chaque patch $\\mathbf{p}_i = [x_{iS}, x_{iS+1}, \\ldots, x_{iS+P-1}]$ capture un segment temporel local
        
        **2. Embedding des patches** :
        - Chaque patch est projeté dans un espace de dimension $d_{\\text{model}}$ via une couche linéaire :
        $$
        \\mathbf{e}_i = \\text{Linear}(\\mathbf{p}_i) \\in \\mathbb{R}^{d_{\\text{model}}}
        $$
        
        **3. Positional Encoding** :
        - Un encodage positionnel est ajouté pour préserver l'ordre temporel :
        $$
        \\mathbf{e}_i^{\\text{pos}} = \\mathbf{e}_i + \\text{PE}(i)
        $$
        
        **4. Transformer Encoder** :
        - Les patches encodés passent à travers un Transformer Encoder (attention multi-têtes) :
        $$
        \\mathbf{h}_i = \\text{TransformerEncoder}(\\mathbf{e}_i^{\\text{pos}})
        $$
        
        **5. Extraction de features** :
        - Les représentations des patches sont concaténées et projetées pour produire des features finales :
        $$
        \\mathbf{f} = \\text{Linear}([\\mathbf{h}_1, \\mathbf{h}_2, \\ldots, \\mathbf{h}_N]) \\in \\mathbb{R}^{d_{\\text{features}}}
        $$
        
        **Avantages de PatchTST** :
        - **Efficacité** : Réduit la complexité computationnelle en traitant des patches plutôt que des points individuels
        - **Captures multi-échelles** : Les patches de différentes tailles capturent des motifs à différentes échelles temporelles
        - **Longue portée** : Le mécanisme d'attention permet de capturer des dépendances à long terme
        - **Robustesse** : Moins sensible au bruit grâce à l'agrégation dans les patches
        """)
    
    with st.expander("🎯 Application dans notre projet : Enrichissement de l'observation RL", expanded=False):
        st.markdown("""
        Dans notre projet d'irrigation intelligente, PatchTST est utilisé comme **extracteur de features temporelles** pour enrichir l'observation de l'agent RL :
        
        **Problème** :
        - L'observation standard de l'agent RL est $\\mathbf{o}_t = [\\psi_t, S_t, R_t, ET0_t]$ (4 dimensions)
        - Cette observation ne contient que l'état actuel, sans information sur les tendances ou les motifs temporels passés
        - L'agent a du mal à anticiper les effets à long terme de ses décisions d'irrigation
        
        **Solution avec PatchTST** :
        - PatchTST analyse un historique de $L$ pas de temps : $\\mathbf{X}_{t-L:t} = [\\mathbf{o}_{t-L}, \\mathbf{o}_{t-L+1}, \\ldots, \\mathbf{o}_t]$
        - Il extrait des features temporelles $\\mathbf{f}_t \\in \\mathbb{R}^{d_{\\text{features}}}$ qui capturent :
          - **Tendances** : Évolution à long terme de la tension, de l'humidité du sol
          - **Saisonnalité** : Patterns récurrents liés aux conditions météorologiques
          - **Motifs** : Relations complexes entre irrigation, pluie, évapotranspiration
        - L'observation enrichie devient : $\\mathbf{o}_t^{\\text{enrichi}} = [\\psi_t, S_t, R_t, ET0_t, \\mathbf{f}_t]$
        
        **Avantages** :
        - ✅ L'agent RL reçoit des informations sur les dynamiques temporelles longues
        - ✅ Meilleure anticipation des effets cumulatifs de l'irrigation
        - ✅ Compréhension des tendances et des patterns saisonniers
        - ✅ Décisions plus informées basées sur l'historique complet
        
        ### 📋 Paramètres de configuration du pré-entraînement
        
        **Nombre de trajectoires** :
        - **Signification** : Nombre de simulations indépendantes utilisées pour générer les séquences d'entraînement
        - **Valeur usuelle** : 32 trajectoires (par défaut)
        - **Impact** : Plus de trajectoires = plus de diversité dans les patterns temporels appris
        - **Tuning** : Augmenter (50-100) pour capturer plus de variabilité météorologique, réduire (10-20) pour accélérer
        - **Note** : Chaque trajectoire génère de nombreuses séquences (une par pas de temps avec historique)
        
        **Nombre d'epochs** :
        - **Signification** : Nombre de passages complets sur l'ensemble des séquences d'entraînement
        - **Valeur usuelle** : 10 epochs (par défaut)
        - **Impact** : Plus d'epochs = meilleur apprentissage des patterns temporels complexes
        - **Tuning** : Augmenter (20-50) si la reconstruction/prédiction continue à s'améliorer, réduire si surapprentissage
        - **Note** : Les Transformers peuvent nécessiter plus d'epochs pour converger que les MLP simples
        
        **Taille des batches** :
        - **Signification** : Nombre de séquences traitées simultanément lors de chaque mise à jour
        - **Valeur usuelle** : 64 (par défaut, plus petit que Neural ODE/CDE)
        - **Impact** : Batch plus grand = gradients plus stables mais plus de mémoire (surtout avec séquences longues)
        - **Tuning** : Réduire (16-32) si mémoire limitée ou séquences très longues, augmenter (128+) si disponible
        - **Note** : La mémoire requise augmente quadratiquement avec la longueur de séquence (attention)
        
        **Taux d'apprentissage (Learning Rate)** :
        - **Signification** : Vitesse d'ajustement des paramètres du Transformer
        - **Valeur usuelle** : $10^{-3}$ (0.001) (par défaut)
        - **Impact** : LR trop élevé = instabilité dans l'apprentissage du Transformer
        - **Tuning** : Réduire (10^{-4} - 10^{-5}) si la loss oscille, augmenter (10^{-2}) si apprentissage trop lent
        - **Note** : Les Transformers bénéficient souvent d'un warmup du LR au début de l'entraînement
        
        **Longueur de séquence** :
        - **Signification** : Nombre de pas de temps dans l'historique analysé par PatchTST
        - **Valeur usuelle** : 30 pas de temps (par défaut, ~1 mois)
        - **Impact** : Séquence plus longue = capture de patterns à plus long terme mais plus de complexité
        - **Tuning** : Augmenter (40-60) pour patterns saisonniers longs, réduire (10-20) pour tendances courtes
        - **Note** : Doit être adapté à l'horizon de planification souhaité (anticipation sur plusieurs semaines)
        
        **Dimension des features** :
        - **Signification** : Taille du vecteur de features temporelles extrait par PatchTST
        - **Valeur usuelle** : 16 dimensions (par défaut)
        - **Impact** : Dimension plus grande = plus d'information mais observation plus grande pour l'agent RL
        - **Tuning** : Augmenter (24-32) pour patterns complexes, réduire (8-12) pour simplicité
        - **Note** : Doit être équilibré avec la capacité de l'agent RL à utiliser ces features
        """)
    
    with st.expander("📐 Architecture du modèle PatchTST dans le projet", expanded=False):
        st.markdown("""
        ### Architecture du PatchTST Feature Extractor
        
        Le modèle PatchTST utilisé dans ce projet est un **extracteur de features** qui prend en entrée une séquence d'observations et produit des features temporelles :
        
        **Entrée** :
        - Séquence d'observations : $\\mathbf{X} \\in \\mathbb{R}^{L \\times 4}$ où $L$ est la longueur de la séquence
        - Chaque observation : $[\\psi, I, R, ET0]$
        
        **Patchification** :
        - Patch length : $P = 5$ (chaque patch couvre 5 pas de temps)
        - Stride : $S = 1$ (patches se chevauchent)
        - Nombre de patches : $N = \\lfloor (L - P) / S \\rfloor + 1$
        
        **Embedding** :
        - Chaque patch $\\mathbf{p}_i \\in \\mathbb{R}^{P \\times 4}$ est aplati en $\\mathbb{R}^{P \\times 4}$
        - Projection linéaire : $\\mathbf{e}_i = \\text{Linear}(\\text{flatten}(\\mathbf{p}_i)) \\in \\mathbb{R}^{d_{\\text{model}}}$
        - $d_{\\text{model}} = 64$ (dimension du modèle)
        
        **Transformer Encoder** :
        - Nombre de couches : $n_{\\text{layers}} = 2$
        - Nombre de têtes d'attention : $n_{\\text{heads}} = 4$
        - Dimension du feed-forward : $d_{\\text{ff}} = 4 \\times d_{\\text{model}} = 256$
        - Positional encoding : Ajouté pour préserver l'ordre temporel
        
        **Extraction de features** :
        - Les représentations des patches sont concaténées : $[\\mathbf{h}_1, \\mathbf{h}_2, \\ldots, \\mathbf{h}_N]$
        - Projection finale : $\\mathbf{f} = \\text{Linear}(\\text{concat}(\\mathbf{h}_1, \\ldots, \\mathbf{h}_N)) \\in \\mathbb{R}^{d_{\\text{features}}}$
        - $d_{\\text{features}} = 16$ (dimension des features extraites)
        
        **Sortie** :
        - Features temporelles : $\\mathbf{f} \\in \\mathbb{R}^{16}$
        - Ces features sont concaténées à l'observation standard pour former l'observation enrichie
        """)
    
    with st.expander("🎓 Processus d'entraînement de PatchTST", expanded=False):
        st.markdown("""
        ### Pré-entraînement de PatchTST
        
        PatchTST est pré-entraîné sur des données simulées avant d'être utilisé comme extracteur de features pour l'agent RL :
        
        **1. Génération de données** :
        - Simulation de $N_{\\text{traj}}$ trajectoires avec le modèle physique
        - Pour chaque trajectoire, extraction de séquences de longueur $L$ : $\\mathbf{X}_{t-L:t}$
        - Les séquences capturent différentes conditions météorologiques et stratégies d'irrigation
        
        **2. Tâche de pré-entraînement** :
        - **Auto-supervisé (reconstruction)** : PatchTST apprend à reconstruire la séquence d'entrée à partir des features extraites
        - **Supervisé (prédiction)** : PatchTST apprend à prédire des statistiques de la séquence (tendance, variance, moyenne)
        
        **3. Fonction de perte** :
        - Pour la reconstruction : $\\mathcal{L} = \\|\\mathbf{X}_{\\text{recon}} - \\mathbf{X}_{\\text{original}}\\|_2^2$
        - Pour la prédiction : $\\mathcal{L} = \\|\\mathbf{y}_{\\text{pred}} - \\mathbf{y}_{\\text{target}}\\|_2^2$
        
        **4. Optimisation** :
        - Optimiseur : Adam
        - Learning rate : $10^{-3}$
        - Nombre d'epochs : 10-20
        - Batch size : 64
        
        **5. Utilisation dans l'environnement RL** :
        - Après pré-entraînement, PatchTST est figé (frozen)
        - À chaque pas de temps, l'historique des observations est passé à PatchTST
        - Les features extraites sont ajoutées à l'observation de l'agent RL
        """)
    
    with st.expander("🔄 Intégration dans l'environnement RL", expanded=False):
        st.markdown("""
        ### Wrapper d'environnement PatchTST
        
        Un wrapper `PatchTSTEnvWrapper` enrichit l'observation de l'environnement RL :
        
        **1. Historique des observations** :
        - Le wrapper maintient un historique des $L$ dernières observations : $\\{\\mathbf{o}_{t-L}, \\ldots, \\mathbf{o}_t\\}$
        - À chaque pas de temps, la nouvelle observation est ajoutée à l'historique
        
        **2. Extraction de features** :
        - L'historique est passé à PatchTST : $\\mathbf{f}_t = \\text{PatchTST}([\\mathbf{o}_{t-L}, \\ldots, \\mathbf{o}_t])$
        - PatchTST est en mode évaluation (pas de gradient)
        
        **3. Observation enrichie** :
        - L'observation originale : $\\mathbf{o}_t = [\\psi_t, S_t, R_t, ET0_t] \\in \\mathbb{R}^4$
        - L'observation enrichie : $\\mathbf{o}_t^{\\text{enrichi}} = [\\psi_t, S_t, R_t, ET0_t, \\mathbf{f}_t] \\in \\mathbb{R}^{4 + d_{\\text{features}}}$
        - L'espace d'observation est mis à jour pour refléter la nouvelle dimension
        
        **4. Entraînement PPO** :
        - L'agent PPO reçoit l'observation enrichie
        - La politique apprend à utiliser les features temporelles pour prendre de meilleures décisions
        - Les features permettent à l'agent de comprendre les tendances et d'anticiper les effets futurs
        """)
    
    with st.expander("📊 Variables et notations", expanded=False):
        st.markdown("""
        ### Variables et notations utilisées
        
        **Série temporelle** :
        - $X = [x_1, x_2, \\ldots, x_T]$ : Série temporelle de longueur $T$
        - $L$ : Longueur de la séquence d'historique utilisée par PatchTST
        - $P$ : Longueur d'un patch (nombre de pas de temps par patch)
        - $S$ : Stride (décalage entre patches consécutifs)
        - $N = \\lfloor (L - P) / S \\rfloor + 1$ : Nombre de patches
        
        **Architecture** :
        - $d_{\\text{model}}$ : Dimension du modèle Transformer (64)
        - $d_{\\text{features}}$ : Dimension des features extraites (16)
        - $n_{\\text{layers}}$ : Nombre de couches Transformer (2)
        - $n_{\\text{heads}}$ : Nombre de têtes d'attention (4)
        - $d_{\\text{ff}}$ : Dimension du feed-forward network (256)
        
        **Observations** :
        - $\\mathbf{o}_t = [\\psi_t, S_t, R_t, ET0_t]$ : Observation standard au temps $t$
        - $\\mathbf{X}_{t-L:t} = [\\mathbf{o}_{t-L}, \\ldots, \\mathbf{o}_t]$ : Historique de $L$ observations
        - $\\mathbf{f}_t$ : Features temporelles extraites par PatchTST
        - $\\mathbf{o}_t^{\\text{enrichi}} = [\\mathbf{o}_t, \\mathbf{f}_t]$ : Observation enrichie
        
        **Entraînement** :
        - $N_{\\text{traj}}$ : Nombre de trajectoires simulées pour le pré-entraînement
        - $\\mathcal{L}$ : Fonction de perte (MSE pour reconstruction ou prédiction)
        - $\\theta$ : Paramètres du modèle PatchTST
        """)
    
    with st.expander("🔬 Relation avec les Physics-Informed Neural Networks (PINN)", expanded=False):
        st.markdown("""
        ### Neural ODE comme modèle Physics-Informed
        
        Le Neural ODE utilisé dans ce projet peut être considéré comme un **modèle hybride physics-informed** :
        
        **Définition des PINN** :
        Les Physics-Informed Neural Networks (PINN) sont des réseaux de neurones qui intègrent explicitement les lois de la physique dans leur architecture ou leur fonction de perte. 
        Ils combinent généralement :
        - Des équations différentielles comme contraintes
        - Des termes de régularisation basés sur la physique
        - Une combinaison de données et de connaissances physiques
        
        **Notre approche hybride** :
        - **Modèle physique** : Fournit la structure et les contraintes physiques (bilan hydrique FAO)
        - **Neural ODE** : Apprend une correction résiduelle qui respecte implicitement la structure physique
        
        **Caractéristiques physics-informed** :
        - ✅ **Intégration explicite de la physique** : Le modèle physique FAO est intégré directement dans l'architecture
        - ✅ **Respect des contraintes physiques** : La correction $\\Delta \\psi$ est appliquée de manière cohérente avec le modèle physique
        - ✅ **Apprentissage guidé par la physique** : Le Neural ODE apprend à partir de données mais dans le contexte d'un modèle physique
        - ✅ **Interprétabilité** : La séparation entre physique et correction permet de comprendre les écarts
        
        **Différence avec les PINN classiques** :
        - **PINN classiques** : Intègrent les équations différentielles directement dans la fonction de perte (ex: $\\mathcal{L} = \\mathcal{L}_{\\text{data}} + \\lambda \\mathcal{L}_{\\text{physics}}$)
        - **Notre approche** : Utilise un modèle physique explicite comme base et apprend une correction résiduelle
        
        **Conclusion** :
        Notre modèle Neural ODE est un **modèle hybride physics-informed** qui combine le meilleur des deux mondes : 
        la robustesse et l'interprétabilité du modèle physique avec la flexibilité d'apprentissage des réseaux de neurones.
        """)
    
    with st.expander("⚖️ Avantages et limitations", expanded=False):
        st.markdown("""
        ### Avantages de PatchTST comme extracteur de features
        
        **✅ Captures temporelles longues** :
        - PatchTST peut capturer des dépendances à long terme grâce au mécanisme d'attention
        - Les patches permettent de capturer des motifs à différentes échelles temporelles
        
        **✅ Efficacité computationnelle** :
        - Le traitement par patches réduit la complexité par rapport à un traitement point par point
        - Le pré-entraînement permet de réutiliser les features sans recalculer à chaque pas
        
        **✅ Robustesse** :
        - L'agrégation dans les patches réduit la sensibilité au bruit
        - Les features extraites sont plus stables que les observations brutes
        
        **✅ Flexibilité** :
        - Peut être pré-entraîné sur différentes tâches (reconstruction, prédiction)
        - Les features peuvent être utilisées pour différents types d'agents RL
        
        ### Limitations
        
        **⚠️ Complexité** :
        - Ajoute une couche de complexité à l'architecture
        - Nécessite un pré-entraînement supplémentaire
        
        **⚠️ Hyperparamètres** :
        - Sensible aux hyperparamètres (longueur de patch, stride, dimension des features)
        - Nécessite un tuning pour chaque application
        
        **⚠️ Mémoire** :
        - Nécessite de maintenir un historique des observations
        - Augmente légèrement la consommation mémoire
        
        **⚠️ Interprétabilité** :
        - Les features extraites sont moins interprétables que les observations brutes
        - Difficile de comprendre exactement ce que chaque feature représente
        """)
    
    with st.expander("🔧 Paramètres recommandés et tuning", expanded=False):
        st.markdown("""
        ### Hyperparamètres du pré-entraînement PatchTST
        
        **Nombre de trajectoires** : 32-50 (recommandé)
        - Plus de trajectoires = plus de diversité dans les patterns
        - Tuning : Augmenter (50-100) pour variabilité météo, réduire (10-20) pour accélérer
        
        **Nombre d'epochs** : 10-20 (recommandé)
        - Plus d'epochs = meilleur apprentissage
        - Tuning : Augmenter (20-50) si loss continue à décroître, réduire si surapprentissage
        
        **Taille des batches** : 64 (recommandé)
        - Plus grand = gradients plus stables mais plus de mémoire
        - Tuning : Réduire (16-32) si mémoire limitée, augmenter (128+) si disponible
        
        **Taux d'apprentissage** : $10^{-3}$ (recommandé)
        - Trop élevé = instabilité
        - Tuning : Réduire ($10^{-4}$-$10^{-5}$) si loss oscille, augmenter ($10^{-2}$) si trop lent
        
        **Longueur de séquence** ($L$) : 30 (recommandé, ~1 mois)
        - Plus long = capture patterns plus longs mais plus complexe
        - Tuning : Augmenter (40-60) pour saisonnalité, réduire (10-20) pour tendances courtes
        
        **Dimension des features** ($d_{\\text{features}}$) : 16 (recommandé)
        - Plus grande = plus d'information mais observation plus grande
        - Tuning : Augmenter (24-32) pour patterns complexes, réduire (8-12) pour simplicité
        
        **Type de tâche** :
        - **Auto-supervisé (reconstruction)** : Recommandé pour début
        - **Supervisé (prédiction)** : Si objectif spécifique
        
        ### Hyperparamètres PPO
        
        - Identiques au Scénario 2
        - Learning rate : $3 \\times 10^{-4}$
        - Gamma : 0.99
        - Observation space : 4 + $d_{\\text{features}}$ dimensions
        
        ### Stratégie de tuning
        
        **1. Pré-entraînement PatchTST** :
        - Commencer avec valeurs par défaut
        - Observer convergence de la loss
        - Ajuster selon résultats
        
        **2. Entraînement PPO** :
        - Comme Scénario 2
        - Observer si features améliorent performance
        - Comparer avec Scénario 2
        """)
    
    with st.expander("🧭 Quand utiliser le Scénario 5 ?", expanded=False):
        st.markdown("""
        ### ✅ Choisir le Scénario 5 si :
        
        - **Besoin de mémoire temporelle longue** :
          - Comprendre tendances et saisonnalité
          - Capturer patterns à long terme (30+ jours)
          - Scénario 2 insuffisant pour contexte temporel
        
        - **Enrichissement d'observation** :
          - Observation standard (4D) insuffisante
          - Besoin de features temporelles avancées
          - Améliorer compréhension de l'agent RL
        
        - **Pas de correction physique nécessaire** :
          - Modèle physique fiable
          - Pas de biais à corriger
          - Focus sur amélioration observation
        
        - **Données de simulation disponibles** :
          - Possibilité de générer trajectoires pour pré-entraînement
          - Pas besoin de données réelles (pré-entraînement auto-supervisé)
          - Qualité de simulation acceptable
        
        - **Performance améliorée recherchée** :
          - Scénario 2 ne suffit pas
          - Besoin de meilleure compréhension temporelle
          - Ressources computationnelles disponibles
        
        ### ❌ Ne pas choisir le Scénario 5 si :
        
        - **Pas de dépendances temporelles longues** :
          - Effets locaux uniquement
          - Pas besoin de tendances/saisonnalité
          - Scénario 2 suffit
        
        - **Biais du modèle physique** :
          - Modèle physique a des biais connus
          - Nécessité de corriger les prédictions
          - → Préférer Scénarios 3-4
        
        - **Simplicité recherchée** :
          - Approche simple suffit
          - Pas de ressources pour pré-entraînement
          - → Préférer Scénarios 1-2
        
        - **Besoin de planification** :
          - Besoin de planification explicite
          - Rollouts d'imagination nécessaires
          - → Préférer Scénario 6
        """)
    
    with st.expander("🛠️ Conseils pratiques", expanded=False):
        st.markdown("""
        ### Workflow recommandé
        
        **1. Pré-entraînement PatchTST** :
        - Générer trajectoires avec modèle physique
        - Extraire séquences de longueur $L$
        - Pré-entraîner PatchTST (reconstruction ou prédiction)
        - Vérifier convergence (loss décroît)
        
        **2. Intégration dans environnement** :
        - Créer wrapper qui enrichit observation
        - Maintenir historique de $L$ observations
        - Extraire features à chaque pas
        
        **3. Entraînement PPO** :
        - Comme Scénario 2
        - Observer si features améliorent performance
        - Comparer avec Scénario 2
        
        **4. Analyse** :
        - Analyser quelles features sont utilisées
        - Vérifier amélioration vs Scénario 2
        - Ajuster hyperparamètres si nécessaire
        
        ### Troubleshooting
        
        **Problème : Features non informatives**
        - **Symptôme** : Pas d'amélioration vs Scénario 2
        - **Solutions** :
          - Augmenter longueur de séquence $L$
          - Augmenter dimension des features
          - Vérifier qualité du pré-entraînement
        
        **Problème : Pré-entraînement ne converge pas**
        - **Symptôme** : Loss ne décroît pas
        - **Solutions** :
          - Réduire learning rate
          - Augmenter nombre de trajectoires
          - Vérifier qualité des données
        
        **Problème : Mémoire insuffisante**
        - **Symptôme** : Erreur mémoire lors pré-entraînement
        - **Solutions** :
          - Réduire batch size
          - Réduire longueur de séquence $L$
          - Réduire dimension des features
        """)
    
    with st.expander("🔬 Relation avec les Physics-Informed Neural Networks (PINN)", expanded=False):
        st.markdown("""
        ### PatchTST : Modèle data-driven, pas un PINN
        
        **Définition des PINN** :
        Les Physics-Informed Neural Networks (PINN) sont des réseaux de neurones qui intègrent explicitement les lois de la physique dans leur architecture ou leur fonction de perte.
        
        **PatchTST dans notre projet** :
        - **Rôle** : Extracteur de features temporelles purement data-driven
        - **Apprentissage** : Basé uniquement sur des patterns temporels dans les données, sans intégration explicite de connaissances physiques
        - **Objectif** : Capturer des tendances, saisonnalités et motifs temporels pour enrichir l'observation de l'agent RL
        
        **Pourquoi PatchTST n'est pas un PINN** :
        - ❌ **Pas d'intégration de physique** : PatchTST n'intègre pas explicitement les équations physiques (bilan hydrique, courbe de rétention, etc.)
        - ❌ **Apprentissage purement data-driven** : Le modèle apprend uniquement à partir de patterns dans les données, sans contraintes physiques
        - ❌ **Pas de régularisation physique** : La fonction de perte ne contient pas de termes basés sur les lois de la physique
        
        **Relation indirecte avec la physique** :
        - ✅ **Données générées par un modèle physique** : PatchTST est pré-entraîné sur des données simulées par le modèle physique FAO
        - ✅ **Features informatives** : Les features extraites capturent indirectement des patterns liés à la physique (tendances de tension, cycles d'irrigation, etc.)
        - ✅ **Complémentarité** : PatchTST enrichit l'observation de l'agent RL qui évolue dans un environnement basé sur un modèle physique
        
        **Conclusion** :
        PatchTST n'est **pas un PINN** mais un modèle **data-driven** qui complète l'approche physics-informed en fournissant des features temporelles avancées à l'agent RL. 
        L'approche globale du projet combine :
        - **Modèles physics-informed** (Neural ODE, Neural CDE) : Pour améliorer la prédiction du modèle physique
        - **Modèles data-driven** (PatchTST) : Pour enrichir la compréhension temporelle de l'agent RL
        """)
    
    with st.expander("🔗 Comparaison avec Neural ODE et Neural CDE", expanded=False):
        st.markdown("""
        ### Différences avec Neural ODE et Neural CDE
        
        **Neural ODE** :
        - **Rôle** : Correction résiduelle du modèle physique
        - **Entrée** : État actuel $[\\psi_t, I_t, R_t, ET0_t]$
        - **Sortie** : Correction $\\Delta \\psi_t$ ajoutée à la prédiction physique
        - **Mémoire** : Aucune (dépend uniquement de l'état actuel)
        
        **Neural CDE** :
        - **Rôle** : Correction résiduelle avec mémoire temporelle
        - **Entrée** : Séquence d'états passés $[\\psi_{t-k}, \\ldots, \\psi_t]$
        - **Sortie** : Correction $\\Delta \\psi_t$ basée sur l'historique
        - **Mémoire** : Court terme (5-10 pas de temps)
        
        **PatchTST** :
        - **Rôle** : Extracteur de features temporelles pour l'agent RL
        - **Entrée** : Historique d'observations $[\\mathbf{o}_{t-L}, \\ldots, \\mathbf{o}_t]$
        - **Sortie** : Features temporelles $\\mathbf{f}_t$ enrichissant l'observation
        - **Mémoire** : Long terme (30+ pas de temps)
        - **Utilisation** : Enrichit l'observation de l'agent, ne modifie pas le modèle physique
        
        **Complémentarité** :
        - Neural ODE/CDE améliorent la **prédiction du modèle physique**
        - PatchTST améliore la **compréhension de l'agent RL** des dynamiques temporelles
        - Les deux approches peuvent être combinées pour un système encore plus performant
        """)
    
    # ========================================================================
    # ONGLET DOCUMENTATION 7 : SCÉNARIOS
    # ========================================================================


def render_doc_scenario6_world_model():
    """
    Affiche le contenu de l'onglet de documentation : Scénario 6 (World Model).
    """
    st.markdown('<h2 class="section-header">🌍 Scénario 6 — Model-Based RL avec World Model</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### ❓ Qu'est-ce qu'un World Model ?
    
    Un **World Model** est un modèle interne de l'environnement appris par l'agent RL. Contrairement aux approches model-free 
    (comme PPO dans les Scénarios 2-5) qui apprennent directement une politique à partir des interactions avec l'environnement, 
    un model-based RL utilise un modèle du monde pour **planifier** et **simuler** les conséquences des actions avant de les exécuter.
    
    **Objectif principal** : Tirer parti d'un modèle du monde pour planifier et améliorer l'efficacité sample tout en gardant 
    les contraintes physiques via le modèle FAO. Cette approche permet une meilleure **sample efficiency** et une **planification à long terme**.
    
    **Pourquoi utiliser un World Model ?**
    - ✅ **Sample efficiency** : Réduit le nombre d'interactions réelles avec l'environnement nécessaires pour apprendre
    - ✅ **Planification** : Permet d'anticiper les effets des actions sur plusieurs pas de temps
    - ✅ **Rollouts d'imagination** : Simulation de trajectoires futures pour améliorer l'apprentissage
    - ✅ **Robustesse** : Compréhension plus profonde de la dynamique de l'environnement
    """)
    
    with st.expander("🔬 Principe général des World Models", expanded=False):
        st.markdown("""
        ### Architecture d'un World Model
        
        Un World Model typique est composé de trois composants principaux :
        
        **1. Encodeur (Representation Model)** :
        - **Rôle** : Comprime les observations brutes en un état latent compact
        - **Entrée** : Observations $\\mathbf{o}_t = [\\psi_t, S_t, R_t, ET0_t] \\in \\mathbb{R}^4$
        - **Sortie** : État latent $\\mathbf{z}_t \\in \\mathbb{R}^{d_z}$ où $d_z$ est la dimension de l'espace latent
        - **Formule** : $\\mathbf{z}_t = \\text{Encoder}(\\mathbf{o}_t)$
        - **Dans notre projet** : PatchTST encode un historique de $L$ observations en un état latent
        
        **2. Modèle de Transition (Dynamics Model)** :
        - **Rôle** : Prédit l'évolution de l'état latent après une action
        - **Entrée** : État latent actuel $\\mathbf{z}_t$ et action $a_t$
        - **Sortie** : État latent futur $\\hat{\\mathbf{z}}_{t+1}$
        - **Formule** : $\\hat{\\mathbf{z}}_{t+1} = f_{\\text{transition}}(\\mathbf{z}_t, a_t)$
        - **Dans notre projet** : Neural ODE ou Neural CDE pour modéliser la dynamique temporelle
        
        **3. Décodeur (Observation Model, optionnel)** :
        - **Rôle** : Reconstruit les observations depuis l'état latent (pour rollouts longs)
        - **Entrée** : État latent $\\mathbf{z}_t$
        - **Sortie** : Observations reconstruites $\\hat{\\mathbf{o}}_t$
        - **Formule** : $\\hat{\\mathbf{o}}_t = \\text{Decoder}(\\mathbf{z}_t)$
        - **Dans notre projet** : MLP qui reconstruit $[\\psi, S, R, ET0]$ depuis $\\mathbf{z}_t$
        
        ### Rollouts d'Imagination
        
        Le World Model permet de générer des **rollouts d'imagination** : des trajectoires simulées dans l'espace latent :
        
        $$
        \\begin{aligned}
        \\mathbf{z}_t &= \\text{Encoder}(\\mathbf{o}_t) \\\\
        \\mathbf{z}_{t+1} &= f_{\\text{transition}}(\\mathbf{z}_t, a_t) \\\\
        \\mathbf{z}_{t+2} &= f_{\\text{transition}}(\\mathbf{z}_{t+1}, a_{t+1}) \\\\
        &\\vdots \\\\
        \\mathbf{z}_{t+H} &= f_{\\text{transition}}(\\mathbf{z}_{t+H-1}, a_{t+H-1})
        \\end{aligned}
        $$
        
        où $H$ est l'horizon d'imagination. Ces rollouts permettent à l'agent de planifier et d'anticiper les conséquences de ses actions.
        
        ### Avantages des World Models
        
        - **Sample Efficiency** : Moins d'interactions réelles nécessaires grâce à la simulation interne
        - **Planification** : Capacité à explorer différentes stratégies sans coût réel
        - **Compression** : L'espace latent capture l'information essentielle de manière compacte
        - **Généralisation** : Compréhension plus profonde des dynamiques permet de mieux généraliser
        """)
    
    with st.expander("🎯 Application dans notre projet : Architecture du Scénario 6", expanded=False):
        st.markdown("""
        ### Architecture complète du World Model
        
        Notre World Model combine plusieurs composants pour créer un système de planification performant :
        
        **1. Encodeur PatchTST** :
        - **Rôle** : Transforme l'historique d'observations en représentation latente riche
        - **Entrée** : Historique $\\mathbf{X}_{t-L:t} = [\\mathbf{o}_{t-L}, \\ldots, \\mathbf{o}_t] \\in \\mathbb{R}^{L \\times 4}$
        - **Sortie** : État latent $\\mathbf{z}_t \\in \\mathbb{R}^{d_z}$ où $d_z$ est la dimension des features PatchTST (16 par défaut)
        - **Réutilisation** : Utilise le modèle PatchTST pré-entraîné du Scénario 5
        - **Formule** : $\\mathbf{z}_t = \\text{PatchTST}(\\mathbf{X}_{t-L:t})$
        
        **2. Modèle de Transition** :
        
        **Phase 1 - Neural ODE** :
        - Modélise la dynamique comme une équation différentielle ordinaire continue
        - **Formule** : $\\frac{d\\mathbf{z}(t)}{dt} = f_{\\theta}(\\mathbf{z}(t), a(t))$
        - Intégration numérique (Euler ou Runge-Kutta) pour obtenir $\\mathbf{z}_{t+1}$
        - **Avantage** : Simple et efficace pour transitions courtes
        
        **Phase 2 - Neural CDE** :
        - Modélise la dynamique avec mémoire temporelle via équations différentielles contrôlées
        - Prend en compte l'historique des états pour mieux prédire
        - **Formule** : $d\\mathbf{z}_t = f_{\\theta}(\\mathbf{z}_t, a_t, \\mathbf{h}_t) dt$
        - **Avantage** : Capture mieux les dépendances temporelles longues
        
        **3. Décodeur (Phase 2 uniquement)** :
        - Reconstruit les observations depuis l'état latent pour rollouts longs
        - **Formule** : $\\hat{\\mathbf{o}}_t = \\text{Decoder}(\\mathbf{z}_t)$
        - Permet de simuler des trajectoires complètes avec observables
        
        **4. Wrapper Physics-Informed (Phase 3)** :
        - Combine les prédictions du World Model avec le modèle physique FAO
        - **Formule** : $\\mathbf{o}_{t+1} = \\alpha \\cdot \\mathbf{o}_{t+1}^{\\text{physique}} + (1-\\alpha) \\cdot \\hat{\\mathbf{o}}_{t+1}^{\\text{world model}}$
        - Où $\\alpha$ est un hyperparamètre de blend (0.3-0.6 recommandé)
        - **Avantage** : Garantit la cohérence physique même si le World Model dérive
        """)
    
    with st.expander("⚙️ Les trois phases du Scénario 6", expanded=False):
        st.markdown("""
        ### Phase 1 : World Model Simple (sans décodeur)
        
        **Architecture** :
        - Encodeur : PatchTST pré-entraîné
        - Transition : Neural ODE simple
        - Décodeur : ❌ Aucun (agent travaille directement dans l'espace latent)
        
        **Rollouts d'imagination** :
        - Horizon court : 5-10 pas de temps
        - Trajectoires simulées uniquement dans l'espace latent
        - Récompenses estimées directement depuis $\\mathbf{z}_t$ (via une fonction de récompense latente)
        
        **Objectif** :
        - Valider le concept avec une implémentation minimale
        - Accélérer l'apprentissage PPO avec rollouts courts
        - Évaluer la faisabilité du World Model
        
        **Avantages** :
        - ✅ Simple et rapide à entraîner
        - ✅ Moins de paramètres à optimiser
        - ✅ Bon point de départ pour validation
        
        **Limites** :
        - ⚠️ Pas de reconstruction des observables
        - ⚠️ Rollouts limités à horizon court
        - ⚠️ Difficile de vérifier la cohérence physique
        
        ### Phase 2 : World Model Complet (avec décodeur)
        
        **Architecture** :
        - Encodeur : PatchTST pré-entraîné
        - Transition : Neural CDE (mémoire temporelle améliorée)
        - Décodeur : ✅ MLP qui reconstruit les observables
        
        **Rollouts d'imagination** :
        - Horizon long : 20-30 pas de temps
        - Trajectoires complètes avec observables reconstruits
        - Planification à long terme possible
        
        **Objectif** :
        - Maximiser les bénéfices du model-based RL
        - Permettre une planification à long terme
        - Capturer des stratégies complexes
        
        **Avantages** :
        - ✅ Rollouts longs pour planification
        - ✅ Observables reconstruits permettent vérification
        - ✅ Neural CDE capture mieux les dépendances temporelles
        
        **Limites** :
        - ⚠️ Plus complexe à entraîner (3 composants)
        - ⚠️ Risque de dérive si pré-entraînement insuffisant
        - ⚠️ Coût computationnel plus élevé
        
        ### Phase 3 : Hybridation Physics-Informed
        
        **Architecture** :
        - Tous les composants de Phase 2
        - **Plus** : Wrapper qui combine World Model et modèle physique FAO
        
        **Principe de blend** :
        - À chaque pas de temps, on combine :
          - Prédiction du modèle physique : $\\mathbf{o}_{t+1}^{\\text{physique}} = f_{\\text{FAO}}(\\mathbf{o}_t, a_t)$
          - Prédiction du World Model : $\\hat{\\mathbf{o}}_{t+1}^{\\text{world model}} = \\text{Decoder}(f_{\\text{CDE}}(\\mathbf{z}_t, a_t))$
        - Prédiction finale : $\\mathbf{o}_{t+1} = \\alpha \\cdot \\mathbf{o}_{t+1}^{\\text{physique}} + (1-\\alpha) \\cdot \\hat{\\mathbf{o}}_{t+1}^{\\text{world model}}$
        
        **Objectif** :
        - Garantir la cohérence physique
        - Limiter la dérive du World Model
        - Combiner efficacité du World Model et robustesse du modèle physique
        
        **Avantages** :
        - ✅ Robuste physiquement (le modèle physique corrige les dérives)
        - ✅ Meilleur des deux mondes (efficacité + robustesse)
        - ✅ Permet de régler le compromis via $\\alpha$
        
        **Recommandation pour $\\alpha$** :
        - $\\alpha = 0.5$ : Équilibre entre physique et World Model
        - $\\alpha = 0.6-0.7$ : Privilégie la physique (recommandé si World Model dérive)
        - $\\alpha = 0.3-0.4$ : Privilégie le World Model (si bien entraîné)
        """)
    
    with st.expander("🚀 Processus d'entraînement détaillé", expanded=False):
        st.markdown("""
        ### Pipeline complet d'entraînement
        
        Le Scénario 6 suit un pipeline progressif en plusieurs étapes :
        
        **Étape 0 : Pré-entraînement PatchTST (Scénario 5)**
        - **Objectif** : Obtenir un encodeur capable de comprimer l'historique d'observations
        - **Processus** :
          1. Génération de $N_{\\text{traj}}$ trajectoires avec le modèle physique
          2. Extraction de séquences de longueur $L$ pour chaque trajectoire
          3. Pré-entraînement de PatchTST sur reconstruction/prédiction
          4. Sauvegarde du modèle PatchTST comme encodeur
        - **Résultat** : Encodeur PatchTST prêt avec features de dimension $d_z$
        
        **Étape 1 : Pré-entraînement du Modèle de Transition (Phase 1 ou 2)**
        - **Objectif** : Apprendre la dynamique de transition dans l'espace latent
        - **Processus** :
          1. Génération de $N_{\\text{traj}}$ nouvelles trajectoires
          2. Encodage avec PatchTST : $\\mathbf{z}_t = \\text{PatchTST}(\\mathbf{X}_{t-L:t})$
          3. Construction de paires $(\\mathbf{z}_t, a_t, \\mathbf{z}_{t+1})$ pour l'entraînement
          4. Minimisation de la perte de prédiction :
             $$
             \\mathcal{L}_{\\text{transition}} = \\frac{1}{N} \\sum_{i=1}^{N} \\|f_{\\theta}(\\mathbf{z}_t^{(i)}, a_t^{(i)}) - \\mathbf{z}_{t+1}^{(i)}\\|_2^2
             $$
          5. (Phase 2 uniquement) Pré-entraînement du décodeur :
             $$
             \\mathcal{L}_{\\text{decodeur}} = \\frac{1}{N} \\sum_{i=1}^{N} \\|\\text{Decoder}(\\mathbf{z}_t^{(i)}) - \\mathbf{o}_t^{(i)}\\|_2^2
             $$
        - **Hyperparamètres** :
          - Nombre de trajectoires : 32-50
          - Nombre d'epochs : 20-50
          - Batch size : 64
          - Learning rate : $10^{-3}$
        
        **Étape 2 : Entraînement PPO avec World Model**
        - **Objectif** : Apprendre une politique optimale en utilisant les rollouts d'imagination
        - **Processus** :
          1. Pour chaque interaction avec l'environnement réel :
             - Observation $\\mathbf{o}_t$ → Encodage $\\mathbf{z}_t = \\text{Encoder}(\\mathbf{o}_t)$
             - Action $a_t \\sim \\pi_\\theta(\\cdot | \\mathbf{z}_t)$
             - Exécution dans l'environnement réel → $\\mathbf{o}_{t+1}, r_t$
          2. Pour chaque rollout d'imagination (horizon $H$) :
             $$
             \\begin{aligned}
             \\mathbf{z}_t &= \\text{Encoder}(\\mathbf{o}_t) \\\\
             \\mathbf{z}_{t+1} &= f_{\\text{transition}}(\\mathbf{z}_t, a_t) \\\\
             \\hat{r}_{t+1} &= r_{\\text{latent}}(\\mathbf{z}_{t+1}) \\quad \\text{(Phase 1)} \\\\
             \\text{ou} \\quad \\hat{\\mathbf{o}}_{t+1} &= \\text{Decoder}(\\mathbf{z}_{t+1}), \\quad \\hat{r}_{t+1} = r(\\hat{\\mathbf{o}}_{t+1}) \\quad \\text{(Phase 2)} \\\\
             &\\vdots
             \\end{aligned}
             $$
          3. Utilisation des rollouts pour enrichir l'apprentissage PPO :
             - Estimation de la fonction de valeur avec trajectoires réelles + imaginaires
             - Calcul du gradient de la politique avec données augmentées
          4. Mise à jour des paramètres PPO : $\\theta \\leftarrow \\theta + \\alpha \\nabla_\\theta J(\\theta)$
        - **Hyperparamètres PPO** :
          - Total timesteps : 10,000-100,000
          - Horizon d'imagination : 5-10 (Phase 1), 20-30 (Phase 2)
          - Learning rate : $3 \\times 10^{-4}$
          - Gamma : 0.99 (discount factor élevé pour planification)
          - GAE lambda : 0.95
        
        **Étape 3 : Phase 3 - Hybridation (optionnelle)**
        - **Objectif** : Intégrer le modèle physique pour garantir la robustesse
        - **Processus** :
          1. Wrapper de l'environnement qui combine World Model et physique
          2. À chaque transition :
             - World Model prédit : $\\hat{\\mathbf{o}}_{t+1}^{\\text{WM}}$
             - Modèle physique prédit : $\\mathbf{o}_{t+1}^{\\text{physique}}$
             - Blend : $\\mathbf{o}_{t+1} = \\alpha \\cdot \\mathbf{o}_{t+1}^{\\text{physique}} + (1-\\alpha) \\cdot \\hat{\\mathbf{o}}_{t+1}^{\\text{WM}}$
          3. Entraînement PPO sur l'environnement hybridé
        - **Réglage de $\\alpha$** : Commencer à 0.5, ajuster selon stabilité
        """)

    with st.expander("📐 Architecture détaillée des composants", expanded=False):
        st.markdown("""
        ### Encodeur PatchTST
        
        **Rôle** : Compresse l'historique d'observations en représentation latente
        
        **Architecture** :
        - PatchTST pré-entraîné (réutilisé du Scénario 5)
        - Entrée : Historique $\\mathbf{X}_{t-L:t} \\in \\mathbb{R}^{L \\times 4}$ où $L$ = longueur de séquence (30 par défaut)
        - Sortie : État latent $\\mathbf{z}_t \\in \\mathbb{R}^{d_z}$ où $d_z$ = dimension des features (16 par défaut)
        
        **Hyperparamètres** :
        - Longueur de séquence : $L = 30$ (recommandé)
        - Dimension des features : $d_z = 16$ (recommandé)
        - Dimension du modèle : $d_{\\text{model}} = 64$
        - Nombre de couches : 2
        - Nombre de têtes d'attention : 4
        
        ### Modèle de Transition - Neural ODE (Phase 1)
        
        **Rôle** : Modélise la dynamique de transition dans l'espace latent
        
        **Architecture** :
        - Réseau de neurones $f_\\theta$ qui définit le champ de vecteurs
        - Intégration numérique (méthode d'Euler ou Runge-Kutta) pour résoudre l'ODE
        
        **Équation** :
        $$
        \\frac{d\\mathbf{z}(t)}{dt} = f_\\theta(\\mathbf{z}(t), a(t))
        $$
        
        **Intégration** :
        $$
        \\mathbf{z}_{t+1} = \\mathbf{z}_t + \\int_t^{t+1} f_\\theta(\\mathbf{z}(s), a(s)) ds
        $$
        
        **Architecture de $f_\\theta$** :
        - MLP avec 2-3 couches cachées
        - Dimension cachée : 64-128
        - Activation : ReLU ou Tanh
        
        ### Modèle de Transition - Neural CDE (Phase 2)
        
        **Rôle** : Modélise la dynamique avec mémoire temporelle
        
        **Architecture** :
        - Neural CDE qui prend en compte l'historique via un chemin contrôlé
        - Permet de capturer des dépendances temporelles plus longues que Neural ODE
        
        **Équation** :
        $$
        d\\mathbf{z}_t = f_\\theta(\\mathbf{z}_t, a_t, \\mathbf{h}_t) dt + g_\\theta(\\mathbf{z}_t, a_t) d\\mathbf{X}_t
        $$
        
        où $\\mathbf{h}_t$ est un état de mémoire et $\\mathbf{X}_t$ un chemin contrôlé
        
        **Avantage** : Meilleure capture des dépendances temporelles longues
        
        ### Décodeur (Phase 2)
        
        **Rôle** : Reconstruit les observables depuis l'état latent
        
        **Architecture** :
        - MLP avec 2-3 couches
        - Entrée : État latent $\\mathbf{z}_t \\in \\mathbb{R}^{d_z}$
        - Sortie : Observables reconstruits $\\hat{\\mathbf{o}}_t = [\\hat{\\psi}_t, \\hat{S}_t, \\hat{R}_t, \\widehat{ET0}_t] \\in \\mathbb{R}^4$
        
        **Formule** :
        $$
        \\hat{\\mathbf{o}}_t = \\text{Decoder}(\\mathbf{z}_t) = \\text{MLP}(\\mathbf{z}_t)
        $$
        
        **Objectif** : Permettre des rollouts longs avec observables vérifiables
        """)
    
    with st.expander("📊 Variables et notations", expanded=False):
        st.markdown("""
        ### Variables principales
        
        **Observations et états** :
        - $\\mathbf{o}_t = [\\psi_t, S_t, R_t, ET0_t] \\in \\mathbb{R}^4$ : Observation au temps $t$
        - $\\mathbf{X}_{t-L:t} = [\\mathbf{o}_{t-L}, \\ldots, \\mathbf{o}_t] \\in \\mathbb{R}^{L \\times 4}$ : Historique de $L$ observations
        - $\\mathbf{z}_t \\in \\mathbb{R}^{d_z}$ : État latent au temps $t$
        - $a_t \\in [0, I_{\\max}]$ : Action (irrigation) au temps $t$
        - $r_t$ : Récompense au temps $t$
        
        **Architecture** :
        - $L$ : Longueur de séquence d'historique (30 par défaut)
        - $d_z$ : Dimension de l'espace latent (16 par défaut)
        - $H$ : Horizon d'imagination (5-30 selon phase)
        - $\\alpha$ : Paramètre de blend pour Phase 3 (0.3-0.7)
        
        **Fonctions et modèles** :
        - $\\text{Encoder}(\\cdot)$ : Encodeur PatchTST
        - $f_{\\text{transition}}(\\cdot)$ : Modèle de transition (Neural ODE ou CDE)
        - $\\text{Decoder}(\\cdot)$ : Décodeur (Phase 2 uniquement)
        - $f_{\\text{FAO}}(\\cdot)$ : Modèle physique FAO
        - $\\pi_\\theta(\\cdot | \\mathbf{z}_t)$ : Politique PPO
        
        **Entraînement** :
        - $N_{\\text{traj}}$ : Nombre de trajectoires pour pré-entraînement
        - $N_{\\text{epochs}}$ : Nombre d'epochs d'entraînement
        - $\\mathcal{L}_{\\text{transition}}$ : Perte du modèle de transition
        - $\\mathcal{L}_{\\text{decodeur}}$ : Perte du décodeur
        """)
    
    with st.expander("⚖️ Avantages et limitations", expanded=False):
        st.markdown("""
        ### ✅ Avantages du Scénario 6
        
        **1. Planification et anticipation** :
        - ✅ **Rollouts d'imagination** : Permet de simuler les conséquences des actions sur plusieurs pas de temps
        - ✅ **Anticipation** : L'agent peut prévoir l'effet de ses décisions avant de les exécuter
        - ✅ **Exploration efficace** : Teste différentes stratégies dans le modèle interne sans coût réel
        
        **2. Efficacité sample** :
        - ✅ **Moins d'interactions réelles** : Réduit le nombre d'épisodes nécessaires pour apprendre
        - ✅ **Apprentissage accéléré** : Les rollouts imaginaires enrichissent l'apprentissage
        - ✅ **Réutilisation** : Le World Model peut être réutilisé pour différents objectifs
        
        **3. Robustesse physique (Phase 3)** :
        - ✅ **Cohérence garantie** : L'hybridation avec le modèle physique limite les dérives
        - ✅ **Compromis réglable** : Le paramètre $\\alpha$ permet d'ajuster le blend
        - ✅ **Sécurité** : Le modèle physique corrige les prédictions aberrantes
        
        **4. Mémoire temporelle** :
        - ✅ **Dépendances longues** : PatchTST + Neural CDE capturent des patterns à long terme
        - ✅ **Tendances** : Comprend l'évolution saisonnière et les tendances
        - ✅ **Contexte riche** : L'historique complet est utilisé pour les décisions
        
        **5. Généralisation** :
        - ✅ **Compréhension profonde** : Le World Model apprend la dynamique sous-jacente
        - ✅ **Transfert** : Peut s'adapter à des conditions légèrement différentes
        - ✅ **Flexibilité** : Peut être utilisé pour différents objectifs sans ré-entraînement complet
        
        ### ⚠️ Limitations et défis
        
        **1. Complexité** :
        - ⚠️ **Architecture complexe** : Plusieurs composants (encodeur, transition, décodeur) à entraîner
        - ⚠️ **Hyperparamètres sensibles** : Nombreux hyperparamètres à régler (horizon, $\\alpha$, dimensions, etc.)
        - ⚠️ **Debugging difficile** : Plus difficile à déboguer qu'un modèle simple
        
        **2. Coût computationnel** :
        - ⚠️ **Pré-entraînement** : Nécessite un pré-entraînement de plusieurs composants
        - ⚠️ **Rollouts** : Génération de rollouts imaginaires ajoute du temps de calcul
        - ⚠️ **Mémoire** : Nécessite de stocker l'historique et les états latents
        
        **3. Dépendance aux données** :
        - ⚠️ **Qualité du pré-entraînement** : La qualité du World Model dépend de la qualité des trajectoires d'entraînement
        - ⚠️ **Couvre des distributions** : Si le pré-entraînement ne couvre pas toutes les situations, le modèle peut dériver
        - ⚠️ **Sim-to-real gap** : Écart entre données simulées et réelles peut affecter les performances
        
        **4. Risque de dérive** :
        - ⚠️ **Erreur cumulative** : Les erreurs dans les prédictions peuvent s'accumuler sur les rollouts longs
        - ⚠️ **Instabilité** : Sans Phase 3, le modèle peut s'éloigner de la physique si mal entraîné
        - ⚠️ **Mode collapse** : Le World Model peut apprendre des modes simplifiés qui ne capturent pas toute la complexité
        
        **5. Tuning délicat** :
        - ⚠️ **Horizon d'imagination** : Doit être choisi avec soin (trop court = peu de planification, trop long = instabilité)
        - ⚠️ **Paramètre $\\alpha$** : Le blend physique/World Model doit être ajusté selon les performances
        - ⚠️ **Dimensions** : Les dimensions de l'espace latent impactent la capacité et la complexité
        
        **6. Interprétabilité** :
        - ⚠️ **Espace latent abstrait** : L'espace latent n'est pas directement interprétable
        - ⚠️ **Black box** : Plus difficile de comprendre pourquoi le modèle prend certaines décisions
        - ⚠️ **Rollouts vérifiables** : Nécessite le décodeur (Phase 2+) pour vérifier la cohérence
        """)
    
    with st.expander("🔧 Paramètres recommandés et tuning", expanded=False):
        st.markdown("""
        ### Hyperparamètres par phase
        
        **Phase 1 - World Model Simple** :
        - **Horizon d'imagination** : 5-10 pas de temps
          - Trop court (< 5) : Peu de planification
          - Trop long (> 10) : Instabilité (pas de décodeur pour vérifier)
        - **Longueur de séquence PatchTST** : $L = 30$ (cohérent avec Scénario 5)
        - **Dimension latente** : $d_z = 16$ (dimension des features PatchTST)
        - **Learning rate transition** : $10^{-3}$ pour Neural ODE
        - **Nombre de trajectoires** : 32-50 pour pré-entraînement
        - **Nombre d'epochs** : 20-50 selon convergence
        
        **Phase 2 - World Model Complet** :
        - **Horizon d'imagination** : 20-30 pas de temps
          - Permet planification à long terme
          - Avec décodeur, peut vérifier la cohérence
        - **Longueur de séquence PatchTST** : $L = 30$
        - **Longueur de séquence CDE** : $L_{\\text{CDE}} = 8-12$ pour la mémoire
        - **Dimension latente** : $d_z = 16$
        - **Learning rate transition** : $10^{-3}$ pour Neural CDE
        - **Learning rate décodeur** : $10^{-3}$
        - **Nombre de trajectoires** : 50-100 pour pré-entraînement robuste
        
        **Phase 3 - Hybridation** :
        - **Paramètre $\\alpha$** : 0.3-0.6 (recommandé)
          - $\\alpha = 0.5$ : Équilibre
          - $\\alpha = 0.6-0.7$ : Privilégie physique (si World Model dérive)
          - $\\alpha = 0.3-0.4$ : Privilégie World Model (si bien entraîné)
        - **Autres paramètres** : Identiques à Phase 2
        
        **Hyperparamètres PPO** :
        - **Gamma** : 0.99 (discount factor élevé pour planification long terme)
        - **GAE lambda** : 0.95 (pour estimation de la valeur)
        - **Learning rate** : $3 \\times 10^{-4}$ (standard)
        - **Entropy coefficient** : 0.01-0.05 (contrôle exploration)
        - **Clip range** : 0.2 (pour PPO)
        - **Batch size** : 64-256
        - **Number of steps** : 2048 par rollout
        
        ### Stratégie de tuning
        
        **1. Commencer simple (Phase 1)** :
        - Valider le concept avec Phase 1 (moins de paramètres)
        - Vérifier que les rollouts courts sont cohérents
        - Évaluer si l'amélioration est significative
        
        **2. Étendre progressivement (Phase 2)** :
        - Si Phase 1 montre des bénéfices, passer à Phase 2
        - Augmenter l'horizon progressivement (10, 15, 20, 30)
        - Vérifier la reconstruction du décodeur
        
        **3. Stabiliser (Phase 3)** :
        - Si Phase 2 dérive, activer Phase 3
        - Commencer avec $\\alpha = 0.5$, ajuster selon résultats
        - Surveiller les métriques de stabilité
        
        **4. Optimisation fine** :
        - Ajuster $\\alpha$ selon la dérive observée
        - Réduire l'horizon si instabilité
        - Augmenter le nombre de trajectoires de pré-entraînement si qualité insuffisante
        """)
    
    with st.expander("🧭 Quand choisir le Scénario 6 ?", expanded=False):
        st.markdown("""
        ### Indicateurs pour choisir le Scénario 6
        
        **✅ Choisir le Scénario 6 si** :
        
        - **Besoin de planification** :
          - Vous voulez anticiper les effets des décisions sur plusieurs jours/semaines
          - La stratégie optimale nécessite de penser à long terme
          - Les autres scénarios montrent des limites sur la planification
        
        - **Données limitées en réel** :
          - Vous avez peu de données réelles mais pouvez générer des simulations
          - Le World Model permet de capitaliser sur un monde simulé
          - Sample efficiency est critique
        
        - **Besoin de robustesse** :
          - Vous voulez combiner l'efficacité du model-based RL avec la robustesse physique
          - Phase 3 permet de rester cohérent physiquement
          - Vous voulez un compromis entre innovation et sécurité
        
        - **Scénario 5 insuffisant** :
          - PatchTST seul (Scénario 5) ne capture pas assez les dynamiques longues
          - Vous voulez une planification plus explicite que l'enrichissement d'observation
          - Les features temporelles ne suffisent pas pour les décisions complexes
        
        - **Recherche/expérimentation** :
          - Vous explorez les approches model-based RL
          - Vous voulez comparer différents horizons de planification
          - Vous testez l'hybridation physique/neural
        
        **❌ Ne pas choisir le Scénario 6 si** :
        
        - **Simplicité recherchée** :
          - Vous voulez une solution simple et rapide à déployer
          - Le Scénario 2 ou 3 suffit pour vos besoins
        
        - **Temps de calcul limité** :
          - Le pré-entraînement et les rollouts sont trop coûteux
          - Vous avez besoin de résultats rapides
        
        - **Données abondantes** :
          - Vous avez beaucoup de données réelles et le sample efficiency n'est pas un problème
          - Model-free RL (Scénarios 2-5) suffit
        
        - **Pas de besoin de planification** :
          - Les décisions sont principalement réactives (court terme)
          - La planification à long terme n'apporte pas de bénéfice
        """)
    
    with st.expander("🛠️ Conseils pratiques et troubleshooting", expanded=False):
        st.markdown("""
        ### Workflow recommandé
        
        **1. Préparation** :
        - ✅ Vérifier la **cohérence météo** : Utiliser les mêmes seeds et paramètres que Scénario 1 pour comparaison
        - ✅ Pré-entraîner PatchTST dans Scénario 5 d'abord
        - ✅ S'assurer que les paramètres du sol sont cohérents
        
        **2. Démarrage progressif** :
        - ✅ Commencer avec **Phase 1** (simple) pour valider le concept
        - ✅ Utiliser des horizons courts (5-7) pour Phase 1
        - ✅ Vérifier que les rollouts sont cohérents
        
        **3. Extension** :
        - ✅ Si Phase 1 réussit, passer à **Phase 2** avec décodeur
        - ✅ Augmenter progressivement l'horizon (10, 15, 20, 30)
        - ✅ Surveiller la qualité de reconstruction du décodeur
        
        **4. Stabilisation** :
        - ✅ Si Phase 2 dérive, activer **Phase 3** avec hybridation
        - ✅ Commencer avec $\\alpha = 0.5$
        - ✅ Ajuster $\\alpha$ selon les résultats (augmenter si dérive, diminuer si stable)
        
        ### Troubleshooting
        
        **Problème : Rollouts incohérents**
        - **Symptôme** : Les rollouts imaginaires produisent des valeurs aberrantes
        - **Solutions** :
          - Réduire l'horizon d'imagination
          - Augmenter le nombre de trajectoires de pré-entraînement
          - Vérifier la qualité du modèle de transition (loss élevée = mauvais pré-entraînement)
          - Passer à Phase 3 pour limiter la dérive
        
        **Problème : Performance dégradée vs Scénario 5**
        - **Symptôme** : Le Scénario 6 ne performe pas mieux que le Scénario 5
        - **Solutions** :
          - Vérifier que l'horizon est adapté (pas trop court, pas trop long)
          - Augmenter le nombre de rollouts par pas
          - Vérifier que le World Model est bien pré-entraîné
          - Comparer les métriques détaillées (pas juste la récompense finale)
        
        **Problème : Dérive physique (Phase 2)**
        - **Symptôme** : Les prédictions s'éloignent de la physique
        - **Solutions** :
          - Activer Phase 3 avec $\\alpha$ élevé (0.6-0.7)
          - Augmenter le poids de la physique dans le blend
          - Vérifier le pré-entraînement du décodeur
          - Réduire l'horizon si trop long
        
        **Problème : Instabilité de l'entraînement PPO**
        - **Symptôme** : Les métriques PPO oscillent ou divergent
        - **Solutions** :
          - Réduire le learning rate
          - Réduire l'horizon d'imagination
          - Augmenter la stabilité du World Model (Phase 3)
          - Vérifier les hyperparamètres PPO (gamma, GAE lambda)
        
        **Problème : Temps d'entraînement trop long**
        - **Symptôme** : Le pré-entraînement prend trop de temps
        - **Solutions** :
          - Réduire le nombre de trajectoires (minimum 32)
          - Réduire le nombre d'epochs (minimum 20)
          - Utiliser un batch size plus grand si mémoire disponible
          - Commencer avec Phase 1 (plus rapide)
        
        ### Métriques à surveiller
        
        - **Loss du modèle de transition** : Doit décroître et converger (< 0.01 idéalement)
        - **Loss du décodeur** (Phase 2+) : Doit être faible pour reconstruction fiable
        - **Cohérence des rollouts** : Les valeurs doivent rester dans des plages réalistes
        - **Récompense moyenne** : Doit augmenter avec l'entraînement
        - **Longueur d'épisode** : Doit être stable
        - **Variance des actions** : Ne doit pas exploser (signe d'instabilité)
        """)
    
    with st.expander("🔗 Comparaison avec les autres scénarios", expanded=False):
        st.markdown("""
        ### Scénario 6 vs Scénario 2 (RL basique)
        
        **Scénario 2** :
        - Model-free RL direct sur modèle physique
        - Pas de planification explicite
        - Simple et rapide
        
        **Scénario 6** :
        - Model-based RL avec World Model
        - Planification via rollouts d'imagination
        - Plus complexe mais meilleure sample efficiency
        
        **Quand choisir Scénario 6** : Besoin de planification et sample efficiency
        
        ### Scénario 6 vs Scénario 3-4 (Neural ODE/CDE)
        
        **Scénarios 3-4** :
        - Correction résiduelle du modèle physique
        - Améliore la prédiction mais pas la planification
        - L'agent apprend directement dans l'environnement
        
        **Scénario 6** :
        - World Model séparé pour planification
        - Permet rollouts d'imagination
        - Architecture plus complexe
        
        **Quand choisir Scénario 6** : Besoin de planification explicite
        
        ### Scénario 6 vs Scénario 5 (PatchTST)
        
        **Scénario 5** :
        - PatchTST enrichit l'observation de l'agent
        - Model-free RL avec observation améliorée
        - Pas de rollouts d'imagination
        
        **Scénario 6** :
        - PatchTST comme encodeur du World Model
        - Model-based RL avec planification
        - Rollouts d'imagination pour améliorer l'apprentissage
        
        **Relation** : Scénario 6 réutilise PatchTST mais l'intègre dans une architecture model-based
        
        **Quand choisir Scénario 6** : Scénario 5 insuffisant pour planification à long terme
        
        ### Synthèse comparative
        
        | Critère | Scénario 2 | Scénario 3-4 | Scénario 5 | Scénario 6 |
        |---------|------------|--------------|------------|------------|
        | **Complexité** | Faible | Moyenne | Moyenne | Élevée |
        | **Planification** | ❌ | ❌ | ❌ | ✅ |
        | **Sample Efficiency** | Faible | Moyenne | Moyenne | **Élevée** |
        | **Robustesse physique** | ✅ | ✅ | ✅ | ✅ (Phase 3) |
        | **Temps d'entraînement** | Rapide | Moyen | Moyen | **Long** |
        | **Interprétabilité** | ✅ | ✅ | Moyenne | Faible |
        """)


def render_doc_scenarios():
    """
    Affiche le contenu de l'onglet de documentation : Scénarios.
    """
    st.markdown('<h2 class="section-header">📋 Les scénarios d\'étude : du modèle physique au jumeau numérique cognitif</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Présentation des scénarios et de leur principe général.
    """)
    
    st.markdown("### 🌱 Scénario 1 — Modèle physique + règle simple")
    st.markdown("""
    - **Principe**: utiliser un modèle bucket (bilan hydrique) et une règle fixe d'irrigation basée sur un seuil de $\\psi$.
    """)
    
    with st.expander("📋 Types de modèles physiques disponibles", expanded=False):
        st.markdown("""
        - **FAO**
          - Modèle de bilan hydrique simplifié inspiré de la méthodologie FAO-56
          - **Équations du modèle** :
            - Bilan hydrique : $S_{t+1} = S_t + \\eta_I I_t + R_t + G_t - ETc_t - D(S_t) - Q_t$
            - Courbe de rétention : $\\psi_{t+1} = f_{\\text{retention}}(S_{t+1})$
            - où $S_t$ est la réserve en eau (mm), $\\psi_t$ la tension matricielle (cbar), $I_t$ l'irrigation (mm), $R_t$ la pluie (mm), $G_t$ la remontée capillaire (mm), $ETc_t$ l'évapotranspiration culture (mm), $D(S_t)$ le drainage (mm), $Q_t$ le ruissellement (mm), et $\\eta_I$ l'efficacité d'irrigation
          - Utilise un modèle bucket (réservoir), le coefficient cultural $Kc$ et l'évapotranspiration de référence $ET0$
          - Bilan hydrique journalier avec courbe de rétention simplifiée
          - Adapté pour des applications pratiques et rapides
        
        - **HYDRUS**
          - Modèle sophistiqué résolvant l'équation de Richards (équation différentielle partielle)
          - Modélise le flux d'eau dans le sol en 1D, 2D ou 3D avec résolution numérique
          - Utilise des courbes de rétention complètes (van Genuchten)
          - Très précis mais complexe et coûteux en calcul
        
        - **Aquacrop**
          - Modèle FAO avancé incluant la croissance de la culture, le développement des racines, et des processus biologiques détaillés
          - Plus sophistiqué que le modèle bucket simple mais toujours basé sur les concepts FAO ($Kc$, $ET0$)
          - Adapté pour la modélisation complète du système culture-sol
        """)
    
    with st.expander("🔧 Modèle implémenté : FAO", expanded=False):
        st.markdown("""
        **Raisons du choix** :
        
        - **Simplicité et rapidité** : Le modèle bucket permet des calculs instantanés, essentiel pour l'apprentissage par renforcement qui nécessite de nombreuses simulations
        
        - **Concepts FAO standardisés** : Utilisation de $Kc$ et $ET0$ (méthodologie FAO-56) reconnus et validés internationalement
        
        - **Adéquation avec les observations** : Le modèle utilise directement la tension $\\psi_t$ mesurée par les tensiomètres, variable clé pour l'irrigation
        
        - **Efficacité computationnelle** : Pas de résolution d'équations différentielles complexes, permettant des milliers d'épisodes d'entraînement RL en temps raisonnable
        
        - **Compromis précision/complexité** : Suffisamment précis pour capturer la dynamique essentielle du bilan hydrique tout en restant simple à implémenter et calibrer
        
        - **Compatibilité RL** : La structure simple du modèle bucket facilite l'intégration avec les algorithmes d'apprentissage par renforcement
        """)
    
    with st.expander("⚖️ Avantages et limitations", expanded=False):
        st.markdown("""
        ### ✅ Avantages du Scénario 1
        
        - **Simplicité** : 
          - Implémentation directe : règles basées sur des seuils de tension ($\\psi$) faciles à comprendre
          - Pas besoin d'entraînement : règles définies manuellement, pas de phase d'apprentissage
          - Déploiement immédiat : peut être mis en place rapidement sans infrastructure complexe
        
        - **Interprétabilité** : 
          - Logique transparente : "si $\\psi_t > \\psi_{\\text{seuil}}$, alors irriguer $I_{\\text{fixe}}$"
          - Facile à expliquer aux agriculteurs : règles compréhensibles sans expertise en IA
          - Débogage simple : comportement prévisible et traçable
        
        - **Rapidité d'exécution** : 
          - Calculs instantanés : évaluation de la règle en temps constant
          - Pas de calculs lourds : pas de réseau de neurones à évaluer
          - Adapté aux systèmes embarqués : faible consommation de ressources
        
        - **Robustesse** : 
          - Comportement stable : pas de variabilité due à l'apprentissage
          - Pas de sur-apprentissage : règles fixes garantissent un comportement cohérent
          - Prévisible : résultats reproductibles pour les mêmes conditions
        
        - **Coût faible** : 
          - Pas d'infrastructure d'entraînement nécessaire
          - Maintenance minimale : règles simples à maintenir
          - Pas de données d'entraînement requises
        
        ### ⚠️ Limitations du Scénario 1
        
        - **Manque d'adaptabilité** : 
          - Conditions météorologiques : ne prend pas en compte les prévisions de pluie ou d'ET0 de manière optimale
          - Variabilité saisonnière : seuils fixes ne s'adaptent pas aux changements de saison
          - Conditions locales : règles génériques qui ne s'adaptent pas aux spécificités de chaque parcelle
        
        - **Pas d'optimisation** : 
          - Efficacité de l'eau : ne minimise pas nécessairement la consommation d'eau
          - Équilibre stress/coût : ne trouve pas le compromis optimal entre stress hydrique et coût de l'eau
          - Pas d'apprentissage : ne s'améliore pas avec l'expérience
        
        - **Rigidité des règles** : 
          - Seuils fixes : ne s'adaptent pas aux variations de conditions
          - Doses fixes : irrigation toujours de la même quantité, sans gradation fine
          - Pas de stratégie préventive : réagit seulement quand le seuil est dépassé, pas de prévision
        
        - **Performance sous-optimale** : 
          - Gaspillage potentiel : peut irriguer même si la pluie est imminente
          - Stress hydrique : peut laisser le sol se dessécher avant d'intervenir
          - Drainage excessif : peut provoquer des pertes d'eau par drainage si irrigation mal calibrée
        
        - **Maintenance manuelle** : 
          - Calibration nécessaire : les seuils doivent être ajustés manuellement selon les conditions
          - Pas d'auto-ajustement : nécessite une intervention humaine pour optimiser les paramètres
          - Expertise requise : besoin de connaissances agronomiques pour définir les bons seuils
        """)
    
    with st.expander("🔧 Paramètres recommandés et tuning", expanded=False):
        st.markdown("""
        ### Paramètres des règles d'irrigation
        
        **Règle à seuil unique** :
        - **Seuil de tension** ($\\psi_{\\text{seuil}}$) : 40-60 cbar (recommandé)
          - Trop bas (< 30) : Irrigation trop fréquente, gaspillage d'eau
          - Trop élevé (> 80) : Stress hydrique important avant irrigation
        - **Dose d'irrigation** : 10-20 mm (recommandé)
          - Doit être adaptée à la capacité du sol et à la culture
        
        **Règle à bande de confort** :
        - **Tension minimale** ($\\psi_{\\min}$) : 20-30 cbar
        - **Tension maximale** ($\\psi_{\\max}$) : 50-70 cbar
        - **Dose proportionnelle** : Ajustée selon l'écart à la zone de confort
        
        **Règle proportionnelle** :
        - **Coefficient de proportionnalité** ($k_I$) : 0.1-0.3
          - Plus élevé = irrigation plus agressive
          - Plus faible = irrigation plus conservatrice
        
        ### Tuning recommandé
        
        **Étape 1 : Calibration initiale** :
        - Commencer avec des valeurs standard (seuil = 50 cbar, dose = 15 mm)
        - Observer le comportement sur une saison complète
        
        **Étape 2 : Ajustement selon résultats** :
        - Si stress hydrique fréquent : Réduire le seuil ou augmenter la dose
        - Si gaspillage d'eau : Augmenter le seuil ou réduire la dose
        - Si drainage excessif : Réduire la dose
        
        **Étape 3 : Ajustement saisonnier** :
        - Adapter les seuils selon la phase de croissance (Kc variable)
        - Tenir compte des prévisions météorologiques
        """)
    
    with st.expander("🧭 Quand utiliser le Scénario 1 ?", expanded=False):
        st.markdown("""
        ### ✅ Choisir le Scénario 1 si :
        
        - **Simplicité recherchée** :
          - Vous voulez une solution simple et rapide à déployer
          - Pas d'infrastructure d'entraînement disponible
          - Besoin de résultats immédiats sans phase d'apprentissage
        
        - **Interprétabilité importante** :
          - Les règles doivent être compréhensibles par les utilisateurs finaux
          - Besoin d'expliquer facilement les décisions prises
          - Conformité réglementaire nécessitant de la transparence
        
        - **Ressources limitées** :
          - Pas de ressources computationnelles pour l'entraînement RL
          - Pas de données historiques suffisantes
          - Système embarqué avec contraintes de calcul
        
        - **Conditions stables** :
          - Conditions météorologiques et pédologiques relativement stables
          - Pas de variabilité saisonnière importante
          - Parcelle bien caractérisée avec paramètres connus
        
        - **Baseline de référence** :
          - Point de départ pour comparer avec d'autres approches
          - Validation du modèle physique avant d'ajouter de la complexité
        
        ### ❌ Ne pas choisir le Scénario 1 si :
        
        - **Optimisation nécessaire** :
          - Besoin de minimiser la consommation d'eau
          - Recherche du compromis optimal stress/coût
          - Conditions variables nécessitant adaptation
        
        - **Données disponibles** :
          - Vous avez des données historiques pour entraîner un modèle RL
          - Possibilité de générer des simulations pour l'entraînement
        
        - **Performance maximale recherchée** :
          - Les règles simples ne suffisent pas pour vos objectifs
          - Besoin d'une stratégie adaptative et optimisée
        """)
    
    with st.expander("🛠️ Conseils pratiques", expanded=False):
        st.markdown("""
        ### Workflow recommandé
        
        **1. Calibration initiale** :
        - Utiliser les valeurs par défaut comme point de départ
        - Observer le comportement sur une saison complète
        - Documenter les performances (stress, consommation d'eau, drainage)
        
        **2. Ajustement itératif** :
        - Ajuster un paramètre à la fois (seuil OU dose, pas les deux)
        - Tester sur plusieurs saisons avec conditions variées
        - Comparer les métriques avant/après ajustement
        
        **3. Validation** :
        - Vérifier la cohérence avec les connaissances agronomiques
        - Comparer avec les pratiques locales
        - Valider sur différentes conditions météorologiques
        
        ### Troubleshooting
        
        **Problème : Stress hydrique fréquent**
        - **Solution** : Réduire le seuil de tension (ex: 50 → 40 cbar) ou augmenter la dose
        
        **Problème : Gaspillage d'eau**
        - **Solution** : Augmenter le seuil (ex: 50 → 60 cbar) ou réduire la dose
        
        **Problème : Drainage excessif**
        - **Solution** : Réduire la dose d'irrigation ou augmenter le seuil
        
        **Problème : Irrigation trop tardive**
        - **Solution** : Utiliser une règle préventive ou réduire le seuil
        """)
    
    with st.expander("🔗 Comparaison avec les autres scénarios", expanded=False):
        st.markdown("""
        ### Scénario 1 vs Scénario 2 (RL basique)
        
        **Scénario 1** :
        - Règles fixes, pas d'apprentissage
        - Simple et rapide
        - Performance sous-optimale
        
        **Scénario 2** :
        - Apprentissage automatique de la politique
        - Plus complexe mais meilleure performance
        - Nécessite entraînement
        
        **Quand choisir Scénario 1** : Simplicité et rapidité prioritaires
        
        ### Scénario 1 vs Scénarios 3-6
        
        **Scénario 1** :
        - Baseline simple
        - Pas d'optimisation
        - Pas d'adaptation
        
        **Scénarios 3-6** :
        - Approches avancées avec apprentissage
        - Optimisation et adaptation
        - Meilleure performance mais plus complexe
        
        **Relation** : Scénario 1 sert de référence pour évaluer les gains des autres scénarios
        """)
    
    st.markdown("### 🎓 Scénario 2 — RL sur modèle physique (avec $\\psi_t$ observée)")
    st.markdown("""
    - **Principe**: un agent RL observe $\\psi_t$ (et le contexte météo) et choisit $I_t$ dans un environnement simulé par le modèle physique.
      - **Espace d'observation**: $o_t = (\\psi_t,\\ t/T,\\ R_{t-k:t},\\ ET0_t,\\ \\hat R_{t+1:t+h},\\ \\widehat{ET0}_{t+1:t+h})$
      - **Espace d'actions**: $I_t \\in [0,\\ I_{\\max}]$ (mm) - continu
      - **Récompense**: $r_t = -\\alpha\\,\\text{stress}(\\psi_t) - \\beta\\, I_t - \\gamma\\, D(S_t)$
      - **Algorithme**: PPO (Proximal Policy Optimization)
    """)
    
    with st.expander("⚖️ Avantages et limitations", expanded=False):
        st.markdown("""
        ### ✅ Avantages du Scénario 2
        
        - **Apprentissage d'une politique optimale** : 
          - Optimisation automatique : l'agent RL apprend à minimiser le stress hydrique tout en économisant l'eau
          - Compromis optimal : trouve automatiquement le meilleur équilibre entre performance agronomique et coût de l'eau
          - Stratégie adaptative : s'ajuste selon les conditions météorologiques et l'état du sol
        
        - **Adaptabilité aux conditions** : 
          - Prévisions météo : utilise les prévisions de pluie et d'ET0 pour anticiper et ajuster l'irrigation
          - Conditions variables : s'adapte aux variations saisonnières et aux événements météorologiques
          - Historique contextuel : prend en compte l'historique récent (pluie, tension) pour des décisions informées
        
        - **Respect de la physique** : 
          - Modèle physique fiable : utilise un modèle bucket validé pour simuler la dynamique du sol
          - Courbe de rétention : respecte la relation $S \\leftrightarrow \\psi$ basée sur les propriétés pédophysiques
          - Bilan hydrique cohérent : les équations physiques garantissent la cohérence des prédictions
        
        - **Flexibilité des actions** : 
          - Actions continues : permet des doses d'irrigation précises et graduées (pas seulement 0 ou dose fixe)
          - Doses adaptatives : ajuste la quantité d'eau selon l'intensité du stress et les conditions
          - Stratégie préventive : peut irriguer préventivement avant que le stress ne devienne critique
        
        - **Performance supérieure** : 
          - Efficacité de l'eau : généralement meilleure que les règles fixes en termes de consommation d'eau
          - Réduction du stress : maintient mieux la tension dans la zone de confort
          - Minimisation du drainage : apprend à éviter les pertes d'eau par drainage excessif
        
        - **Réutilisabilité** : 
          - Modèle entraîné : une fois entraîné, le modèle peut être utilisé sur différentes saisons
          - Transfert possible : peut être adapté à d'autres parcelles avec ré-entraînement
          - Amélioration continue : peut être ré-entraîné avec de nouvelles données pour s'améliorer
        
        ### ⚠️ Limitations du Scénario 2
        
        - **Dépendance à la qualité du modèle physique** : 
          - Biais du modèle : si le modèle bucket a des biais (paramètres mal calibrés, processus négligés), 
            la politique apprise sera biaisée
          - Erreurs de paramétrisation : des erreurs dans les paramètres du sol ($S_{fc}$, $\\psi_{fc}$, $k_d$) 
            se propagent dans les décisions
          - Processus non modélisés : phénomènes non capturés par le modèle (hétérogénéité spatiale, 
            interactions complexes) ne sont pas pris en compte
        
        - **Phase d'entraînement nécessaire** : 
          - Temps d'entraînement : nécessite une phase d'apprentissage (plusieurs milliers de timesteps) 
            avant d'être utilisable
          - Ressources computationnelles : entraînement PPO nécessite des ressources CPU/GPU
          - Expertise technique : nécessite des compétences en RL pour l'entraînement et le réglage
        
        - **Données d'entraînement** : 
          - Simulation requise : besoin de générer des données de simulation pour l'entraînement
          - Qualité de la simulation : la qualité de l'entraînement dépend de la qualité de la simulation météo
          - Robustesse : nécessite d'entraîner sur plusieurs saisons/scénarios pour être robuste
        
        - **Complexité de déploiement** : 
          - Infrastructure : nécessite une infrastructure pour exécuter le modèle entraîné
          - Maintenance : le modèle peut nécessiter un ré-entraînement périodique
          - Interprétabilité réduite : moins interprétable que les règles simples (boîte noire)
        
        - **Hyperparamètres à régler** : 
          - Tuning nécessaire : nombreux hyperparamètres à ajuster (learning rate, gamma, GAE-$\\lambda$, etc.)
          - Sensibilité : la performance peut être sensible aux choix d'hyperparamètres
          - Expertise requise : nécessite une compréhension du RL pour optimiser les hyperparamètres
        
        - **Stabilité de l'apprentissage** : 
          - Convergence : l'entraînement peut ne pas converger ou converger vers un optimum local
          - Variabilité : la performance peut varier entre différentes exécutions d'entraînement
          - Normalisation : nécessite une normalisation soigneuse des observations et récompenses
        
        - **Observations cohérentes** : 
          - Alignement temporel : nécessite que les observations soient alignées temporellement
          - Données manquantes : doit gérer les cas de données manquantes ou irrégulières
          - Prévisions météo : dépend de la qualité des prévisions météorologiques disponibles
        """)
    
    with st.expander("🔧 Paramètres recommandés et tuning", expanded=False):
        st.markdown("""
        ### Hyperparamètres PPO
        
        **Learning rate** : $3 \\times 10^{-4}$ (recommandé)
        - Trop élevé (> $10^{-3}$) : Instabilité, oscillations
        - Trop faible (< $10^{-5}$) : Apprentissage trop lent
        - Tuning : Réduire si loss oscille, augmenter si convergence lente
        
        **Gamma (discount factor)** : 0.99 (recommandé)
        - Contrôle l'importance des récompenses futures
        - Élevé (0.99) : Planification à long terme
        - Faible (0.95) : Focus sur court terme
        
        **GAE lambda** : 0.95 (recommandé)
        - Contrôle le biais/variance de l'estimation de la valeur
        - Élevé (0.95-0.99) : Moins de variance, plus de biais
        - Faible (0.8-0.9) : Plus de variance, moins de biais
        
        **Entropy coefficient** : 0.01-0.05
        - Encourage l'exploration
        - Élevé : Plus d'exploration, convergence plus lente
        - Faible : Moins d'exploration, risque de sous-optimum local
        
        **Clip range** : 0.2 (standard PPO)
        - Limite les changements de politique
        - Élevé : Permet plus de changements, moins stable
        - Faible : Changements limités, plus stable
        
        **Batch size** : 64-256
        - Plus grand : Gradients plus stables mais plus de mémoire
        - Plus petit : Moins de mémoire mais gradients plus variables
        
        **Number of steps per rollout** : 2048
        - Plus grand : Meilleure estimation mais plus de mémoire
        - Plus petit : Moins de mémoire mais estimation moins précise
        
        ### Hyperparamètres de l'environnement
        
        **Paramètres de récompense** :
        - $\\alpha$ (pénalité stress) : 1.0 (recommandé)
        - $\\beta$ (pénalité irrigation) : 0.05 (recommandé)
        - $\\gamma$ (pénalité drainage) : 0.01 (recommandé)
        - Tuning : Ajuster selon priorités (eau vs stress)
        
        **Paramètres du sol** :
        - Utiliser les valeurs par défaut sauf si données spécifiques disponibles
        - Calibrer $S_{fc}$, $\\psi_{fc}$ selon mesures réelles si possible
        
        ### Stratégie de tuning
        
        **1. Commencer avec valeurs par défaut** :
        - Utiliser les valeurs recommandées ci-dessus
        - Entraîner sur 50,000-100,000 timesteps
        
        **2. Observer les métriques** :
        - Récompense moyenne : Doit augmenter
        - Longueur d'épisode : Doit être stable
        - Variance des actions : Ne doit pas exploser
        
        **3. Ajuster si nécessaire** :
        - Si instabilité : Réduire learning rate, augmenter clip range
        - Si convergence lente : Augmenter learning rate, réduire entropy
        - Si sous-optimum : Augmenter entropy, réduire clip range
        """)
    
    with st.expander("🧭 Quand utiliser le Scénario 2 ?", expanded=False):
        st.markdown("""
        ### ✅ Choisir le Scénario 2 si :
        
        - **Optimisation recherchée** :
          - Besoin de minimiser la consommation d'eau
          - Recherche du compromis optimal stress/coût
          - Performance supérieure aux règles simples
        
        - **Données disponibles** :
          - Possibilité de générer des simulations pour l'entraînement
          - Modèle physique fiable et bien calibré
          - Conditions météorologiques variées pour robustesse
        
        - **Ressources computationnelles** :
          - Infrastructure disponible pour l'entraînement PPO
          - Temps d'entraînement acceptable (quelques heures)
          - Expertise en RL disponible
        
        - **Adaptabilité nécessaire** :
          - Conditions variables nécessitant adaptation
          - Besoin de stratégie préventive
          - Optimisation selon objectifs multiples
        
        - **Point de départ pour approches avancées** :
          - Baseline pour comparer avec Scénarios 3-6
          - Validation de l'approche RL avant complexification
        
        ### ❌ Ne pas choisir le Scénario 2 si :
        
        - **Simplicité prioritaire** :
          - Besoin de solution simple et rapide
          - Pas d'infrastructure d'entraînement
          - Règles simples suffisent
        
        - **Modèle physique incertain** :
          - Paramètres du sol mal connus
          - Modèle physique non validé
          - Données de simulation de mauvaise qualité
        
        - **Données limitées** :
          - Pas de possibilité de générer des simulations
          - Conditions trop spécifiques pour généraliser
        
        - **Besoin de correction physique** :
          - Modèle physique a des biais connus
          - Nécessité de corriger les prédictions physiques
          - → Préférer Scénarios 3-4
        """)
    
    with st.expander("🛠️ Conseils pratiques", expanded=False):
        st.markdown("""
        ### Workflow recommandé
        
        **1. Préparation** :
        - Vérifier la cohérence météo (mêmes seeds/params que Scénario 1)
        - Valider le modèle physique sur quelques épisodes
        - Configurer les hyperparamètres avec valeurs par défaut
        
        **2. Entraînement initial** :
        - Commencer avec 50,000 timesteps
        - Observer les métriques (récompense, longueur d'épisode)
        - Vérifier la convergence
        
        **3. Tuning itératif** :
        - Ajuster les hyperparamètres si nécessaire
        - Ré-entraîner avec nouveaux paramètres
        - Comparer les performances
        
        **4. Évaluation** :
        - Tester sur nouvelles saisons (seeds différents)
        - Comparer avec Scénario 1 (baseline)
        - Analyser les décisions prises
        
        ### Troubleshooting
        
        **Problème : Instabilité de l'entraînement**
        - **Symptôme** : Loss oscille, récompense ne converge pas
        - **Solutions** :
          - Réduire learning rate (ex: $3 \\times 10^{-4} \\to 10^{-4}$)
          - Augmenter clip range (ex: 0.2 → 0.3)
          - Normaliser les observations et récompenses
        
        **Problème : Convergence lente**
        - **Symptôme** : Récompense augmente très lentement
        - **Solutions** :
          - Augmenter learning rate (avec prudence)
          - Augmenter entropy coefficient pour plus d'exploration
          - Vérifier la normalisation des récompenses
        
        **Problème : Sous-optimum local**
        - **Symptôme** : Performance plafonne à un niveau sous-optimal
        - **Solutions** :
          - Augmenter entropy coefficient
          - Réduire clip range pour permettre plus de changements
          - Augmenter le nombre de timesteps d'entraînement
        
        **Problème : Politique trop conservatrice**
        - **Symptôme** : Irrigation insuffisante, stress hydrique
        - **Solutions** :
          - Ajuster les poids de récompense ($\\alpha$ vs $\\beta$)
          - Augmenter la pénalité de stress ($\\alpha$)
          - Réduire la pénalité d'irrigation ($\\beta$)
        
        ### Métriques à surveiller
        
        - **Récompense moyenne** : Doit augmenter avec l'entraînement
        - **Longueur d'épisode** : Doit être stable (≈ longueur de saison)
        - **Variance des actions** : Ne doit pas exploser (signe d'instabilité)
        - **Policy loss** : Doit décroître et converger
        - **Value loss** : Doit décroître (estimation de la valeur)
        """)
    
    with st.expander("🔗 Comparaison avec les autres scénarios", expanded=False):
        st.markdown("""
        ### Scénario 2 vs Scénario 1 (Règles simples)
        
        **Scénario 1** :
        - Règles fixes, pas d'apprentissage
        - Simple et rapide
        - Performance sous-optimale
        
        **Scénario 2** :
        - Apprentissage automatique
        - Plus complexe mais meilleure performance
        - Nécessite entraînement
        
        **Quand choisir Scénario 2** : Optimisation et adaptabilité recherchées
        
        ### Scénario 2 vs Scénarios 3-4 (Neural ODE/CDE)
        
        **Scénario 2** :
        - RL direct sur modèle physique
        - Pas de correction du modèle physique
        - Plus simple
        
        **Scénarios 3-4** :
        - Correction résiduelle du modèle physique
        - Améliore la prédiction physique
        - Plus complexe
        
        **Quand choisir Scénarios 3-4** : Modèle physique a des biais connus
        
        ### Scénario 2 vs Scénario 5 (PatchTST)
        
        **Scénario 2** :
        - Observation standard (4 dimensions)
        - Pas de mémoire temporelle explicite
        
        **Scénario 5** :
        - Observation enrichie avec features temporelles
        - Mémoire longue via PatchTST
        
        **Quand choisir Scénario 5** : Besoin de comprendre tendances et saisonnalité
        
        ### Scénario 2 vs Scénario 6 (World Model)
        
        **Scénario 2** :
        - Model-free RL
        - Pas de planification explicite
        
        **Scénario 6** :
        - Model-based RL avec planification
        - Rollouts d'imagination
        
        **Quand choisir Scénario 6** : Besoin de planification et sample efficiency
        """)
    
    st.markdown("### 🔬 Scénario 3 — RL sur modèle hybride Physique + Neural ODE")
    st.markdown("""
    - **Principe**: corriger la prédiction physique de $\\psi_{t+1}$ par une correction neuronale locale $\\Delta \\psi$ 
      apprise à partir de $(\\psi_t, I_t, R_t, ET0_t)$.
    """)
    
    with st.expander("⚖️ Avantages et limitations", expanded=False):
        st.markdown("""
        ### ✅ Avantages du Neural ODE
        
        - **Combinaison physique et données** : Le Neural ODE (Ordinary Differential Equation) combine le meilleur des deux mondes :
          
          - **Modèle physique comme base** : Le modèle bucket fournit une prédiction initiale $\\psi_{t+1}^{phys}$ basée sur 
            les lois physiques connues (bilan hydrique, courbe de rétention). Cette base garantit que les prédictions 
            respectent les principes fondamentaux de la dynamique du sol.
          
          - **Correction neuronale adaptative** : Un réseau de neurones apprend une correction locale $\\Delta \\psi$ qui 
            ajuste la prédiction physique en fonction des données observées. Cette correction capture les phénomènes 
            non modélisés ou mal paramétrés dans le modèle physique.
        
        - **Correction des biais du modèle** : Le Neural ODE peut corriger plusieurs types de biais :
          
          - **Biais de paramétrisation** : Si les paramètres du sol (par exemple $S_{fc}$, $\\psi_{fc}$, $k_d$) sont mal 
            estimés ou varient dans l'espace, le Neural ODE apprend à compenser ces erreurs.
          
          - **Phénomènes non modélisés** : Certains processus physiques peuvent être négligés ou simplifiés dans le modèle 
            bucket (par exemple, hétérogénéité spatiale, effets de la structure du sol, interactions racines-sol complexes). 
            Le Neural ODE peut apprendre à capturer ces effets à partir des données.
          
          - **Erreurs de mesure** : Les capteurs peuvent avoir des biais systématiques ou des erreurs de calibration. 
            Le Neural ODE peut apprendre à les corriger si ces erreurs sont cohérentes dans le temps.
        
        - **Apprentissage continu** : Contrairement aux modèles purement physiques qui sont statiques, le Neural ODE 
          peut être ré-entraîné avec de nouvelles données pour s'adapter à l'évolution des conditions (par exemple, 
          changement de structure du sol, vieillissement des capteurs).
        
        - **Interprétabilité** : La structure hybride permet de séparer la contribution du modèle physique (interprétable) 
          de la correction neuronale (qui peut être analysée pour comprendre quels phénomènes sont mal capturés).
        
        - **Efficacité computationnelle** : En utilisant le modèle physique comme base, le Neural ODE nécessite moins 
          de données et d'entraînement qu'un modèle purement neuronal, tout en étant plus performant qu'un modèle 
          purement physique.
        
        ### ⚠️ Limitations du Neural ODE
        
        - **Données nécessaires** : Nécessite des données réelles pour l'apprentissage du réseau de correction. 
          Plus les données sont nombreuses et représentatives, meilleure sera la correction.
        
        - **Hypothèse de régularité** : Le Neural ODE suppose généralement des observations à intervalles réguliers. 
          Les données manquantes ou irrégulières nécessitent un pré-traitement (interpolation, imputation).
        
        - **Complexité** : Plus complexe qu'un modèle purement physique, nécessite une expertise en deep learning 
          pour l'entraînement et le réglage des hyperparamètres.
        
        - **Dépendance aux données d'entraînement** : La qualité de la correction dépend de la représentativité 
          des données d'entraînement. Si les conditions changent significativement (nouveau type de sol, nouvelle culture), 
          le modèle peut nécessiter un ré-entraînement.
        """)
    
    st.markdown("### 🧠 Scénario 4 — RL sur modèle hybride Physique + Neural CDE")
    st.markdown("""
    - **Principe**: exploiter des trajectoires (possiblement irrégulières) $[\\psi, I, R, ET0]$ via un état latent (CDE) 
      pour produire une correction temporelle cohérente.
    """)
    
    with st.expander("⚖️ Avantages et limitations", expanded=False):
        st.markdown("""
        ### ✅ Avantages du Neural CDE
        
        - **Gestion des données irrégulières** : Le Neural CDE (Controlled Differential Equation) est particulièrement 
          adapté aux séries temporelles avec des observations à intervalles irréguliers. En pratique, cela signifie :
          
          - **Fiabilité des capteurs** : Les tensiomètres et autres capteurs peuvent avoir des pannes temporaires, 
            des défaillances de communication, ou nécessiter des calibrations périodiques. Le Neural CDE peut 
            gérer ces périodes de données manquantes sans nécessiter d'interpolation artificielle.
          
          - **Fréquence d'échantillonnage variable** : Contrairement aux modèles classiques qui supposent des mesures 
            à intervalles réguliers (par exemple, toutes les heures ou tous les jours), le Neural CDE peut traiter 
            des données qui arrivent à des moments différents (ex. : mesure à 8h un jour, à 10h le lendemain, puis 
            aucune mesure pendant 2 jours).
          
          - **Robustesse aux pannes** : Si un capteur tombe en panne pendant plusieurs jours, le modèle peut continuer 
            à fonctionner en utilisant les dernières observations valides et en extrapolant de manière cohérente grâce 
            à l'état latent du CDE.
        
        - **Meilleure modélisation temporelle** : Le CDE modélise explicitement l'évolution continue du système, 
          ce qui permet une meilleure compréhension de la dynamique du sol entre les observations.
        
        - **Adaptation aux contraintes opérationnelles** : Dans un contexte réel, les mesures peuvent être prises 
          à des moments opportuns (visites de terrain, maintenance), pas nécessairement à intervalles fixes.
        
        ### ⚠️ Limitations du Neural CDE
        
        - **Complexité** : Plus complexe à implémenter et à entraîner que les modèles précédents
        - **Données nécessaires** : Nécessite plus de données pour l'apprentissage, notamment pour calibrer 
          l'état latent du CDE
        - **Temps de calcul** : Généralement plus coûteux en temps de calcul que les approches plus simples
        """)

    st.markdown("### 🔮 Scénario 5 — RL + PatchTST (features temporelles)")
    st.markdown("""
    - **Principe** : utiliser PatchTST comme encodeur de séquences $[\\psi, I, R, ET0]$ pour enrichir l'observation de l'agent.
    - **Rôle** : pas de correction physique, mais un meilleur contexte temporel pour l'agent RL.
    - **Pipeline** : pré-entraînement auto-supervisé sur données simulées → wrapper d'environnement qui concatène les features.
    - **Quand l'utiliser** : besoin de mémoire longue (tendance/seasonality) sans toucher au modèle physique.
    """)

    st.markdown("### 🌍 Scénario 6 — World Model (model-based RL)")
    st.markdown("""
    - **Principe** : apprendre un modèle du monde (PatchTST + transition ODE/CDE + décodeur) pour faire des rollouts d'imagination.
    - **Phases** :
        - **Phase 1** : Transition ODE, rollouts courts.
        - **Phase 2** : Transition CDE + décodeur, rollouts longs.
        - **Phase 3** : Physics-informed (blend world model + modèle physique).
    - **Objectif** : planification, efficacité sample et robustesse via hybridation.
    """)

    st.markdown("### 📊 Comparaison des scénarios")
    st.markdown("""
    | Scénario | Complexité | Adaptabilité | Données nécessaires | Performance attendue |
    |----------|------------|--------------|---------------------|----------------------|
    | 1. Règle simple | Faible | Faible | Aucune | Basique |
    | 2. RL physique | Moyenne | Élevée | Simulation | Bonne |
    | 3. RL + Neural ODE | Élevée | Très élevée | Réelles + Simulation | Très bonne |
    | 4. RL + Neural CDE | Très élevée | Très élevée | Réelles + Simulation | Excellente |
    | 5. RL + PatchTST | Élevée | Élevée | Simulation (pré-train) | Très bonne (contexte temporel) |
    | 6. World Model (Phases 1-3) | Très élevée | Très élevée | Simulation + pré-train | Excellente (planification) |
    """)
    
