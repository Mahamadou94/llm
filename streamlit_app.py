import streamlit as st
import pandas as pd
import asyncio
import base64
import re
import matplotlib.pyplot as plt
from datetime import datetime
from config_parser import Parser
from pymongo import MongoClient, ASCENDING
from llm import LLM

# 📌 Charger la configuration MongoDB
@st.cache_resource
def get_db_connection():
    config = Parser("config.ini")
    client = MongoClient(config.Mongo['connexion_string'])
    db = client[config.Mongo['data_base']]
    collection = db[config.Mongo['collection']]
    return collection, config

collection, config = get_db_connection()

# 📌 Fonction pour récupérer les avis clients depuis MongoDB
def fetch_documents():
    return list(collection.find().sort('timestamp', ASCENDING))

# 📌 Appliquer une image de fond à la sidebar uniquement
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    bg_css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/webp;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .title-box {{
        background-color: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }}
    .title-box h1 {{
        color: black;
        font-size: 40px;
        font-weight: bold;
    }}
    .title-box p {{
        font-size: 18px;
        color: black;
    }}
    .button-container {{
        text-align: center;
        margin-top: 20px;
    }}
    .stButton>button {{
        font-size: 18px;
        padding: 10px 20px;
        border-radius: 10px;
        background-color: #4CAF50;
        color: white;
        border: none;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
    }}
    .stButton>button:hover {{
        background-color: #45a049;
    }}
    .result-container {{
        background: white; 
        color: black;
        padding: 20px;
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        line-height: 1.5;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }}
    </style>
    """
    st.markdown(bg_css, unsafe_allow_html=True)

# 📌 Charger le background uniquement pour la sidebar et non pour les résultats
set_background("/home/mahamadou/mlops_project/static/image2.webp")  

# 📌 Interface utilisateur avec Streamlit (Titre + Bouton de Lancement)
st.markdown("<div class='title-box'><h1>📊 Rapport d'analyse de sentiments d'avis clients </h1><p>Instruments de musique achétés sur Amazon</p></div>", unsafe_allow_html=True)

# 📌 Stocker les résultats pour éviter de relancer l'analyse
if "results" not in st.session_state:
    st.session_state["results"] = {}

# 📌 Barre latérale avec options de filtrage
st.sidebar.title("🔍 Options de Visualisation")

if st.session_state["results"]:
    chunk_keys = list(st.session_state["results"].keys())
    selected_chunk = st.sidebar.selectbox("Sélectionnez un chunk :", chunk_keys, index=0)
else:
    selected_chunk = None

# 📌 Options de filtrage des résultats
filter_option = st.sidebar.radio("Sélectionnez une section :", 
    ["Tout afficher", "Synthèse Globale", "Problèmes Clés", "Points Forts", "Analyse Détaillée", "Benchmark Concurrentiel", "Recommandations Stratégiques"])

# 📌 Progression en temps réel
progress_bar = st.progress(0)
status_text = st.empty()

# 📌 Fonction pour traiter les documents
async def process_documents():
    documents = fetch_documents()  # Prendre toute la data
    texts = [doc['input_text'] for doc in documents]
    texts = texts[5]
    chunks = [texts[i:i + 500] for i in range(0, len(texts), 500)]  

    llm = LLM(config, {'system_prompt': 'system_prompt.txt', 'user_prompt': 'user_prompt.txt'}, chunks)

    results = {}
    total_chunks = len(chunks)

    analysis_file = "complete_analysis10.txt"
    responses_file = "llm_responses10.txt"

    with open(analysis_file, 'w') as af, open(responses_file, 'w') as rf:
        for i, chunk in enumerate(chunks):
            result = await llm.infer(chunk)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted_result = f"Chunk {i+1} ({timestamp}):\n{result}\n"

            separator = f"\n\n**🔥 FIN DE L'ANALYSE DE 500 AVIS CLIENTS 😊 ({timestamp}) 🔥**\n" + "=" * 50 + "\n"

            af.write(formatted_result + separator)
            rf.write(result + separator)

            results[f"Chunk {i+1}"] = formatted_result + separator

            progress_bar.progress((i+1) / total_chunks)
            status_text.text(f"Traitement : {((i+1) / total_chunks) * 100:.2f}% terminé")

    st.session_state['results'] = results  
    return results

# 📌 Bouton pour exécuter l'analyse
st.markdown("<div class='button-container'>", unsafe_allow_html=True)
if st.button("🔄 Lancer l'analyse"):
    results = asyncio.run(process_documents())
    st.success("✔️ Analyse terminée !")
    st.balloons()
st.markdown("</div>", unsafe_allow_html=True)

# 📌 Affichage des résultats avec filtrage dynamique et chunks
if selected_chunk and selected_chunk in st.session_state['results']:
    st.subheader(f"📜 Résultats de l'analyse - {selected_chunk}")

    chunk_results = st.session_state['results'][selected_chunk]

    sections = chunk_results.split("### ")
    section_dict = {sec.split("\n")[0]: sec for sec in sections if sec.strip()}

    if filter_option == "Tout afficher":
        display_text = chunk_results
    else:
        display_text = section_dict.get(filter_option, "Aucune donnée disponible.")

    st.markdown(f"<div class='result-container'>{display_text}</div>", unsafe_allow_html=True)

    # 📌 Vérifier si des pourcentages de sentiment sont disponibles
    if "Synthèse Globale" in section_dict:
        summary = section_dict["Synthèse Globale"]
        sentiments = {"Positifs": 0, "Négatifs": 0, "Neutres": 0}

        matches = re.findall(r"(\d+)%\s*(positif|négatif|neutre)", summary, re.IGNORECASE)
        for match in matches:
            percentage, sentiment = match
            if "positif" in sentiment.lower():
                sentiments["Positifs"] = int(percentage)
            elif "négatif" in sentiment.lower():
                sentiments["Négatifs"] = int(percentage)
            elif "neutre" in sentiment.lower():
                sentiments["Neutres"] = int(percentage)

        if any(sentiments.values()):
            fig, ax = plt.subplots(facecolor='white')
            ax.pie(sentiments.values(), labels=sentiments.keys(), autopct='%1.1f%%', startangle=90, colors=["green", "red", "gray"], wedgeprops={'edgecolor': 'black'}) 
            st.pyplot(fig)
        else:
            st.warning("⚠️ Aucune donnée de sentiment trouvée pour ce chunk.")

    # 📌 Boutons de téléchargement
    with open("complete_analysis10.txt", "rb") as f:
        st.download_button("📥 Télécharger l'analyse complète", f, file_name="complete_analysis.txt", mime="text/plain")

    with open("llm_responses10.txt", "rb") as f:
        st.download_button("📥 Télécharger uniquement les réponses LLM", f, file_name="llm_responses.txt", mime="text/plain")