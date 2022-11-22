# streamlit run app.py
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
import streamlit as st 
import spacy 
import spacy_streamlit
from spacy import displacy
from io import StringIO
import pandas as pd

#Funktion, die eine Liste zu einem String umwandelt
def listToString(s):
    str1 = ""
    for ele in s:
        str1 += ele
    return str1

#Funktion, um einen Dataframe in einer csv zu speichern
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#Styling: Seitentitel, Favicon, etc. 
st.set_page_config(page_title='NerDH Visualisierer', page_icon='../nerdh_tutorial/docs/img/favicon.ico')
st.markdown(""" <style> 
        #MainMenu {visibility: hidden;} 
        footer {visibility: hidden;} 
        </style> """, unsafe_allow_html=True)
st.markdown('## NER-Visualisierer für deutsche (historische) Texte')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#Spacy-Modell auswählen 
model = "de_fnhd_nerdh"
model = st.sidebar.selectbox("Wähle ein Modell:", ["de_fnhd_nerdh", "de_core_news_sm", "de_core_news_md", "de_core_news_lg"], key="model")
with st.spinner('Modell wird geladen...'):
    nlp = spacy.load(model)
st.success('Modell ' + model + ' ist geladen!')

#Informationen über die verschiedenen Modelle
if model == "de_fnhd_nerdh":
    with st.sidebar.expander("Über das Modell"):
        st.write("""
            Das Modell basiert auf frühneuhochdeutschen Texten der digitalen Edition **Philipp Hainhofer: *Reiseberichte & Sammlungsbeschreibungen 1594-1636*** (https://hainhofer.hab.de/). 
            \n Trainiert wurde das Modell mit der Pipeline der Python Bibliothek [Spacy](https://spacy.io/) und dem Text [**München 1611**](https://hainhofer.hab.de/reiseberichte/muenchen1611?v={%22view%22:%22info%22}). 
            \n Das Modell kann [hier](https://huggingface.co/easyh/de_fnhd_nerdh/resolve/main/de_fnhd_nerdh-any-py3-none-any.whl) (586MB) heruntergeladen und als Python-Package installiert werden.
            \n F-Score: **0.92**. Dieser wurde getestet mit Texten aus der Edition.
            \n Mehr Informationen zum Prozess des Trainings etc. gibt es [hier](https://easyh.github.io/NerDH/tut/).
        """)
    with st.sidebar.expander("Named Entities Labels"):
        st.write("""
            **PERSON:** Einzelperson oder Familie
            \n**ORT:** Geographische Einheit, d. h. Länder, Städte, Staaten oder Flüsse.
            \n**ORGANISATION:** Institutionen,(Ordens-)Gemeinschaften, Verbindungen, etc.
            \n**OBJEKT:** Architektur, Gebäude, Kunst, etc.       
            \n**ZEIT:** Datum, Monat, Jahr, Uhrzeit etc.
            """)
elif model == "de_core_news_sm":
    with st.sidebar.expander("Über das Modell"):
            st.write("""
                Das kleinste deutsche Spacy Modell mit nur 13MB. Grund dafür sind die fehlenden Worteinbettungen (Word-Vectors). Trainiert wurde das Modell mit folgenden Quellen: [Tiger Corpus](), [Tiger2Dep]() und [WikiNER]().
                Anwendungsbereich für das Modell sind haupsächtliche moderne Texte und News-Berichte (F-Score: 0.82).
                Daher wird dieses Modell mit historischen Texten nicht wirklich gut abschneiden.
                \n Mehr Informationen zum Modell gibts hier [hier](https://spacy.io/models/de#de_core_news_sm).
            """)
    with st.sidebar.expander("Named Entities Labels"):
            st.write("""
                **PER:** Einzelperson oder Familie
                \n**LOC:** Geographische Einheit, d. h. Länder, Städte, Staaten.
                \n**ORG:** Unternehmen, Agenturen, Institutionen, Regierungseinrichtungen
                \n**MISC:** Gemischte Kategorie (Ereignisse, Nationalitäten, Kunstgegenstände)
            """)
elif model == "de_core_news_md":
    with st.sidebar.expander("Über das Modell"):
            st.write("""
                Das mittlere deutsche Spacy Modell mit 42MB und 20.000 Worteinbettungen. Trainiert wurde das Modell mit folgenden Quellen: [Tiger Corpus](), [Tiger2Dep]() und [WikiNER]().
                Anwendungsbereich für das Modell sind haupsächtliche moderne Texte und News-Berichte (F-Score: 0.84).
                Daher wird dieses Modell mit historischen Texten nicht wirklich gut abschneiden.
                \n Mehr Informationen zum Modell gibts hier [hier](https://spacy.io/models/de#de_core_news_md).
            """)
    with st.sidebar.expander("Named Entities Labels"):
            st.write("""
                 **PER:** Einzelperson oder Familie
                \n**LOC:** Geographische Einheit, d. h. Länder, Städte, Staaten.
                \n**ORG:** Unternehmen, Agenturen, Institutionen, Regierungseinrichtungen
                \n**MISC:** Gemischte Kategorie (Ereignisse, Nationalitäten, Kunstgegenstände)
            """)
else: 
    with st.sidebar.expander("Über das Modell"):
        st.write("""
        Das Größte von den drei deutschen Spacy Modellen mit 541MB und 500.000 Worteinbettungen. Trainiert wurde das Modell mit folgenden Quellen: [Tiger Corpus](), [Tiger2Dep]() und [WikiNER]().
        Anwendungsbereich für das Modell sind haupsächtliche moderne Texte und News-Berichte (F-Score: 0.85).
        Daher wird dieses Modell mit historischen Texten nicht wirklich gut abschneiden.
        \n Mehr Informationen zum Modell gibts hier [hier](https://spacy.io/models/de#de_core_news_lg).
        """)
    with st.sidebar.expander("Named Entities Labels"):
            st.write("""
                 **PER:** Einzelperson oder Familie
                \n**LOC:** Geographische Einheit, d. h. Länder, Städte, Staaten.
                \n**ORG:** Unternehmen, Agenturen, Institutionen, Regierungseinrichtungen
                \n**MISC:** Gemischte Kategorie (Ereignisse, Nationalitäten, Kunstgegenstände)
            """)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#Text hochladen 
st.markdown('### Text hochladen **oder** direkt ins Textfeld einfügen.')
uploaded_file = st.file_uploader("Texte nur im .txt-Format und utf-8- oder utf-16 Kodierung hochladen.")
#Standardtext im Textfeld
DEFAULT_TEXT = """Vmb 12 Vhr bin Jch von Dachaw wider hinweck geritten, vnd vmb 3 Vhr zu Adelshausen, beÿ dem Hannß Wilhalm Hund, Jhrer Durchleucht Rath vnd Cammerer, eingekheret, welcher vermaint, mich v̈ber nacht zu behalten, Habe mich aber entschuldiget, vnd Jhme vnd seiner frawen versprochen, ainmal zu bequemberer zeit, mit meiner haußfrawen zu Jhm zu spatzirn. Jetzt allain ainen trunckh mit Jhme gethan, vnd vmb 4 Vhren wider fort auf Augspurg noch in die Vier meil geritten, vnd zu Abents nach 9 Vhr, Gott lob vnd danckh, glücklich vnd wol, vnd, obwol noch vnuerdienter, mit vil empfangener ehr zu Hauß ankommen."""

#Einlesen des Textes
with st.spinner("Text wird eingelesen..."):
    UPLOADED_TEXT_list = []
    if uploaded_file:
        # To convert to a string based IO:
        try:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        except:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-16"))
        for line in stringio:       
            read_text = str(line)
            clean_text = read_text.replace("\n", " ")
            UPLOADED_TEXT_list.append(clean_text)
        UPLOADED_TEXT = listToString(UPLOADED_TEXT_list)    
        text = st.text_area(" ", UPLOADED_TEXT, height=200)
    else:
        text = st.text_area(" ", DEFAULT_TEXT, height=200)
st.success("Text ist eingelesen!")

st.markdown("---")
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

st.markdown("### Named Entities")
#Farben für die verschiedenen Entitäten 
colors = {"PER": "#fdec3e", "PERSON": "#fdec3e", "LOC": "#7e56c2", "ORT": "#7e56c2", "ORG": "#209485" , "ORGANISATION": "#209485" , "MISC": "#eb4034",  "ZEIT": "#4c9c4b", "OBJEKT": "#7e56c2"}

#Spacy-Streamlit NER Visualizer
#NER-Prozess wird gestartet, je nach Model werden hier die entsprechenden Entitäten gewechselt.
with st.spinner('Named Entities werden gesucht...'):
    doc = nlp(text)
    if model == "de_fnhd_nerdh":    
        entities = st.multiselect('Entitäten auswählen', ['PERSON', 'ORT', 'ORGANISATION', 'OBJEKT', 'ZEIT', 'Alle Entitäten'], default= ['Alle Entitäten'])
        if 'Alle Entitäten' in entities:
            entities = ['PERSON', 'ORT', 'ORGANISATION', 'OBJEKT', 'ZEIT']

        options = {"ents": entities,"colors": colors}
        ent_html = displacy.render(doc, style="ent", options=options, jupyter=False)
        st.markdown(ent_html, unsafe_allow_html=True)
        #spacy_streamlit.visualize_ner(doc, labels = ["PERSON", "ORT", "ORGANISATION", "OBJEKT", "ZEIT",], show_table=False,    colors = colors)
    else: 
        entities = st.multiselect('Entitäten auswählen', ["PER", "LOC", "ORG", "MISC", 'Alle Entitäten'], default= ['Alle Entitäten'])
        if 'Alle Entitäten' in entities:
            entities = ["PER", "LOC", "ORG", "MISC"]

        options = {"ents": entities,"colors": colors}
        ent_html = displacy.render(doc, style="ent", options=options, jupyter=False)
        st.markdown(ent_html, unsafe_allow_html=True)
        #spacy_streamlit.visualize_ner(doc, labels = ["PER", "LOC", "ORG", "MISC"], show_table=False,    colors = colors)
st.markdown(' ')
st.success('Suchprozess ist abgeschlossen!')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#Download-Funktion der Entitäten
st.sidebar.markdown('\n\n')
st.sidebar.markdown('''
### NER-Ergebnnisse in einer .csv-Datei downloaden.
Die Datei enthält die ausgewählten Entitäten.
''')

#Um die NER-Ergebnisse downloaden zu können, werden die Entitäten in einer csv gespeichert
results = []
for ent in doc.ents:
    if ent.label_ in entities:
        results.append([ent.text,ent.label_])
df_results = pd.DataFrame(results, columns = ['text', 'label'])
csv = convert_df(df_results)

st.sidebar.download_button( 
    "Ergebnisse downloaden",
   csv,
   model + '_' + str(entities) + ".csv",
   "text/csv",
   key='download-csv'
)



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#Modell in Python Umgebung installieren 

if model == "de_fnhd_nerdh":
    st.markdown('---') #### Modell in Python installieren und laden')
    st.markdown('#### Modell in Python installieren und laden')
    st.markdown('''
    ```py
    !pip install https://huggingface.co/easyh/de_fnhd_nerdh/resolve/main/de_fnhd_nerdh-any-py3-none-any.whl

    import spacy
    nlp = spacy.load("de_fnhd_nerdh")
    ```
    ''')


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#Referenzen über das Projekt
st.sidebar.markdown('\n\n')
st.sidebar.markdown('\n\n')
with st.sidebar.expander("Referenzen"):
    st.write('''
    Github: https://github.com/easyh/NerDH\n
    Spacy: https://spacy.io/\n
    NerDH-Tutorial: https://easyh.github.io/NerDH/\n
    Trainingsdaten: https://hainhofer.hab.de/
    ''')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#Button zurück zum NerDH Tutorial
st.sidebar.markdown('\n\n')
st.sidebar.write(f'''
    <a target="_blank" href="https://easyh.github.io/NerDH/">
        <button style="color:white;background-color: #209485;  border: none; display: inline-block; border-radius: 8px; padding: 10px 22px;">
           Zurück zum Tutorial
        </button>
    </a>
    ''',
    unsafe_allow_html=True
)