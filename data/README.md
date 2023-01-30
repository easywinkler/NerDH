## **Über die Daten**
___________________________________________

Im Ordner [`texte`](https://github.com/easyh/NerDH/tree/main/data/texte) befindet sich der Trainingstext [München 1611](https://hainhofer.hab.de/reiseberichte/muenchen1611?v={%22view%22:%22info%22}) sowie der Testtext [München 1603](https://hainhofer.hab.de/reiseberichte/muenchen1603) aus der digitalen Edition [**Philipp Hainhofer: Reiseberichte & Sammlungsbeschreibungen 1594 - 1636**](https://hainhofer.hab.de/). Die Texte wurden mit dem [Notebook `02_preprocessingText.ipynb`](https://github.com/easyh/NerDH/blob/main/notebooks/02_preprocessingText.ipynb) für den weiteren Prozess vorbereitet. 

---

Im Ordner [`datensets`](https://github.com/easyh/NerDH/tree/main/data/datensets) befinden sich die annotierten Datensets mit Goldstandard Anspruch. Annotiert wurden die Texte mit dem [NER Annotator for Spacy](https://tecoholic.github.io/ner-annotator/). 

[`taggedData.json`](https://github.com/easyh/NerDH/blob/main/data/datensets/taggedData.json) umfasst den annotierten Text [München 1611](https://hainhofer.hab.de/reiseberichte/muenchen1611?v={%22view%22:%22info%22})  und [`testData.json`](https://github.com/easyh/NerDH/blob/main/data/datensets/testData.json) den Testtext [München 1603](https://hainhofer.hab.de/reiseberichte/muenchen1603).  Das Datenset [`taggedData.json`](https://github.com/easyh/NerDH/blob/main/data/datensets/taggedData.json) wurde in die Datensets [`trainData.json`](https://github.com/easyh/NerDH/blob/main/data/datensets/trainData.json) und [`validationData.json`](https://github.com/easyh/NerDH/blob/main/data/datensets/validationData.json). Die finale Verteilung der Datensets ist dabei wie folgt: 

<div>
<img src="../nerdh_tutorial/docs/img/datenset_lg.png" width="400"/>
</div>

Da der [NER Annotator for Spacy](https://tecoholic.github.io/ner-annotator/) die Daten in einem `JSON`-Format exportiert, müssen diese fürs Training noch in das `spacy`-Format konvertiert werden. Dieser Schritt sowie das *Spliten* des Datensets wurde mit dem Notebook [`03_createDatasets_spacy.ipynb`](https://github.com/easyh/NerDH/blob/main/notebooks/03_createDatasets_spacy.ipynb) durchgeführt.
