# **NER Tutorial mit spaCy**

In dem Tutorial werden wir das Natural Language Processing (NLP) Tool `spaCy` näher kennenlernen. Auch wenn `spaCy` im NLP Bereich mehr zu bieten hat als nur  Named Entity Recogniton (NER), wird darauf der Fokus des Tutorials liegen. 

Starten werden wir mit einer allgemeinen Einführung in `spaCy`, in welcher wir die grundlegenden Funktionen für Textanalyseverfahren wie NER kennenlernen werden. 

Im zweiten Teil des Tutorials werden wir Schritt für Schritt lernen, wie man mit `spaCy` ein eigenes NER Modell trainiert. Als Trainingstext wird der Reisebericht [**München 1611**](https://hainhofer.hab.de/reiseberichte/muenchen1611?v={%22view%22:%22info%22}) aus der digitalen Edition [**Philipp Hainhofer**](https://hainhofer.hab.de/) verwendet. Als Testtext [**München 1603**](https://hainhofer.hab.de/reiseberichte/muenchen1603) aus der Edition.


??? int "Interaktive Notebooks"

    In diesem Tutorial werden lediglich die Code Beispiele dargestellt. Für die interaktive Benutzung bieten sich folgende drei Möglichkeiten an:   
    
    - Jupyter Notebook auf [mybinder.org](https://mybinder.org/v2/gh/easyh/NerDH/HEAD)

    - [Github Repository Download](https://github.com/easyh/NerDH) 

    -  Kopieren des Codes in ein eigenes Dokument (Kopiersymbol in der rechten oberen Ecke).
    
    Empfohlen wird die Verwendung der Jupyter Notebooks auf [mybinder.org](https://mybinder.org/v2/gh/easyh/NerDH/HEAD). Durch die Verknüpfung des Github Repositorys sind die interaktiven Notebooks direkt im Browser zu verwenden.  

    === ":fontawesome-solid-laptop-code: mybinder.org"

        [Hier geht es zu allen Jupyter Notebooks auf mybinder.org](https://mybinder.org/v2/gh/easyh/NerDH/HEAD){ .md-button }

        **Das Laden des Workspaces wird einen kurzen Moment in Anspruch nehmen.**


    === ":fontawesome-brands-github: Github"

        [:fontawesome-brands-github: easyh/NerDH](https://github.com/easyh/NerDH){ .md-button }

            git clone https://github.com/easyh/NerDH.git

        Die Jupyter Notebooks sind im Ordner `notebooks` hinterlegt.

<br>
**Viel Spaß!**  :muscle: :nerd:
<br>

<br>

---

##
## **1. Einführung in spaCy**
Zunächst wird die Bibliothek `spaCy` von der theoretischen Seite vorgestellt. Anschließend werden wir die Schritte zur Installation von `spaCy` und zum Herunterladen der vortrainierten Sprachmodelle durchgehen. 


### **1.1 Was ist spaCy?**

`spaCy` ist ein leistungsstarkes Tool zur Verarbeitung natürlicher Sprache. Die NLP-Bibliothek für maschinelles Lernen wird seit 2016 von *Explosion AI* stetig weiterentwickelt und befindet sich mittlerweile in der dritten Version (`v.3.4`). Das ist auch die Version, mit der wir hier im Tutorial arbeiten werden. *Explosion AI* ist ein Berliner Team aus Informatikern und Computerlinguisten. 

Die Software-Bibiliothek unterstützt [64 europäische Sprachen](https://spacy.io/usage/models) mit statistischen Modellen, die in der Lage sind Texte zu parsen, Wortteile zu identifizieren und Entitäten zu extrahieren. Zudem ist `spaCy` auch in der Lage, benutzerdefinierte Modelle auf domänenspezifsche Texte zu verbessern bzw. von Grund auf zu trainieren. Für 24 von 64 unterstützten Sprachen insgesamt bietet `spaCy` bereits trainierte Pipelines mit unterschiedlichen Package-Größen an. Weitere sind in Aussicht.[^1] 

??? info "Leistungsumfang von spaCy"
    
    |             |                                                 |  |
    | --------------------- | -------------------------------------------------------------- | ---- |
    |  **Programming Language** | Python |
    |  **Neural network methods** | Neuronale Netze sind eine Reihe von Algorithmen. Sie interpretieren sensorische Daten durch eine Art maschinelle Wahrnehmung, indem sie den rohen Input kennzeichnen oder in Gruppen zusammenfassen. Die Muster, die sie erkennen, sind numerisch und in Vektoren enthalten. | [:link:](https://spacy.io/usage/training)| 
    |  **Integrated word vectors** | Die Ähnlichkeit von Wörtern wird durch den Vergleich von Wortvektoren oder Worteinbettungen, mehrdimensionalen Bedeutungsdarstellungen eines Wortes, ermittelt. Wortvektoren können mit einem Algorithmus wie word2vec erzeugt werden | [:link:](https://spacy.io/usage/linguistic-features#vectors-similarity)| 
    |  **Multi-language support** | Unterstützt insgesamt 64 Sprachen und hat bereits trainierte Pipelines für 24 Sprachen.| [:link:](https://spacy.io/usage/models)| 
    |  **Tokenization** | Bei der Tokenisierung wird ein Text in sinnvolle Segmente, so genannte Token, zerlegt. Die Eingabe für den Tokenizer ist ein Unicode-Text, und die Ausgabe ist ein Doc-Objekt. | [:link:](https://spacy.io/usage/linguistic-features#tokenization)| 
    |  **Part-of-Speech-Tagging** | Darunter versteht man die Zuordnung von Wörtern und Satzzeichen eines Textes zu Wortarten (engl. *part of speech*). Hierzu wird sowohl die Definition des Wortes als auch der Kontext berücksichtigt. | [:link:](https://spacy.io/usage/linguistic-features#pos-tagging)| 
    |  **Sentence segmentation** | Diese Aufgabe beinhaltet die Identifizierung von Satzgrenzen zwischen Wörtern in verschiedenen Sätzen.| [:link:](https://spacy.io/usage/linguistic-features#sbd)| 
    |  **Lemmatization** | Lemmatisierung ist der Prozess der Gruppierung der gebeugten Formen eines Wortes, damit sie als ein einziges Element analysiert werden können, das durch das Lemma des Wortes oder die Wörterbuchform identifiziert wird| [:link:](https://spacy.io/usage/linguistic-features#lemmatization)| 
    |  **Dependency Parsing** | Beim Dependency Parsing (DP) werden die Abhängigkeiten zwischen den Wörtern eines Satzes untersucht, um seine grammatische Struktur zu analysieren. Auf dieser Grundlage wird ein Satz in mehrere Komponenten zerlegt. Der Mechanismus basiert auf dem Konzept, dass es zwischen jeder sprachlichen Einheit eines Satzes eine direkte Verbindung gibt. Diese Verbindungen werden als Abhängigkeiten bezeichnet.| [:link:](https://spacy.io/usage/linguistic-features#dependency-parse)| 
    |  **Named Entity Recognition** | spaCy kann verschiedene Arten von benannten Entitäten in einem Dokument erkennen, indem es das Modell um eine Vorhersage bittet. Da Modelle statistisch sind und stark von den Beispielen abhängen, mit denen sie trainiert wurden, funktioniert dies nicht immer perfekt und muss je nach Anwendungsfall möglicherweise später angepasst werden. | [:link:](https://spacy.io/usage/linguistic-features#named-entities)| 
    |  **Named Entity Linking** | Um die benannten Entitäten in der "realen Welt" zu verankern, bietet spaCy Funktionen für das Entity Linking, bei dem eine textuelle Entität in einen eindeutigen Identifikator aus einer Wissensbasis (KB) aufgelöst wird. Es kann eine eigene KnowledgeBase erstellt und einen neuen EntityLinker mit dieser Wissensbasis trainiert werden. | [:link:](https://spacy.io/usage/linguistic-features#entity-linking)| 

Wie wir sehen, stellt Named Entity Recognition nur ein Bruchteil des Leistungsumfangs von `spaCy` dar. Daher ist ein Blick  in die Dokumentation für weiterführende Informationen sehr empfehlenswert: [https://spacy.io/usage](https://spacy.io/usage). Ebenfalls zu empfehlen ist das [Tutorial von `spaCy`](https://course.spacy.io/de), welches den vollen Leistumgsumfang berücksichtigt.



### **1.2 Installation spaCy**

Um `spaCy` zu installieren, muss nur Folgendes im Terminal eingeben werden[^2].

    pip install spacy 

Wenn `spaCy` vorher noch nie verwendet wurde, müssen die Modelle/Pipelines noch heruntergeladen werden, damit wir mit diesen arbeiten können. Da wir NER mit deutschen Texten machen möchten, sind für uns erstmal nur die deutschen Sprachmodelle von `spaCy` interessant. 

!!! info "spaCy Deutsche Sprackpakete[^3]"

     | Name            |       Precision | Recall | F-Score   | NE-Typen | Word-Vectors | Größe |
     | ---------------- | --------- | --------| ---------| -----------| ---- | ---- |
     | de_core_news_sm   |       0.83 | 0.82 | 0.82   | LOC, MISC, ORG, PER | 0 | 13MB |
     | de_core_news_md   |       0.85 | 0.84 | 0.84   | LOC, MISC, ORG, PER | 20.000 | 42MB |
     |  de_core_news_lg   |       0.85 | 0.85 | 0.85   | LOC, MISC, ORG, PER | 300.000 | 541MB |
 
     Es gibt zwar noch das Paket `de_core_news_trf`, allerdings enthält das Paket keine NER-Pipeline, sodass es für unsere Zwecke nicht relevant ist. 
     Wie wir sehen, unterscheiden sich die Pakete besonders hinsichtlich ihrer Größe. Verantwortlich dafür ist die Anzahl der **Word Vectors/Worteinbettungen**. Eine höhere Anzahl von **Word Vectors** kann für die Ermittlung der Named Entites von Bedeutung sein. 

    Mehr Infos zu den Sprachpaketen von `spaCy` gibt es [hier](https://spacy.io/models/de).

    ??? question "Precision, Recall, F-Score[^4]"

        **Precision:** Berechnet man, indem man die Zahl der richtig gefundenen Vorkommnisse einer Kategorie durch die Gesamtzahl der Markierungen einer Kategorie teilt​. Wurden z.B. 100 Frauenfiguren im Text markiert, davon sind aber nur 97 richtig, so ist `P 97:100=0,97 oder 97%`.

        **Recall:** Berechnet man, indem man die Zahl der richtig gefundenen Vorkommnisse einer Kategorie durch die Gesamtzahl der tatsächlichen Vorkommnisse einer Kategorie teilt. Wurden z.B. 100 Frauenfiguren im Text richtig markiert, insgesamt gibt es aber 150 Ausdrücke, die Frauenfiguren bezeichnen, so ist `R 100:150=0,66 oder 66%`.

        **F-Score:** Kombiniert die Werte von Precision und Recall miteiander. Ein Wert nahe 1 bzw. 100% ist sehr gut.

Die Sprachpakete können mit folgenden Befehlen heruntergeladen werden.

``` py
python -m spacy download de_core_news_sm
```
``` py
python -m spacy download de_core_news_md
```
``` py
python -m spacy download de_core_news_lg
```

Mit folgendem Befehl können wir uns unsere `spaCy`-Version sowie die bereits installierten Modelle/Pipelines anzeigen lassen.

    python -m spacy info


<br>

---

##
## **2. Erste Schritte mit spaCy**

Bevor wir nun mit der Erkennung von Named Entities und deren Visualisierung beginnen, lernen wir zunächts ein paar grundlegende Objekte und Funktionen von spacy kennen. Bis auf den ersten Codeteil sind die anderen NLP-Funktionen für unsere NER-Aufgabe nicht wichtig, sollen aber dennoch kurz vorgestellt werden, da sie wichtige Grundlagen für Textanalysen sind.

=== "Code"
   
    ``` py
    #zuerst importieren wir spaCy
    import spacy

    #in der Variable text ist der Text gespeichert, der analysiert werden soll.
    text = "Mia Müller wohnt in Trier und studiert an der Universität Trier. Aufgewachsen ist sie in München mit ihrem Bruder Tom."

    #Als nächstes müssen wir ein Modellobjekt laden. 
    #Hierfür verwenden wir die Funktion spacy.load(). 
    #Diese nimmt ein Argument entgegen, nämlich das Modell, dass wir laden möchten.
    #Wir werden das kleine deutsche Modell verwenden.
    nlp = spacy.load("de_core_news_sm")

    #Nachdem wir das nlp-Objekt erstellt haben, können wir es verwenden, um einen Text zu analysieren.
    #Zu diesem Zweck erstellen wir ein doc-Objekt.
    #Dieses Objekt wird eine Menge Daten über den Text enthalten.
    doc = nlp(text)

    #Wir testen, ob das doc-Objekt unseren Text übernommen hat.
    print(doc)
    ```
=== "Output"

    ```
    Mia Müller wohnt in Trier und studiert an der Universität Trier. Aufgewachsen ist sie in München mit ihrem Bruder Tom.
    ```   

??? code "Sentence Tokenizer"

    === "Code" 
    
        ``` py
        for sent in doc.sents:
            print(sent)
        ```
    === "Output"

        ```
        Mia Müller wohnt in Trier und studiert an der Universität Trier.
        Aufgewachsen ist sie in München mit ihrem Bruder Tom.
        ```     

??? code "Part-of-Speech Tagging"
      
    === "Code"
    
        ``` py
        for token in doc: 
            print(token.text, token.pos_)
        ```
    === "Output"

        ```
        Mia PROPN
        Müller PROPN
        wohnt VERB
        in ADP
        Trier PROPN
        und CCONJ
        studiert VERB
        an ADP
        der DET
        niversität NOUN
        Trier PROPN
        . PUNCT
        Aufgewachsen VERB
        ist AUX
        sie PRON
        in ADP
        München PROPN
        mit ADP
        ihrem DET
        Bruder NOUN
        Tom PROPN
        . PUNCT
        ```  

??? code "Substantive und Substantiv-Bausteine extrahieren"

    === "Code"
    
        ``` py
        for chunk in doc.noun_chunks:
            print(chunk.text)
        ```
    === "Output"

        ```
        Mia
        Trier
        der Universität Trier
        sie
        München
        ihrem Bruder Tom
        ```

??? code "Verben extrahieren"

    === "Code"
    
        ``` py
        #Verben sind im POS-TAg als "VERB" oder "AUX" definiert, daher iterieren wir für über alle POS-TAGS, die übereinstimmen.
        verbs = ["VERB", "AUX"]
        for token in doc:
            if token.pos_ in verbs:
                print (token.text, token.pos_)
        ```
    === "Output"

        ```
        wohnt VERB
        studiert VERB
        Aufgewachsen VERB
        ist AUX
        ```


??? code "Lemmatisierung"
    === "Code"
    
        ``` py
        for token in doc:
            print(token.text, token.lemma_)
        ```
    === "Output"

        ```
        Mia Mia
        Müller Müller
        wohnt wohnen
        in in
        Trier Trier
        und und
        studiert studieren
        an an
        der der
        Universität Universität
        Trier Trier
        . --
        Aufgewachsen aufgewachsen
        ist sein
        sie sie
        in in
        München München
        mit mit
        ihrem ihr
        Bruder Bruder
        Tom Tom
        . --
        ```


### **2.1 Entitäten erkennen** 

Der Code um Named Entities in einem Dokument zu erkennen ist ähnlich simpel wie der Code der anderen Funktionen. Wir iterieren erneut über unser `doc-Objekt` mit der Funktion `.ents `. 

=== "Code"
   
    ``` py
    for ent in doc.ents:
        print(ent.text,ent.label_)
    ```
=== "Output"

    ```
    Mia Müller PER
    Trier LOC
    Universität Trier. ORG
    München LOC
    Tom. PER
    ``` 
Wie wir sehen, hat das kleine Sprachmodell hier sehr gute Arbeit geleistet. Alle vorkommenden Named Entities wurden richtig erkannt. Sogar die zwei Wörter `Universität` und  `Trier` wurden als `ORGANISATION` verstanden und nicht nur als `ORT`. Dieses Phänomen wird auch als **Nested Entities** bezeichnet, weil in einer Entität gleich mehrere stecken können.


Um beispielsweise nur die benannten Entitäten zu extrahieren, die beispielsweise als PERSON identifiziert wurden, können wir eine einfache if-Anweisung in den Mix einfügen.

=== "Code"
   
    ``` py
    for ent in doc.ents:
        if ent.label_ == "PER":
        print(ent)
    ```
=== "Output"

    ```
    Mia Müller
    Tom.
    ``` 

Wir können uns zusätzlich noch ausgeben lassen, an welcher Stelle im Text die Named Entities zu finden sind. 

=== "Code"
   
    ``` py
    for ent in doc.ents:
        print(ent.text, ent,start_char, ent.end_char, ent.label_)
    ```
=== "Output"

    ```
    Mia Müller 0 10 PER
    Trier 20 25 LOC
    Universität Trier. 46 64 ORG
    München 89 96 LOC
    Tom. 114 118 PER
    ``` 
     


### **2.2 NER visualisieren**

`spaCy` hat eine eingebaute Funktion zur Visualisierung der Entitäten namens `displacy`. Der schnellste Weg, ein `doc-Objekt` zu visualisieren ist `displacy.serve`. Dadurch wird ein einfacher Webserver gestartet und das Ergebnis kann im Browser betrachtet werden. Da wir innerhalb eines Jupyter Notebooks arbeiten, verwenden wir die Funktion `displacy.render`. Zunächst müssen wir dazu noch `displacy` importieren.

=== "Code"
   
    ``` py
    from spacy import displacy

    displacy.render(doc, style="ent")
    ```
=== "Output"
    <div>
    <figure style="margin-bottom: 1rem">
    <div class="entities" style="line-height: 2.5; direction: ltr">
    <mark class="entity" style="color: black; background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
        Mia Müller
        <span style="font-size: 0.6em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PER</span>
    </mark>
    wohnt in 
    <mark class="entity" style="color: black; background: #ff9561; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
        Trier
        <span style="font-size: 0.6em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">LOC</span>
    </mark>
    und studiert an der 
    <mark class="entity" style="color: black; background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
        Universität Trier.
        <span style="font-size: 0.6em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
    </mark>
    Aufgewachsen ist sie in 
    <mark class="entity" style="color: black; background: #ff9561; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
        München
        <span style="font-size: 0.6em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">LOC</span>
    </mark>
    mit ihrem Bruder 
    <mark class="entity" style=" color: black; background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
        Tom.
        <span style="font-size: 0.6em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PER</span>
    </mark>
    </div>
    </figure>
    </div>

Hier können wir jetzt noch eigene Anpassungen wie die Auswahl der Entitäten, als auch die Farbe ausführen. Die individuellen Farben geben wir für alle vier Entitätstypen an, allerdings wollen wir uns hier nur die Personen (`PER`) und Orte (`LOC`) ausgeben lassen.    

=== "Code"
   
    ``` py
    colors = {"PER": "#fdec3e", "LOC": "#7e56c2", "ORG": "#209485" , "MISC": "#eb4034"}
    options = {"ents": ["PER", "LOC"], "colors": colors}

    displacy.render(doc, style="ent", options=options)
    ```
=== "Output"
    <div>
    <figure style="margin-bottom: 1rem">
    <div class="entities" style="line-height: 2.5; direction: ltr">
    <mark class="entity" style="color: black; background: #fdec3e; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
        Mia Müller
        <span style="font-size: 0.6em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PER</span>
    </mark>
    wohnt in 
    <mark class="entity" style="color: black; background: #7e56c2; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
        Trier
        <span style="font-size: 0.6em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">LOC</span>
    </mark>
    und studiert an der Universität Trier.
    Aufgewachsen ist sie in 
    <mark class="entity" style="color: black; background: #7e56c2; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
        München
        <span style="font-size: 0.6em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">LOC</span>
    </mark>
    mit ihrem Bruder 
    <mark class="entity" style=" color: black; background: #fdec3e; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
        Tom.
        <span style="font-size: 0.6em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PER</span>
    </mark>
    </div>
    </figure>
    </div>



<br>




!!! int "Interaktive Notebooks"

    Der Code zu diesem Kapitel befindet sich hier: `notebooks/01_firstSteps_spacy.ipynb`.

    === ":fontawesome-solid-laptop-code: mybinder.org"

        [Hier geht es zum Jupyter Notebook auf mybinder.org](https://mybinder.org/v2/gh/easyh/NerDH/HEAD){ .md-button }

         **Das Laden des Workspaces wird einen kurzen Moment in Anspruch nehmen.**


    === ":fontawesome-brands-github: Github"

        [Hier geht es zum Notebook auf Github](https://github.com/easyh/NerDH/blob/main/notebooks/01_firstSteps_spacy.ipynb){ .md-button }




<br>




---
##
## **3. Eigenes Modell trainieren mit spaCy**

Je nach Anwendungsfall macht es wenig Sinn nur mit dem Standardmodell zu arbeiten. Wesshalb es besser ist, ein neues Modell zu trainieren bzw. auf ein bestehendes aufzubauen. Besonders bei historischen Texten entstehen Probleme und Schwierigkeiten aufgrund ihrer Heterogenität. 

!!! ex "NER mit fnhd. Text"
    Folgendes Beispiel zeigt, wie das kleine `spaCy` Sprachmodell mit einem Satz aus dem frühneuhochdeutschen abschneidet. Der Satz ist aus unserem Traningstext [**München 1611**](https://hainhofer.hab.de/reiseberichte/muenchen1611?v={%22view%22:%22info%22}) von Philipp Hainhofer.

    === "Code"
        
        ```py
        text_fnhd = "Aines mörders Conterfett, genant Christoff Froschhammer von Vlingingen, der Hat 345 mörd, mit seiner aignen hand, vnd 400 mord in gesellschafft anderer, gethan, ist Anno 1578 zu Welß in Steÿrmarck gerichtet worden, vnd auß dem stifft Saltzburg gebürtig gewesen."
        nlp = spacy.load("de_core_news_sm")
        doc_fnhd = nlp(text_fnhd)
        for ent in doc_fnhd.ents:
            print(ent.text,ent.label_)
        ```

    === "Output"

        ```
        Aines mörders MISC
        Christoff Froschhammer PER
        Anno LOC
        Welß PER
        Steÿrmarck MISC
        Saltzburg LOC
        ```

    Wie wir sehen wurden nur zwei Entitäten richtig erkannt: `Christoff Froschhammer` als `PERSON` und `Saltzburg` als `ORT`. `Vlingingen`, sowie `Welß` und `Steÿrmarck` wurden nicht richtig oder garnicht als `ORT` erkannt. Zudem hat das Modell Entitäten erkannt, die eigentlich keine Entitäten sind: `Anno` und `Aines mörder`.


Vor dem Training eines eigenen NER-Modells ist es wichtig, folgende drei Fragen zu klären: 

??? question "Was sind meine Daten?"
    Unsere Daten entstammen der digitalen Edition [**Philipp Hainhofer**](https://hainhofer.hab.de/). Hier haben wir uns für den Reisebericht [**München 1611**](https://hainhofer.hab.de/reiseberichte/muenchen1611?v={%22view%22:%22info%22}) als Trainingstext und [**München 1603**](https://hainhofer.hab.de/reiseberichte/muenchen1603) als Testtext entschieden. Die Texte der Edition stehen in `TEI-XML`, `PDF` sowie `TXT` zum Download verfügbar. Wir verwenden die Daten im  `TXT`-Format, denn einige Studien haben gezeigt, dass eine umfangreiche XML-Annotation, die dem NER-Prozess vorausgeht, die Leistung beeinträchtigen kann. NER-Systeme sollten daher idealerweise angewendet werden, bevor ein Korpus mit Standards wie TEI annotiert wird. In dieser Reihenfolge können die NER-Ergebnisse dann auch bei der TEI-Codierung sehr hilfreich sein. 

??? question "Welche Entitäten möchte ich auswählen?"
    Hier geht es darum, die relevanten Kategorien und Entitäten auszuwählen. Dafür ist es wichtig zu wissen, welche und wie viele NE-Katogorien man für die jeweiligen Texte verwenden will.
    
    In unserem Modell werden wir die folgenden Entitäten trainieren: 

    | Named Entities               | Beschreibung                                    | 
    | ------------------ | -------------------------------------------------------------- | 
    |  `PERSON`       | Einzelperson oder Familie                                | 
    |  `ORT`       | Geographische Einheit, d. h. Länder, Städte, Staaten, Flüsse                                  | 
    |  `ORGANISATION`       | Institutionen,(Ordens-)Gemeinschaften, Verbindungen, etc.                                 | 
    |  `OBJEKT`       |  Architektur, Gebäude, Kunst, etc.                                    |
    |  `ZEIT`       |  Bücher mit eindeutigem Namen (z.B. Das Alte Testament)                                    | 


??? question "Wie möchte ich diese Entitäten kennzeichen?"
    Hier geht es darum, wie die Entitäten annotiert werden. Dabei sollte sich auf eine einheitliche Annotationskonvention beschränkt werden. Das Ergebnis sollte am ein Ende ein Goldstandard sein, anhand dessen später das NER-Modell bewertet wird, indem die per Hand angefertigen Annotationen mit der Ausgabe des NER-Systems verglichen werden. 


Im Laufe dieses Kapitels werden wir lernen, wie das Training eines eigenen Modells mit `spaCy` umgesetzt wird. Unser Workflow für das Training eines eigenen NER-Modells sieht wie folgt aus: 

<figure markdown>
  ![Annotation Goldtstandard](img/workflow.svg){ width="500" }
  <figcaption style="font-size: 0.8em;">Workflow: Training eines eigenen NER Modells</figcaption>
</figure>


### **3.1 Preprocessing**
Bevor wir mit dem Trainings- & dem anschließenden Evaluierungsprozess starten, müssen wir zunächst einen Goldstandard erstellen und diesen dann in Trainings-, Validierungs- & Testdaten aufteilen. Dieser Schritt gehört zum sogennanten **Preprocessing** des maschinellen Lernens. 

Vor der Annotation des Goldstandards, sollten wir unsere beiden Texte allerdings noch etwas kennenlernen und vorbereiten. Auf die nähere Ausführung soll an dieser Stelle verzichtet werden, allerdings ist alles wichtige zu diesem Schritt im Notebook `02_preprocessingText.ipynb` festgehalten.

!!! int "Interaktive Notebooks"

    Der Code zu diesem Kapitel befindet sich hier: `notebooks/02_preprocessingText.ipynb`.

    === ":fontawesome-solid-laptop-code: mybinder.org"

        [Hier geht es zum Jupyter Notebook auf mybinder.org](https://mybinder.org/v2/gh/easyh/NerDH/HEAD){ .md-button }

         **Das Laden des Workspaces wird einen kurzen Moment in Anspruch nehmen.**


    === ":fontawesome-brands-github: Github"

        [Hier geht es zum Notebook auf Github](https://github.com/easyh/NerDH/blob/main/notebooks/02_preprocessingText.ipynb){ .md-button }

<br>

#### **3.1.1 Annotation eines Goldstandards**

Die Annotation des Goldstandrds ist der erste wichtige Schritt, um ein Trainingskorpus zu erstellen. 

!!! Goldstandard info  

    Ein Goldstandard ist eine manuelle Referenz-Annotation, bei der die relevanten Named Entities annotiert werden. Es ist die finale Version der annotierten Daten, die für den Trainingsprozess verwendet werden. Je besser und genauer die Auszeichnungen für den Goldstandard gemacht werden, desto besser ist das Trainingsergebnis. Anhand des Goldstandards, kann später die Qualität eines NER-Modell bewertet werden, quantifziert als Recall, Precision und F-Score.

Es gibt mehrere Möglichkeiten Texte so zu annotieren, dass sie maschinenlesbar sind. Hier muss je nach Anwendungsbereich entschieden werden, welcher Annotationsstandard der passendste ist. In den Digital Humanities werden die meisten Texte mithilfe der auf XML basierenden Anwendung der [Text Encoding Initiative (TEI)](https://tei-c.org/) annotiert. Ein anderer bekannter Standard zum Annotieren von Named Entities ist das [IOB-Schema](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)) (bzw. BIO-Schema). 

Da wir allerdings ein NER-Modell mit `spaCy` trainieren wollen, müssen wir unseren Goldstandard entsprechend für die Software anpassen. `spaCy` verlangt ein spezielles Traningsformat, in welchem die Daten vorliegen müssen. Das Beispiel zeigt das klassische Format: 

```
TRAIN_DATA = ["TEXT AS A STRING",{"entities:"[(START,END,LABEL)]}]
```

Es gibt zahlreiche Auszeichnungstools, die für Anwendungen des maschinellen Lernens entwickelt wurden und bei der Annotation von Texten helfen. Darunter auch einige, welche die Daten direkt in das entsprechende Trainingsformat für `spaCy` bringen. Dafür wurde speziell [`prodigy`](https://prodi.gy/)  entwickelt, was allerdings kostenpflichtig ist. Aber es gibt auch einige Open-Source-Programme, die eine gute Alternative darstellen. 

Wir haben unseren Goldstandard mit dem [NER-Annotator](https://tecoholic.github.io/ner-annotator/) erstellt. Hier kann ein Text im `TXT`-Format importiert, annotiert und dann anschließend im `JSON`-Format exportiert werden. Die `JSON`-Datei enthält dann die annotierten Daten, die in dem für `spaCy` geeigneten Format vorliegen. Allerdings müssen die `JSON`-Dateien für den Trainingsprozess nochmal in `.spacy`-Dateien umgewandelt werden. Der [NER-Annotator](https://tecoholic.github.io/ner-annotator/) stellt hier Code zur Verfügung (näheres dazu im nächsten Kapitel zur Erstellung der Datensets).  

Das folgende Bild zeigt, wie die Annotation erfolgt. In der oberen Zeile können die Entitäten Kategorien festgelegt werden. Ist eine Kategorie mit einem Haken markiert (hier `ORT`) kann im Text damit die entsprechende Entität markiert werden. So arbeiten wir uns Stück für Stück durch unseren Text, bis wir fertig sind. Enthält ein Satz keine Entitäten, dann überspringen (`skip`) wir diesen. In unserer Datei sind später nur Sätze enthalten, die Entitäten enthalten.

<figure markdown>
  ![Annotation Goldtstandard](img/history_example.png){ width="700" }
  <figcaption style="font-size: 0.8em;">Erstellung des Goldstandards für das Training mit dem NER-Annotator-for-spaCy. </figcaption>
</figure>

In der `JSON`-Datei ist dieser Satz dann im folgenden Format wieder zu finden: 

```json
{"classes": ["PERSON","ORT","ORGANISATION","OBJEKT","ZEIT"],
    "annotations": [["Aines mörders Conterfett, genant Christoff Froschhammer von Vlingingen, der Hat 345 mörd, 
            mit seiner aignen hand, vnd 400 mord in gesellschafft anderer, gethan, ist Anno 1578 zu Welß 
            in Steÿrmarck gerichtet worden, vnd auß dem stifft Saltzburg gebürtig gewesen.",
        {"entities": [[33,55,"PERSON"],[60,70,"ORT"],[165,174,"ZEIT"],[178,182,"ORT"],[186,196,"ORT"],[234,243,"ORT"]]}]]}
```

Der Prozess des Annotieren kann je nach Länge des Textes sehr aufwändig und anstrengend sein. Bei sehr langen Texten bietet es sich an, den Text in einzelene Dateien zu unterteilen, um dann in Etappen die Annotation durchzuführen. Am Ende können die einzelnen `JSON`-Dateien, dann wieder zu einer Datei zusammengefügt werden, die dann den finalen Goldstandard darstellt. 


!!! info "Die folgende Tabelle soll einen kurzen Überblick über den Trainingstext  [**München 1611**](https://hainhofer.hab.de/reiseberichte/muenchen1611) und den Testtext [**München 1603**](https://hainhofer.hab.de/reiseberichte/muenchen1603) und dessen annotierten Enititäten geben."

    | Text | Sätze | Tokens/Wörter | Zeichen | `PERSON` |  `ORT` |  `ORG`    | `OBJEKT`  | `ZEIT` | Gesamt Entitäten |
    | --- | ------- | ------- | -------| --------- | ------ | ----------------- | ------- | ------- | ------- | 
    | [**München 1611**](https://hainhofer.hab.de/reiseberichte/muenchen1611) | 1 595 | 41 087 | 225 952 | 1019 |  840 |  63 |  238 | 451 | 2611  |
    | [**München 1603**](https://hainhofer.hab.de/reiseberichte/muenchen1603) | 147 | 5 928 | 31 747  | 54 | 55 |  16 | 53  | 22  | 200 |
    | **GESAMT** | 1 742 | 47 015 | 257 699 | 1073 | 895  | 79  | 291  | 473   | 2811 |

Der Goldstandard für den Traningstext befindet sich [**hier :fontawesome-brands-github:**](https://github.com/easyh/NerDH/blob/main/data/datensets/taggedData.json) (`data/datensets/taggedData.json`).

<br>

#### **3.1.2 Training-, Validierung- und Testdaten**

Beim maschinellen Lernen benötigen wir einen Trainingsdatensatz, um ein Modell richtig zu trainieren und einen Testdatensatz, um das Modell zu bewerten.  Der Datensatz [`taggedData.json`](https://github.com/easyh/NerDH/blob/main/data/datensets/taggedData.json) umfasst den komplett annotierten Text [**München 1611**](https://hainhofer.hab.de/reiseberichte/muenchen1611?v={%22view%22:%22info%22}). Bei **unüberwachten Lernmethoden** wird dieser Datensatz in der Regel in mindestens drei verschiedene Datensätze unterteilt: **Training-, Validierung- und Testdaten**. In unserem Fall übernehmen die Trainingsdaten ([`testData.json`](https://github.com/easyh/NerDH/blob/main/data/datensets/taggedData.json)) bei uns die annotierten Daten aus dem Text [**München 1603**](https://hainhofer.hab.de/reiseberichte/muenchen1603).


??? info "Unüberwachtes und Überwachtes Lernen"
    Bei **unüberwachten Lernmethode**  brauchen wir keine Beispiele, da hier direkt mit den Eingabedaten trainiert wird. Der Algorithmus ermittelt hier von selbst Muster und Zusammenhänge. Dieser Prozesss funktioniert mit minalen menschlichem Aufwand. In unserem Fall arbeiten wir allerdings mit der  **überwachten Lernmethode**, bei der Beispieldaten in Form eines annotierten Goldtstandards notwendig sind, damit der Algoritmus lernen kann. 

Die Aufteilung unsrer Daten sieht wie folgt aus. Den Datensatz [`taggedData.json`](https://github.com/easyh/NerDH/blob/main/data/datensets/taggedData.json) werden wir noch in **Trainingsdaten** und **Validierungsdaten** einteilen müssen.

<figure markdown>
  ![Datensetsverteilung](img/datenset.png){ width="500" }
  <figcaption style="font-size: 0.8em;">Aufteilung in Training-, Validierung- und Testdaten. </figcaption>
</figure>

??? info "Training-, Validierung- und Testdaten[^7]"
    
    | Datensatz | Anteil | Erklärung |
    | ---------- | -------- | ------ |
    | **Traningsdaten** |  70% | Ein Trainingsdatensatz ist eine Sammlung von Beispielen, die verwendet werden, um einem Algorithmus beizubringen, Muster und Zusammenhänge in den Daten zu erkennen. Der Algorithmus passt seine Gewichte anhand der Trainingsdaten an, indem er aus ihnen lernt. Trainingsdaten werden für Klassifikations- und Regressionsprobleme benötigt, bei denen es darum geht, Vorhersagen für bestimmte Zielvariablen zu treffen. Es kann vorkommen, dass Algorithmen, die auf Trainingsdaten lernen, zu sehr auf die Muster in diesen Daten angepasst werden und somit nicht gut auf neue, noch nicht gesehene Daten anwendbar sind. Dies wird als "Überanpassung" oder "Overfitting" bezeichnet. Das bedeutet, dass der Algorithmus zu starke Regeln aus den Trainingsdaten lernt, die auf die Gesamtheit der Daten nicht gut anwendbar sind.|
    | **Validierungsdaten** | 20% | Der Validierungsdatensatz ist eine Sammlung von Beispieldaten, die verwendet werden, um die Hyperparameter eines Modells anzupassen. Hyperparameter sind Einstellungen, die vor dem Training festgelegt werden und Einfluss auf das Lernverhalten des Modells haben. Beispiele für Hyperparameter bei künstlichen neuronalen Netzen sind die Anzahl der Neuronen in jeder Schicht oder die Lernrate. Durch die Verwendung von Validierungsdaten beim Training kann verhindert werden, dass das Modell zu sehr auf die Trainingsdaten angepasst wird und somit auf neue, noch nicht gesehene Daten nicht gut anwendbar ist.|
    | **Testdaten** | 10% | Die Testdaten sind von den Trainingsdaten unabhängig und werden während des Trainingsprozesses nicht verwendet. Sie dienen dazu, das trainierte Modell zu bewerten und zu überprüfen, wie gut es auf neue, noch nicht gesehene Daten anwendbar ist. Die Testdaten sollten dieselbe Wahrscheinlichkeitsverteilung wie der Trainingsdatensatz aufweisen. Wenn das Modell gut auf die Testdaten anwendbar ist, kann es vermutlich auch auf andere, bisher ungesehene Daten angewendet werden. |
    

Um jetzt unseren großen Datensatz [`taggedData.json`](https://github.com/easyh/NerDH/blob/main/data/datensets/taggedData.json) in zwei Datensets aufzuteilen, lesen wir diesen zunächts ein und speichern nur die Einträge von  `annotations` in der Variablen `TAGGED_DATA`, damit wir die Einträge zählen können. Danach ermitteln wir die Grenze (80:20), damit wir den urspünglichen Datensatz in kleinere Datensätze zu je 80% und 20%.

=== "Code"
    ```py
    import json

    f = open('../data/datensets/taggedData.json')
    data = json.load(f)
    TAGGED_DATA = data['annotations']

    print(len(TAGGED_DATA)*0.8)
    ```

=== "Output"

    ```
    696.8000000000001
    ```

Bevor wir den Datensatz an der ermittelten Grenzen in die zwei Datensets aufteilen, mischen wir die Einträge noch einmal durch, damit die Verteilung zufällig ist. Hier lassen wir uns dann jeweils die Länge ausgeben, um das Ergebnis zu überprüfen. Zusätzlich lassen wir uns auch noch die Größe des Testdatensatz ausgeben, um zu überprüfen, ob die 70:20:10 Verteilung ungefähr hinhaut.

=== "Code"
    ```py
    import random 

    random.shuffle(TAGGED_DATA)
    train_data = TAGGED_DATA[:697]
    val_data = TAGGED_DATA[697:]

    print("Traningsdaten: " + str(len(train_data)))
    print("Validierungsdaten: " + str(len(val_data)))

    #Zum vergleich, lassen wir uns auch die Göße von unseren Testdaten ausgeben
    f = open('../data/datensets/testData.json')
    data = json.load(f)
    test_data = data['annotations']
    print("Testdaten: " + str(len(test_data)))
    ```

=== "Output"

    ```
    Traningsdaten: 697
    Validierungsdaten: 174
    Testdaten: 87
    ```

Anschließend speichern wir die Datensets im `JSON`-Format ab.

=== "Code"
    ```py
    with open ('../data/datensets/trainData.json', 'w', encoding='utf-8') as train: 
        json.dump(train_data, train, ensure_ascii=False, indent=4)
    with open ('../data/datensets/valuationData.json', 'w', encoding='utf-8') as val: 
        json.dump(val_data, val, ensure_ascii=False, indent=4)
    ```

Da wir weiter oben nur die Einträge aus annotations der JSON-Datei übernommen haben, müssen wir jetzt noch einmal manuell bei `trainData.json` sowie `validationData.json` die `classes` hinzufügen, damit unsere Kategorien für die Entitäten nicht verloren gehen. Dazu setzen wir an den Anfang des Dokuments folgendes und ans Ende eine `}` um das Dokument zu schließen.

```json
{"classes": ["PERSON","ORT", "ORGANISATION","OBJEKT","ZEIT"],
    "annotations": 
```
Jetzt müssen die Datensets im `JSON`-Format nurnoch ins `spaCy`-Format konvertiert werden. Dafür importieren wir zunächst die ensprechenden Bibliotheken und das mittlere Sprachmodell von `spaCy`.

=== "Code"
    ```py
    import spacy
    from spacy.tokens import DocBin
    from tqdm import tqdm

    nlp = spacy.load("de_core_news_md") # load a new spacy model
    db = DocBin() # create a DocBin object
    ```

Jetzt müssen wir für jedes Datenset nur noch folgenden Code ausführen, welcher uns vom [NER-Annotator](https://tecoholic.github.io/ner-annotator/) vorgegeben wird, damit die Datensets im `spaCy`-Datenformat sind. 

=== "Traningsdaten"
    ```py
    f = open('../data/datensets/trainData.json')
    TRAIN_DATA = json.load(f)

    for text, annot in tqdm(TRAIN_DATA['annotations']): 
    doc = nlp.make_doc(text) 
    ents = []
    for start, end, label in annot["entities"]:
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    doc.ents = ents 
    db.add(doc)

    db.to_disk("../data/datasets/trainData.spacy") # save the docbin object
    ```
=== "Validierungsdaten"
    ```py
    f = open('../data/datensets/valuationData.json')
    VAL_DATA = json.load(f)

    for text, annot in tqdm(VAL_DATA['annotations']): 
    doc = nlp.make_doc(text) 
    ents = []
    for start, end, label in annot["entities"]:
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    doc.ents = ents 
    db.add(doc)

    db.to_disk("../data/datasets/valuationData.spacy") # save the docbin object
    ```
=== "Testdaten"
    ```py
    f = open('../data/datensets/testData.json')
    TEST_DATA = json.load(f)

    for text, annot in tqdm(TEST_DATA['annotations']): 
    doc = nlp.make_doc(text) 
    ents = []
    for start, end, label in annot["entities"]:
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    doc.ents = ents 
    db.add(doc)

    db.to_disk("../data/datasets/testData.spacy") # save the docbin object
    ```
Damit ist das Preprocessing abgeschlossen und wir können mit dem Training des Modells beginnen.


<br>

!!! int "Interaktive Notebooks"

    Der Code zu diesem Kapitel befindet sich hier: `notebooks/03_createDatasets_spacy.ipynb`.

    === ":fontawesome-solid-laptop-code: mybinder.org"

        [Hier geht es zum Jupyter Notebook auf mybinder.org](https://mybinder.org/v2/gh/easyh/NerDH/HEAD){ .md-button }

         **Das Laden des Workspaces wird einen kurzen Moment in Anspruch nehmen.**


    === ":fontawesome-brands-github: Github"

        [Hier geht es zum Notebook auf Github](https://github.com/easyh/NerDH/blob/main/notebooks/03_createDatasets_spacy.ipynb){ .md-button }

<br>





### **3.3 Training**

Da wir jetzt alle unsere Daten haben, ist es an der Zeit unser  Modell zu trainieren. In `spaCy` können wir die Architektur unserer Netzes sowie die Hyperparamter für unser Modell weitgehend steuern. Dies geschieht in der `config.cfg` Datei. Diese Konfigurationsdatei wird `spaCy` während dem Trainingsprozess übergeben, damit das System weiß, was und wie es trainieren soll. Um die `config.cfg` Datei zu erstellen, können wir die [praktische GUI von `spaCy`](https://spacy.io/usage/training#quickstart) selbst benutzen.

Für unsere Zwecke wählen wir `German`, `ner` und `CPU`(GPU ist etwas komplexer). Jenachdem ob wir ohne Wortvektoren oder mit ihnen trainieren wollen, wählen wir `efficiency` oder `accuracy`. In unserem Beispiel haben wir mit `accuracy`, also mit Wortvektoren trainiert. Hier werden dann die Vektoren des großen deutschen Modell `de_core_news_lg` übernommen (das können wir natürlich auch mit dem kleinen bzw. mittleren Modell machen). Diese Datei speichern wir zunächst als `base_config.cfg` ab. Die `base_config.cfg`-Datei befindet sich ebenfalls auf Github im Ordner `notebooks` oder kann hier einfach kopiert werden. 

??? code "`base_config.cfg`"
    ```cfg
    [paths]
    train = null
    dev = null
    vectors = "de_core_news_lg"
    [system]
    gpu_allocator = null

    [nlp]
    lang = "de"
    pipeline = ["tok2vec","ner"]
    batch_size = 1000

    [components]

    [components.tok2vec]
    factory = "tok2vec"

    [components.tok2vec.model]
    @architectures = "spacy.Tok2Vec.v2"

    [components.tok2vec.model.embed]
    @architectures = "spacy.MultiHashEmbed.v2"
    width = ${components.tok2vec.model.encode.width}
    attrs = ["NORM", "PREFIX", "SUFFIX", "SHAPE"]
    rows = [5000, 1000, 2500, 2500]
    include_static_vectors = true

    [components.tok2vec.model.encode]
    @architectures = "spacy.MaxoutWindowEncoder.v2"
    width = 256
    depth = 8
    window_size = 1
    maxout_pieces = 3

    [components.ner]
    factory = "ner"

    [components.ner.model]
    @architectures = "spacy.TransitionBasedParser.v2"
    state_type = "ner"
    extra_state_tokens = false
    hidden_width = 64
    maxout_pieces = 2
    use_upper = true
    nO = null

    [components.ner.model.tok2vec]
    @architectures = "spacy.Tok2VecListener.v1"
    width = ${components.tok2vec.model.encode.width}

    [corpora]

    [corpora.train]
    @readers = "spacy.Corpus.v1"
    path = ${paths.train}
    max_length = 0

    [corpora.dev]
    @readers = "spacy.Corpus.v1"
    path = ${paths.dev}
    max_length = 0

    [training]
    dev_corpus = "corpora.dev"
    train_corpus = "corpora.train"

    [training.optimizer]
    @optimizers = "Adam.v1"

    [training.batcher]
    @batchers = "spacy.batch_by_words.v1"
    discard_oversize = false
    tolerance = 0.2

    [training.batcher.size]
    @schedules = "compounding.v1"
    start = 100
    stop = 1000
    compound = 1.001

    [initialize]
    vectors = ${paths.vectors}
    ```

Da die `base_config.cfg`-Datei nun korrekt eingerichtet ist, müssen wir sie in eine `config.cfg`-Datei umwandeln. Dazu müssen wir einen Terminalbefehl ausführen. Durch Ausführen des folgenden Befehls erhalten wir die korrekt formatierte `config.cfg`-Datei.

```
python -m spacy init fill-config ./base_config.cfg ./config.cfg
```
Jetzt haben wir alles, um unser erstes Modell zu trainieren. Wir erstellen einen Ordner namens `output`, hier wird unser fertiges Modell dann abgelegt. Wir müssen nur den folgenden Befehl ausführen, etwas warten und schon haben wir ein trainiertes Modell.  Hinter `paths.train` setzen wir den Pfad zu unseren Trainingsdaten und hinter `paths.dev` unsere Valuierungsdaten

```
python -m spacy train config.cfg --output ./output --paths.train ./datensets/trainData.spacy --paths.dev ./datensets/valuationData.spacy
```

??? code "Trainingsausgabe"

    Die Ausgabe zeigt uns die Epochen, die Anzahl der Stichproben sowie einige Metriken für unser Modell. 

    ```
    ℹ Saving to output directory: output
    ℹ Using CPU

    =========================== Initializing pipeline ===========================
    [2022-11-19 13:23:54,923] [INFO] Set up nlp object from config
    [2022-11-19 13:23:54,941] [INFO] Pipeline: ['tok2vec', 'ner']
    [2022-11-19 13:23:54,949] [INFO] Created vocabulary
    [2022-11-19 13:24:00,463] [INFO] Added vectors: de_core_news_lg
    [2022-11-19 13:24:06,491] [INFO] Finished initializing nlp object
    [2022-11-19 13:24:09,274] [INFO] Initialized pipeline components: ['tok2vec', 'ner']
    ✔ Initialized pipeline

    ============================= Training pipeline =============================
    ℹ Pipeline: ['tok2vec', 'ner']
    ℹ Initial learn rate: 0.001
    E    #       LOSS TOK2VEC  LOSS NER  ENTS_F  ENTS_P  ENTS_R  SCORE 
    ---  ------  ------------  --------  ------  ------  ------  ------
    0       0          0.00     85.09    0.00    0.00    0.00    0.00
    0     200        880.61   2879.21   55.96   75.37   44.50    0.56
    1     400         65.55   1510.84   71.81   67.93   76.15    0.72
    2     600        109.01   1268.59   82.27   81.65   82.89    0.82
    4     800         78.71    901.58   86.86   87.08   86.64    0.87
    5    1000         74.96    737.24   88.99   91.39   86.73    0.89
    7    1200         80.31    559.43   90.62   91.06   90.18    0.91
    show more (open the raw output data in a text editor) ...

    143    5200        180.50     58.97   93.98   93.86   94.10    0.94
    151    5400        211.91     75.53   93.96   93.79   94.14    0.94
    159    5600        376.93    128.75   94.00   93.98   94.02    0.94
    ✔ Saved pipeline to output directory
    output/model-last
    ```

In unserem `output`-Ordner befinden sich nun zwei Unterordner: `model-best`und `model-last`. Beide dieser Modelle können jetzt in der `spacy.load()` Funktion eingelesen und ausprobiert werden. Hier wird dann einfach der Pfad angegeben, wo das Modell liegt. 

```py
import spacy 

nlp = spacy.load(output/model-best)

```
Im Github Repositry gibt es keinen `output`-Ordner mit einem Modell, da das trainierte Modell zu groß ist. Allerdings gibt es von `spaCy` die Möglichkeit ein Modell in ein Python Package zu packen, dass dann wie jedes andere Package mit `pip install`installiert werden kann. 

??? note "Modell als Python Package verpacken" 
    Dafür müssen wir uns einmal in das innere unseres Modells klicken und in der `meta.json`-Datei noch kleine Änderungen machen.  Bei `name` vergeben wir dem Modell einen Namen, mit dem wir es laden möchten. Zusätzlich geben wir noch eine `version` an: Da es unser erstes Modell ist setzen wir die Version auf `0.0.1`. Natürlich können wir hier auch noch mehr Informationen wie z.B. `description`, `author` oder `email`angeben. 

    ```json
    {
    "lang":"de",
    "name":"de_fnhd_nerdh",
    "version":"0.0.1",
    "spacy_version":">=3.4.1,<3.5.0",
    "description":"",
    "author":"",
    "email":"",
    ...}
    ```

    Dann erstellen wir einen Ordner `package`, hier wird unser verpacktes Modell gespeichert, und geben folgenden Terminalbefehl ein: 

    ```
    python -m spacy package ./output/model-best ./package --build wheel
    ```
    Lokal können wir unser Modell jetzt installieren einfach wie folgt installieren. Wir müssen uns hierfür allerdings im Verzeichnis `dist` befinden oder dorthin navigieren: 

    ```
    pip install package/de_fnhd_nerdh-0.0.1/dist/de_fnhd_nerdh-0.0.1-py3-none-any.whl
    ```
    Jetzt können wir unser selbst erstelltes Modell wie die `spaCy` Modelle benutzen. 

    ```py
    import spacy 

    nlp = spacy.load(de_fnhd_nerdh)
    ```
   
    ??? note "Modellpackage veröffentlichen"

        Mithilfe von [`spacy-huggingface-hub`](https://github.com/explosion/spacy-huggingface-hub) können wir unser verpacktes Modell veröffentlichen und anderen Usern anbieten. Dazu müssen wir uns vorher einen Account auf [huggingface.co](https://huggingface.co/) erstellen und das Package installieren. 

        ```
        pip install spacy-huggingface-hub
        ```
        Jetzt können wir uns über den Terminal in [huggingface.co](https://huggingface.co/) einloggen, damit unser Package unserem Profil zugeordnet wird. Hier werden wir nach einem Token gefragt, den wir [hier](https://huggingface.co/settings/tokens) abrufen können. 

        ```
        huggingface-cli login
        ```
        Mit folgendem Befehl pushen wir unser Projekt auf [huggingface.co](https://huggingface.co/). 

        ```
        python -m spacy huggingface-hub push package/de_fnhd_nerdh-0.0.1/dist/de_fnhd_nerdh-0.0.1-py3-none-any.whl
        ```
        Der Befehl wird dann zwei Dinge ausgeben: 
        
        - wo das Repository auf [huggingface.co](https://huggingface.co/) gefunden wird 
        - Link, mit welchem das Modell installiert werden kann

Das Modell, welches hier im Tutorial erstellt wurde kann mit diesem Befehl als Python Package installiert werden.

```
pip install https://huggingface.co/easyh/de_fnhd_nerdh/resolve/main/de_fnhd_nerdh-any-py3-none-any.whl
```

<br>

!!! int "Interaktive Notebooks"

    Der Code zu diesem Kapitel befindet sich hier: `notebooks/04_trainEvaluateModel_spacy.ipynb`.

    === ":fontawesome-solid-laptop-code: mybinder.org"

        [Hier geht es zum Jupyter Notebook auf mybinder.org](https://mybinder.org/v2/gh/easyh/NerDH/HEAD){ .md-button }

         **Das Laden des Workspaces wird einen kurzen Moment in Anspruch nehmen.**


    === ":fontawesome-brands-github: Github"

        [Hier geht es zum Notebook auf Github](https://github.com/easyh/NerDH/blob/main/notebooks/04_trainEvaluateModel_spacy.ipynb){ .md-button }

<br>

### **3.4 Evaluierung** 

Der nächste logische Schritt ist jetzt natürlich die Evaluation unseres Modells. Hierfür benötigen wir unsere Testdaten. Das Modell wird dann anhand von **F-Score**, **Precision** und **Recall** bewertet. `spaCy` hat hierfür einen einfachen Terminalbefehl. 

```
python -m spacy evaluate de_fnhd_nerdh ./data/datensets/trainData.spacy
```

Das Output sollte dann in etwa so aussehen: 
```
ℹ Using CPU

================================== Results ==================================

TOK     100.00
NER P   94.85 
NER R   91.06 
NER F   92.92 
SPEED   2297  


=============================== NER (per type) ===============================

                   P       R       F
PERSON         94.81   91.41   93.08
ZEIT           92.64   91.89   92.27
ORT            95.94   94.26   95.09
OBJEKT         94.44   81.79   87.66
ORGANISATION   95.59   82.28   88.44
```
Mit diesen Werten können wir ziemlich zufrieden sein. Sollte das allerdings nicht der Fall sein, dann würden wir einfach unseren Trainingsprozess mit kleinen Veränderungen nochmal erneut starten. Nicht aufgeben! :muscle::nerd:


<br>

!!! int "Interaktive Notebooks"

    Der Code zu diesem Kapitel befindet sich hier: `notebooks/04_trainEvaluateModel_spacy.ipynb`.

    === ":fontawesome-solid-laptop-code: mybinder.org"

        [Hier geht es zum Jupyter Notebook auf mybinder.org](https://mybinder.org/v2/gh/easyh/NerDH/HEAD){ .md-button }

        **Das Laden des Workspaces wird einen kurzen Moment in Anspruch nehmen.**


    === ":fontawesome-brands-github: Github"

        [Hier geht es zum Notebook auf Github](https://github.com/easyh/NerDH/blob/main/notebooks/04_trainEvaluateModel_spacy.ipynb){ .md-button }

<br>





<br>


[^1]: spaCy. Industrial-strength Natural Language Processing in Python. [https://spacy.io/](http://web.archive.org/web/20230102123431/https://spacy.io)
[^2]: spaCy. Industrial-strength Natural Language Processing in Python. [https://spacy.io/usage](http://web.archive.org/web/20230102123639/https://spacy.io/usage)
[^3]: spaCy. Industrial-strength Natural Language Processing in Python.[https://spacy.io/models/de](http://web.archive.org/web/20230102123719/https://spacy.io/models/de)
[^4]: Schumacher, M. K. (2020). Named Entity Recognition und Reverse Engineering. Lebe lieber literarisch. [https://lebelieberliterarisch.de/
named-entity-recognition-und-reverse-engineering/](http://web.archive.org/web/20230102124132/https://lebelieberliterarisch.de/named-entity-recognition-und-reverse-engineering/)
[^7]: datasolut GmbH. (2021). Was sind Trainingsdaten im Machine Learning? - datasolut Wiki. [https://datasolut.com/wiki/trainingsdaten-und-testdaten-machine-learning/](http://web.archive.org/web/20230102124244/https://datasolut.com/wiki/trainingsdaten-und-testdaten-machine-learning/)