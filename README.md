
# GIR Projekt

### Preprocessing
- Entfernt alle Einträge, deren "Location" nicht im Rohtext vorkommt (Meinstens aufgrund unterschiedlicher Schreibweise)
- Script: preprocess.py

- Erstellt aus jedem Eintrag ein einzelnes .txt File (Inputformat für edinburg geoparser)
- Script: create_raw_txt_files_for_edgeoparser.py

### Edinburgh Geoparser
- Für jedes einzelne File im Unterordner (aufgeteilt in Batches), run Geoparser Pipeline, speichere Stdout als .out.xml file:
- script: ./run-multiple-files.sh

#### Post-Processing:
- Einlesen jedes Output-Files in ein Pickle Data Format:
- script: process_edinburgh_output_to_pkl.py

### Datenaufbereitung für Analyse
- Liest Edinburg Output ein, Irchel Geoparser Output (kopiert von Google Drive) und Rohdaten und erstellt Dataframe für Auswertung
- script: match_outputs_with_original.py

### Auswertung

-scripts: result_analysis.py
