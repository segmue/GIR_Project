# GIR Projekt

Dieses Projekt entstand im Rahmen des Moduls GEO871 Geographic Information Retrieval (HS2024).

## Projektbeschreibung
Das Ziel dieses Projekts ist die Evaluierung und Analyse von Geoparsing-Methoden, um Ortsreferenzen aus unstrukturiertem Text zu extrahieren und aufzulösen. Der Workflow umfasst die Vorverarbeitung von Rohdaten, die Anwendung verschiedener Geoparser, die Harmonisierung der Ergebnisse und die anschließende Auswertung anhand definierter Metriken.

**[Link zum Paper](GEO871_Sebastian_Gmür.pdf)**

## Ordner-Struktur
- **data**: Heruntergeladene ACLED-Daten.
- **data_preprocessed**: Bereinigte ACLED-Daten für Geoparsing.
- **edgeoparser_raw_txt**: Einzelne .txt-Dateien als Eingabeformat für den Edinburgh Geoparser.
- **edgeoparser_output**: Einzelne .xml-Ausgabedateien des Edinburgh Geoparsers.
- **edgeoparser_output_pre_cleaned**: Zusammengeführte Pandas-Pickle-Dateien der .xml-Ausgaben pro Region.
- **irchel_geoparser_output**: Pickle-Dateien der Batch-Verarbeitung des Irchel Geoparsers.
- **result**: Harmonisierte Pickle-Dateien der Geoparser-Ausgaben und der Ground Truth, inklusive Zusammenfassungsstatistiken.

## Preprocessing
1. Entfernt Einträge, deren "Location" nicht im Rohtext vorkommt (meistens wegen unterschiedlicher Schreibweisen).
   - **Script**: `preprocess.py`
2. Erstellt aus jedem Eintrag eine einzelne .txt-Datei (Eingabeformat für den Edinburgh Geoparser).
   - **Script**: `create_raw_txt_files_for_edgeoparser.py`

## Edinburgh Geoparser
1. Anwendung der Geoparser-Pipeline auf jedes einzelne File im Batch-Ordner. Speichert die Stdout-Ausgabe als `.out.xml`.
   - **Originaler Geoparser-Code**: [Edinburgh Geoparser](https://www.ltg.ed.ac.uk/software/geoparser/)
   - **Script**: `./run-multiple-files.sh`
2. Konvertiert die Output-Dateien in das Pickle-Datenformat.
   - **Script**: `process_edinburgh_output_to_pkl.py`

## Irchel Geoparser
- Google Colab Notebook, das ACLED-Rohdaten von Google Drive mountet, in Batch-Größen von 10.000 Einträgen verarbeitet und die Ergebnisse wieder auf Drive speichert.
  - **Notebook**: `Irchel_Geoparser_ACLED.ipynb`

## Datenaufbereitung für Analyse
- Liest die Ausgaben des Edinburgh Geoparsers, des Irchel Geoparsers und die Rohdaten ein. Erstellt einen harmonisierten DataFrame für die Auswertung.
  - **Script**: `match_outputs_with_original.py`

## Auswertung
- Berechnet verschiedene Metriken (siehe Paper), speichert diese als .csv und erstellt Grafiken.
  - **Script**: `result_analysis.py`
