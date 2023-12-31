# Emotionen & Entscheidung: Sentiment-Analyse politischer Reden während der Covid-19-Pandemie

**Im Rahmen des Seminars Politische Debatten und Polarisierung im Deutschen Bundestag**

<center>

![Sentiment Analysis](https://github.com/PHofmannn/Politische-Debatten/blob/main/SChwerpunjt.png)

</center>

## Überblick

In diesem Repository befindet sich der Code und die Daten für die Sentiment-Analyse politischer Reden und Kommunikationen im Kontext der Covid-19-Pandemie. Für die Sentiment- und Topic-Analyse habe ich Transformer-Modelle<sup>1,2</sup> genutzt, um diese Analyse effizient und zuverlässig durchzuführen.

## Hintergrund

Die Covid-19-Pandemie hat unsere Welt grundlegend verändert und erfordert, dass evidenzbasierte Entscheidungen und politische Maßnahmen getroffen werden, um die Krise zu bewältigen. Sentiment-Analysen können dabei
helfen, Trends, Stimmungsänderungen und politische Reaktionen auf die Pandemie zu verfolgen.

## Forschungsfragen

In meiner Arbeit stehen folgende Hauptfragen im Mittelpunkt:

1. **Gibt es einen Zusammenhang zwischen der 7-Tages-Inzidenz und dem Sentiment im Bundestag?**
   
   Ich analysiere, ob die Veränderungen in der Stimmung der politischen Reden im Bundestag mit den Schwankungen der 7-Tages-Inzidenz von Covid-19 in Deutschland korrelieren.

2. **Welche Themen waren während des Lockdowns und am Peak der 7-Tages-Inzidenz relevant?**

   Ich untersuche, welche politischen Themen während des Lockdowns und bei höchster 7-Tages-Inzidenz besonders hervorgehoben wurden und wie sich die Stimmung dazu veränderte.

## Methodik

Meine Analyse basiert auf Transformer-Modellen, die eine robuste und effiziente Methode für die Sentiment-Analyse politischer Reden bieten. Ich verwende offene Datenquellen, vom [RKI](https://github.com/robert-koch-institut/COVID-19_7-Tage-Inzidenz_in_Deutschland/blob/main/COVID-19-Faelle_7-Tage-Inzidenz_Deutschland.csv) zur Identifizierung der 7-Tages-Inzidenz, das [oliverguhr/german-sentiment-bert](https://huggingface.co/oliverguhr/german-sentiment-bert) für die Sentiment-Analyse und das [chkla/parlbert-topic-german](https://huggingface.co/chkla/parlbert-topic-german) zur Klassifikation der Themen.

## Ergebnisse

![Sentiment Analysis](https://github.com/PHofmannn/Politische-Debatten/blob/main/SentimentScore.png)

Die Ergebnisse meiner Analyse wurden im Rahmen eines [Posters](https://github.com/PHofmannn/Politische-Debatten/blob/main/Emotionen%26Entscheidung_Poster.pdf) festgehalten.

Das jupyter Notebook zu meiner Arbeit findet man [hier.](https://github.com/PHofmannn/Politische-Debatten/blob/main/SentimentAnalyse.ipynb)

Die Schritte zur Datenvorbereitung und Filterung der Reden der 19. und 20. Wahlperiode mit der Anwendung der Sentimentanalyse sind [hier zu finden.](https://github.com/PHofmannn/Politische-Debatten/blob/main/Data_Preprocessing.ipynb)

## Autorin

- Paula Hofmann (https://github.com/PHofmannn)

## Kontakt

Bei Fragen oder Anmerkungen: paulahofmann@icloud.com

---

1. Guhr, Oliver, et al. "Training a broad-coverage German sentiment classification model for dialog systems." Proceedings of the Twelfth Language Resources and Evaluation Conference. 2020

2. Klamm, Christopher, Ines Rehbein, and Simone Paolo Ponzetto. "FrameASt: A framework for second-level agenda setting in parliamentary debates through the lens of comparative agenda topics." (2022): 92-100.

