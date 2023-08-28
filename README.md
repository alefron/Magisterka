# Magisterka

Część plików (poniżej zaznaczono które) została pobrana z zewnętrznych źródeł. Na to repozytorium zostały wrzucony jedynie w celu ułatwienia pracy.

**skrypty:**
youtube_downloader.py - ściąganie piosenek z youtube. Wczytuje plik csv, pobiera końcówkę linku do yt i pobiera film, konwertuje do mp3 i zapisuje. Tak dla każdego wiersza.

mp3ToWav.py - konwertowanie wszystkich pobranych plików mp3 do wav, bo pobrane pliki mp3 mają niepoprawny header i biblioteki nie mogą na nich poprawnie pracować

countSongsWithHighCoverage.py - liczy ile piosenek które udało się pobrać z youtuba ma percent_coverage powyżej podanej wartości. percent_coverage jest odczytywana dla każdej piosenki z pliku csv pobranego z publicznego repo

align_audio.py - pobrany z internetu. Dostosowuje długość piosenek pobranych z YouTube tak aby pasowały do anotacji salami. Czyli wyrównuje offsety. Znów na podstawie danych z pliku csv
źródło: https://github.com/jblsmith/matching-salami

compareAudioLength.py - wypisuje ile plików audio zostało faktycznie przyciętych w wyniku działania skryptu align_audio

internetArchiveDownloader.py - pobieracz plików mp3 które są dostępne do pobrania w ramach dotasetu salami. Linki są w pliku csv

**pliki:**
id_index_internetarchive.csv - lista piosenek w datasecie salami dostępnych do pobrania w formie bezpośrednich linków. Zawiera te linki
źródło: https://github.com/DDMAL/salami-data-public

metadata.csv:
dane o wszystkich piosenkach tworzących zbiór danych SALAMI. Wymienione ważne dla mnie pola:
- ID piosenki
- źródło (Codaich, IA, RWC, Isophonic)
- IDs annotatorów
- czas trwania piosenki (w sekundach?)
- tytuł i artysta
- klasa i gatunek
źródło: https://github.com/DDMAL/salami-data-public

salami_youtube_pairings.csv - plik parujący piosenki do których oryginalnie nie udostępniono linków z linkami do YouTube
źródło: https://github.com/jblsmith/matching-salami

**artykuły:**

https://grrrr.org/pub/ullrich_schlueter_grill-2014-ismir.pdf
https://studenttheses.uu.nl/bitstream/handle/20.500.12932/36522/bachelor%20thesis%20leander%20van%20boven.pdf?sequence=1&isAllowed=y
https://grrrr.org/data/pub/grill_schlueter-2015-ismir.pdf
https://eprints.soton.ac.uk/271171/1/AHMsalami-v2.pdf
https://d1wqtxts1xzle7.cloudfront.net/30800272/PS4-14-libre.pdf?1391869066=&response-content-disposition=inline%3B+filename%3DDesign_and_creation_of_a_large_scale_dat.pdf&Expires=1692622974&Signature=RZ6BPJgTZvH4Q5xmErTGBy-eUhtEei9nm5lVtfX6TcuWjbsIQlAu8Dh71fIs5ysmVEa3gbOVl~80NGAEryyUGF1ndfUdBRrMWsTuffBPaT45w92aeiKKUY03y1XYv71e5D-0I-LGjhB0w9MPD7I0JzFddva--Qz9un1SmdOzTYzYcPb9wtrNDAa-6HwLfdCNNopT2ibnhdm7t2ZfSsUAE2GmtWlCZIhsxZj9CY5kcVLkikrpADEsYyOIXcdQRcCAefPJ5FqEDpjQjAcl-pa0wP2RVAks1JVv2IC2WdPwBiezgSuKquQkoVsf-llTeX25jkBoFIQ-LymIpO4Xs9yqOA__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA

**linki:**
MSAF doc - https://msaf.readthedocs.io/en/latest/tutorial.html
