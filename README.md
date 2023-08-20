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
