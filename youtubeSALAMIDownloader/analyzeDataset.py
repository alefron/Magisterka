import numpy as np

labels_coding = {
 'Bridge': np.array([1., 0., 0., 0., 0., 0., 0., 0., 0.]),
 'Chorus': np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.]),
 'Instrumental': np.array([0., 0., 1., 0., 0., 0., 0., 0., 0.]),
 'Interlude': np.array([0., 0., 0., 1., 0., 0., 0., 0., 0.]),
 'Intro': np.array([0., 0., 0., 0., 1., 0., 0., 0., 0.]),
 'No-function': np.array([0., 0., 0., 0., 0., 1., 0., 0., 0.]),
 'Outro': np.array([0., 0., 0., 0., 0., 0., 1., 0., 0.]),
 'Silence': np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.]),
 'Verse': np.array([0., 0., 0., 0., 0., 0., 0., 0., 1.])
}

# labels to tablica ndarray zawierająca tablice z 9 liczbami (0 lub 1)
# labels_coding to słownik przekładający kombinacje na słowo


def analyzeDataset(labels):
    # Inicjalizacja słownika do przechowywania statystyk
    word_stats = {word: {'count': 0, 'percentage': 0} for word in labels_coding.keys()}

    # Obliczanie statystyk
    total_samples = labels.shape[0]  # Całkowita liczba próbek

    for label_row in labels:
        matching_words = [word for word, encoding in labels_coding.items() if np.array_equal(label_row, encoding)]
        if matching_words:
            for word in matching_words:
                word_stats[word]['count'] += 1

    # Obliczanie procentowych statystyk
    for word, stats in word_stats.items():
        stats['percentage'] = (stats['count'] / total_samples) * 100

    return word_stats

    # Wyświetlenie statystyk
    for word, stats in word_stats.items():
        print(f"Word: {word}, Count: {stats['count']}, Percentage: {stats['percentage']}%")
