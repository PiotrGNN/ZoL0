class SentimentAnalyzer:
    """Very simple sentiment analyzer stub."""
    def analyze(self, text: str):
        positive_words = ['good', 'great', 'strong']
        score = sum(word in text.lower() for word in positive_words) / len(positive_words)
        return float(score)
