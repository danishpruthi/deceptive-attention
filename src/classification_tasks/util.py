import numpy as np
import torch

def pretty_importance_scores_vertical(words, scores, total_width=100):
    """Print importance scores w/ the sentence displayed vertically"""
    print(" ".join(words))
    scores = np.asarray(scores)
    scores /= np.sum(np.abs(scores) + 1e-20)
    widths = (scores * total_width).astype(int)
    min_width = widths.min()
    offsets = np.minimum(0, widths) - min_width
    max_word_size = max(len(word) for word in words)
    for word, width, offset in zip(words, widths, offsets):
        bar = " " * int(offset) + "#" * int(np.abs(width))
        print(f"{word:{max_word_size}s} {bar}")

def entropy(p):
    return torch.distributions.Categorical(probs=p).entropy()

def confidence_penalty(p, beta):
    return -1 * beta * entropy(p)

def annealed_confidence_penalty(p, beta, eta):
    return -1 * beta * max(eta - entropy(p))


def anonymize(words):
    gender_replacements = {
        "he": "they",
        "she": "they",
        "her": "their",
        "his": "their",
        "him": "them",
        "himself": "themself",
        "herself": "themself",
        "hers": "their",
    }

    anon_words = []
    for word in words.split():
        if word in gender_replacements:
            anon_words.append(gender_replacements[word])
        else:
            anon_words.append(word)
    return " ".join(anon_words)
