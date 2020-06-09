from collections import defaultdict 
from tqdm import tqdm


def compute_frequencies(lines, topk):
    w2f = defaultdict(lambda: 0.0) # word 2 frequency

    w2lines = defaultdict(lambda: [])

    for idx, line in tqdm(enumerate(lines)):
        for word in line.split():
            w2f[word] += 1.0

    most_frequent_words = sorted(w2f.items(), key=lambda x: -x[1])[:topk]
    just_words = set([w[0] for w in most_frequent_words])

    return just_words

def unkify_lines(lines, top_words):
    new_lines = []
    for line in tqdm(lines):
        new_words = []
        for word in line.split():
            if word in top_words:
                new_words.append(word)
            else:
                new_words.append("<unk>")

        new_lines.append(" ".join(new_words))

    return new_lines
