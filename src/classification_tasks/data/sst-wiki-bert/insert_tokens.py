for typ in ["train", "test", "dev"]:
    infile = typ + ".txt" + ".old"
    blfile = typ + ".txt" + ".block"
    new_text = ''
    with open(infile) as inf, open(blfile) as blf:
        lines = inf.readlines()
        blocks_all = blf.readlines()
        for i, line in enumerate(lines):
            blocks = [int(item) for item in blocks_all[i].split() ]
            lst = line.split('\t', 1)
            label = lst[0]
            tokens = lst[1].split()
            assert len(tokens) == len(blocks)
            len_sst = sum(blocks)
            new_tokens = tokens[len_sst:] + ['[SEP]'] + tokens[0:len_sst]
            new_text += label + '\t' + ' '.join(new_tokens) + '\n'

    with open(typ + ".txt", 'w') as outf:
        outf.write(new_text)
