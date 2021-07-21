import os

from transformers import AutoTokenizer


def preprocess(dataset, model_name_or_path, max_len, output_file):
    NL = "\n"
    subword_len_counter = 0

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    max_len -= tokenizer.num_special_tokens_to_add()

    out_fp = open(output_file)
    with open(dataset, "rt") as f_p:
        for line in f_p:
            line = line.rstrip()

            if not line:
                out_fp.write(line + NL)
                subword_len_counter = 0
                continue

            token = line.split()[0]

            current_subwords_len = len(tokenizer.tokenize(token))

            # Token contains strange control characters like \x96 or \x95
            # Just filter out the complete line
            if current_subwords_len == 0:
                continue

            if (subword_len_counter + current_subwords_len) > max_len:
                out_fp.write("" + NL)
                out_fp.write(line + NL)
                subword_len_counter = current_subwords_len
                continue

            subword_len_counter += current_subwords_len

            out_fp.write(line + NL)
    out_fp.close()


if __name__ == "__main__":
    from argparse import ArgumentParser

    import gamechangerml.src.text_classif.utils.log_init as li

    li.initialize_logger(to_file=False, log_name="none")

    parser = ArgumentParser(
        prog="python " + os.path.split(__file__)[-1],
        description="Create NER training data in CoNLL format",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        dest="dataset",
        type=str,
        help="formatted training data",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model-name-or-path",
        dest="model",
        type=str,
        help="pytorch model",
        required=True,
    )
    parser.add_argument(
        "-l", "max-length", dest="max_len", type=int, help="max token length"
    )
    parser.add_argument(
        "-o",
        "--output-txt",
        dest="output_txt",
        type=str,
        required=True,
        help="output file in CoNLL-2003 format with len(token) <= max_len",
    )
