import torch
from sequence_models.pretrained import load_model_and_alphabet
from sequence_models.constants import trR_ALPHABET



trR_ALPHABET = "ARNDCQEGHILKMFPSTWYV-"

trR_ALPHABET_DICT = {char: idx for idx, char in enumerate(trR_ALPHABET)}
print(trR_ALPHABET_DICT)


def tokenize_sequence(sequence, alphabet_dict):
    """
    Converts each character in the sequence to its corresponding integer token using the alphabet dictionary.
    """
    return [alphabet_dict[char] for char in sequence]

def alphabet_with_mask(seqs, padding_idx):
    """
    Tokenizes and pads sequences, and generates a mask for padding.

    Args:
    seqs (list of list of str): List of sequences (each sequence is a list containing a string).
    padding_idx (int): Index used for padding (e.g., 20 for '-').

    Returns:
    x (torch.Tensor): Tensor of padded tokenized sequences.
    input_mask (torch.Tensor): Tensor where 1 indicates a valid token and 0 indicates padding.
    """
    tokenized_seqs = [tokenize_sequence(seq[0], trR_ALPHABET_DICT) for seq in seqs]
    max_length = max(len(seq) for seq in tokenized_seqs)
    padded_tokens = [seq + [padding_idx] * (max_length - len(seq)) for seq in tokenized_seqs]
    x = torch.tensor(padded_tokens)
    input_mask = (x != padding_idx).float()

    return x, input_mask