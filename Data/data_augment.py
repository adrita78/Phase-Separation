
def random_replace_with_A(inputs, labels, factor):
    assert inputs.shape[0] == labels.shape[0]
    #assert len(labels.shape) == 1
    new_inputs = []
    new_labels = []

    for idx in range(inputs.shape[0]):
        ip = inputs[idx].clone()
        label = labels[idx]
        padding_idex = trR_ALPHABET_DICT['-']
        unpadded_len = (ip != padding_idx).sum().item()
        num_to_replace = round(unpadded_len * factor)
        indices = np.random.choice(unpadded_len, num_to_replace, replace=False)
        ip[indices] = trR_ALPHABET_DICT['A']
        new_inputs.append(ip)
        new_labels.append(label)

    modified_inputs = torch.stack(new_inputs)
    updated_mask = update_input_mask(modified_inputs, padding_idx=padding_idx)

    return modified_inputs, updated_mask, torch.tensor(new_labels)
