import torch


@torch.no_grad()
def match_label(boxes, labels, matcher):
    matches = []
    match_flags = []
    if isinstance(boxes, list):
        for b, label in zip(boxes, labels):
            label_box = label[:, :4]
            match, match_flag = matcher(b, label_box)
            matches.append(match)
            match_flags.append(match_flag)
    else:
        for label in labels:
            label_box = label[:, :4]
            match, match_flag = matcher(boxes, label_box)
            matches.append(match)
            match_flags.append(match_flag)
    return matches, match_flags


@torch.no_grad()
def sample_pos_neg(match_flags, sample_num, pos_fraction):
    for i, match_flag in enumerate(match_flags):
        positive = torch.nonzero(match_flag > 0, as_tuple=True)[0]
        negative = torch.nonzero(match_flag == 0, as_tuple=True)[0]

        num_pos = int(sample_num * pos_fraction)
        # protect against not enough positive examples
        num_pos = min(positive.numel(), num_pos)
        num_neg = sample_num - num_pos
        # protect against not enough negative examples
        num_neg = min(negative.numel(), num_neg)

        # randomly select positive and negative examples
        perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
        perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

        pos_idx = positive[perm1]
        neg_idx = negative[perm2]

        match_flag.fill_(-1)
        match_flag.scatter_(0, pos_idx, 1)
        match_flag.scatter_(0, neg_idx, 0)

        match_flags[i] = match_flag
    return match_flags


def freeze(model, if_freeze):
    model.requires_grad_(not if_freeze)
