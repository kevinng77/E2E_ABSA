import math
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super(SelfAttention, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model,
                                               num_heads=num_heads,
                                               dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs, attention_mask=None, key_padding_mask=None):
        inputs_att, _ = self.self_attn(inputs, inputs, inputs,
                                       attn_mask=attention_mask,
                                       key_padding_mask=key_padding_mask)
        outputs = inputs + self.dropout(inputs_att)
        outputs = self.layer_norm(outputs)
        return outputs


class LSTM(nn.Module):
    def __init__(self, d_model, hidden_dim, num_layers, args):
        """
        hidden_dim: 768, output hidden dimension. 768/2 for each direction.
        """
        super(LSTM, self).__init__()
        self.hidden_dim = int(hidden_dim / 2)
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=d_model,
                            hidden_size=self.hidden_dim,
                            num_layers=num_layers,
                            bidirectional=True,
                            batch_first=True)
        # （batch_size,seq_len,input_size)

        self.device = args.device

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        return out


class CRF(nn.Module):
    # borrow the code from
    # https://github.com/allenai/allennlp/blob/master/allennlp/modules/conditional_random_field.py
    def __init__(self, num_tags, constraints=None, include_start_end_transitions=None):
        """
        if tokens do not start or end with special tokens(i.e. [CLS]),
        set include_start_end_transitions to True.
        """
        super(CRF, self).__init__()
        self.num_tags = num_tags
        self.include_start_end_transitions = include_start_end_transitions
        self.transitions = nn.Parameter(torch.Tensor(self.num_tags, self.num_tags))
        constraint_mask = torch.Tensor(self.num_tags + 2, self.num_tags + 2).fill_(1.)
        if include_start_end_transitions:
            self.start_transitions = nn.Parameter(torch.Tensor(num_tags))
            self.end_transitions = nn.Parameter(torch.Tensor(num_tags))
        # register the constraint_mask
        self.constraint_mask = nn.Parameter(constraint_mask, requires_grad=False)
        self.reset_parameters()

    def forward(self, inputs, tags, mask=None):
        """
        :param inputs: (bsz, seq_len, num_tags), logits calculated from a linear layer
        :param tags: (bsz, seq_len)
        :param mask: (bsz, seq_len), mask for the padding token
        return: -log likelihood
        """
        if mask is None:
            mask = torch.ones(*tags.size(), dtype=torch.long)
        log_denominator = self._input_likelihood(inputs, mask)
        log_numerator = self._joint_likelihood(inputs, tags, mask)
        return torch.sum(log_numerator - log_denominator)

    def reset_parameters(self):
        """
        initialize the parameters in CRF
        """
        nn.init.xavier_normal_(self.transitions)
        if self.include_start_end_transitions:
            nn.init.normal_(self.start_transitions)
            nn.init.normal_(self.end_transitions)

    def _input_likelihood(self, logits, mask):
        """
        :param logits:
        emission score calculated by a linear layer, shape: (batch_size, seq_len, num_tags)
        """
        bsz, seq_len, num_tags = logits.size()
        # Transpose batch size and sequence dimensions
        mask = mask.float().transpose(0, 1).contiguous()
        logits = logits.transpose(0, 1).contiguous()

        # Initial alpha is the (batch_size, num_tags) tensor of likelihoods combining the
        # transitions to the initial states and the logits for the first timestep.
        if self.include_start_end_transitions:
            alpha = self.start_transitions.view(1, num_tags) + logits[0]
        else:
            alpha = logits[0]

        for t in range(1, seq_len):
            # iteration starts from 1
            emit_scores = logits[t].view(bsz, 1, num_tags)
            transition_scores = self.transitions.view(1, num_tags, num_tags)
            broadcast_alpha = alpha.view(bsz, num_tags, 1)

            # calculate the likelihood
            inner = broadcast_alpha + emit_scores + transition_scores

            # mask the padded token when met the padded token, retain the previous alpha
            alpha = (logsumexp(inner, 1) * mask[t].view(bsz, 1) + alpha * (1 - mask[t]).view(bsz, 1))
            # alpha [batch_size, num_tag]
        # Every sequence needs to end with a transition to the stop_tag.
        if self.include_start_end_transitions:
            stops = alpha + self.end_transitions.view(1, num_tags)
        else:
            stops = alpha

        # Finally we log_sum_exp along the num_tags dim, result is (batch_size,)
        return logsumexp(stops)

    def _joint_likelihood(self, logits, tags, mask):
        """
        calculate the likelihood for the input tag sequence
        :param logits:
        :param tags: shape: (bsz, seq_len)
        :param mask: shape: (bsz, seq_len)
        :return:
        """
        bsz, seq_len, _ = logits.size()
        tags = tags * mask
        # Transpose batch size and sequence dimensions:
        logits = logits.transpose(0, 1).contiguous()
        mask = mask.float().transpose(0, 1).contiguous()
        tags = tags.transpose(0, 1).contiguous()

        # Start with the transition scores from start_tag to the first tag in each input
        if self.include_start_end_transitions:
            score = self.start_transitions.index_select(0, tags[0])
        else:
            score = 0.0

        for t in range(seq_len - 1):
            current_tag, next_tag = tags[t], tags[t + 1]
            # tags[t] [batch_size,]
            # print(f"tags[t] shape {tags[t].shape}")
            # print("current tag", current_tag)
            # print("next tag",next_tag)
            # The scores for transitioning from current_tag to next_tag
            transition_score = self.transitions[current_tag.view(-1), next_tag.view(-1)]
            # The score for using current_tag

            emit_score = logits[t].gather(1, current_tag.view(bsz, 1)).view(-1)
            score = score + transition_score * mask[t + 1] + emit_score * mask[t]

        last_tag_index = mask.sum(0).long() - 1
        last_tags = tags.gather(0, last_tag_index.view(1, bsz)).squeeze(0)

        # Compute score of transitioning to `stop_tag` from each "last tag".
        if self.include_start_end_transitions:
            last_transition_score = self.end_transitions.index_select(0, last_tags)
        else:
            last_transition_score = 0.0

        last_inputs = logits[-1]  # (batch_size, num_tags)
        last_input_score = last_inputs.gather(1, last_tags.view(-1, 1))  # (batch_size, 1)
        last_input_score = last_input_score.squeeze()  # (batch_size,)

        score = score + last_transition_score + last_input_score * mask[-1]

        return score

    def viterbi_tags(self, logits, mask):
        """
        :param logits: (bsz, seq_len, num_tags), emission scores
        :param mask:
        :return:
        """
        _, max_seq_len, num_tags = logits.size()

        # Get the tensors out of the variables
        logits, mask = logits.data, mask.data

        # Augment transitions matrix with start and end transitions
        start_tag = num_tags
        end_tag = num_tags + 1
        transitions = torch.Tensor(num_tags + 2, num_tags + 2).fill_(-10000.)

        # Apply transition constraints
        constrained_transitions = (
                self.transitions * self.constraint_mask[:num_tags, :num_tags] +
                -10000.0 * (1 - self.constraint_mask[:num_tags, :num_tags])
        )

        transitions[:num_tags, :num_tags] = constrained_transitions.data

        if self.include_start_end_transitions:
            transitions[start_tag, :num_tags] = (
                    self.start_transitions.detach() * self.constraint_mask[start_tag, :num_tags].data +
                    -10000.0 * (1 - self.constraint_mask[start_tag, :num_tags].detach())
            )
            transitions[:num_tags, end_tag] = (
                    self.end_transitions.detach() * self.constraint_mask[:num_tags, end_tag].data +
                    -10000.0 * (1 - self.constraint_mask[:num_tags, end_tag].detach())
            )
        else:
            transitions[start_tag, :num_tags] = (-10000.0 *
                                                 (1 - self.constraint_mask[start_tag, :num_tags].detach()))
            transitions[:num_tags, end_tag] = -10000.0 * (1 - self.constraint_mask[:num_tags, end_tag].detach())

        best_paths = []
        # Pad the max sequence length by 2 to account for start_tag + end_tag.
        tag_sequence = torch.Tensor(max_seq_len + 2, num_tags + 2)

        for prediction, prediction_mask in zip(logits, mask):
            # perform viterbi decoding sample by sample
            seq_len = torch.sum(prediction_mask)
            # Start with everything totally unlikely
            tag_sequence.fill_(-10000.)
            # At timestep 0 we must have the START_TAG
            tag_sequence[0, start_tag] = 0.
            # At steps 1, ..., sequence_length we just use the incoming prediction
            tag_sequence[1:(seq_len + 1), :num_tags] = prediction[:seq_len]
            # And at the last timestep we must have the END_TAG
            tag_sequence[seq_len + 1, end_tag] = 0.
            viterbi_path = viterbi_decode(tag_sequence[:(seq_len + 2)], transitions)
            viterbi_path = viterbi_path[1:-1]
            best_paths.append(viterbi_path)
        return best_paths


def logsumexp(tensor, dim=-1, keepdim=False):
    """
    :param tensor:
    :param dim:
    :param keepdim:
    :return:
    """
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()


def viterbi_decode(tag_sequence, transition_matrix,
                   tag_observations=None, allowed_start_transitions=None,
                   allowed_end_transitions=None):
    # borrow code from
    # https://github.com/lixin4ever/BERT-E2E-ABSA/blob/master/seq_utils.py
    """
    Perform Viterbi decoding in log space over a sequence given a transition matrix
    specifying pairwise (transition) potentials between tags and a matrix of shape
    (sequence_length, num_tags) specifying unary potentials for possible tags per
    timestep.
    Parameters
    ----------
    tag_sequence : torch.Tensor, required.
        A tensor of shape (sequence_length, num_tags) representing scores for
        a set of tags over a given sequence.
    transition_matrix : torch.Tensor, required.
        A tensor of shape (num_tags, num_tags) representing the binary potentials
        for transitioning between a given pair of tags.
    tag_observations : Optional[List[int]], optional, (default = None)
        A list of length ``sequence_length`` containing the class ids of observed
        elements in the sequence, with unobserved elements being set to -1. Note that
        it is possible to provide evidence which results in degenerate labelings if
        the sequences of tags you provide as evidence cannot transition between each
        other, or those transitions are extremely unlikely. In this situation we log a
        warning, but the responsibility for providing self-consistent evidence ultimately
        lies with the user.
    allowed_start_transitions : torch.Tensor, optional, (default = None)
        An optional tensor of shape (num_tags,) describing which tags the START token
        may transition *to*. If provided, additional transition constraints will be used for
        determining the start element of the sequence.
    allowed_end_transitions : torch.Tensor, optional, (default = None)
        An optional tensor of shape (num_tags,) describing which tags may transition *to* the
        end tag. If provided, additional transition constraints will be used for determining
        the end element of the sequence.
    Returns
    -------
    viterbi_path : List[int]
        The tag indices of the maximum likelihood tag sequence.
    viterbi_score : torch.Tensor
        The score of the viterbi path.
    """
    sequence_length, num_tags = list(tag_sequence.size())

    has_start_end_restrictions = allowed_end_transitions is not None or allowed_start_transitions is not None

    if has_start_end_restrictions:

        if allowed_end_transitions is None:
            allowed_end_transitions = torch.zeros(num_tags)
        if allowed_start_transitions is None:
            allowed_start_transitions = torch.zeros(num_tags)

        num_tags = num_tags + 2
        new_transition_matrix = torch.zeros(num_tags, num_tags)
        new_transition_matrix[:-2, :-2] = transition_matrix

        # Start and end transitions are fully defined, but cannot transition between each other.
        # pylint: disable=not-callable
        allowed_start_transitions = torch.cat([allowed_start_transitions, torch.tensor([-math.inf, -math.inf])])
        allowed_end_transitions = torch.cat([allowed_end_transitions, torch.tensor([-math.inf, -math.inf])])
        # pylint: enable=not-callable

        # First define how we may transition FROM the start and end tags.
        new_transition_matrix[-2, :] = allowed_start_transitions
        # We cannot transition from the end tag to any tag.
        new_transition_matrix[-1, :] = -math.inf

        new_transition_matrix[:, -1] = allowed_end_transitions
        # We cannot transition to the start tag from any tag.
        new_transition_matrix[:, -2] = -math.inf

        transition_matrix = new_transition_matrix

    if tag_observations:
        if len(tag_observations) != sequence_length:
            raise Exception("Observations were provided, but they were not the same length "
                            "as the sequence. Found sequence of length: {} and evidence: {}"
                            .format(sequence_length, tag_observations))
    else:
        tag_observations = [-1 for _ in range(sequence_length)]

    if has_start_end_restrictions:
        tag_observations = [num_tags - 2] + tag_observations + [num_tags - 1]
        zero_sentinel = torch.zeros(1, num_tags)
        extra_tags_sentinel = torch.ones(sequence_length, 2) * -math.inf
        tag_sequence = torch.cat([tag_sequence, extra_tags_sentinel], -1)
        tag_sequence = torch.cat([zero_sentinel, tag_sequence, zero_sentinel], 0)
        sequence_length = tag_sequence.size(0)

    path_scores = []
    path_indices = []

    if tag_observations[0] != -1:
        one_hot = torch.zeros(num_tags)
        one_hot[tag_observations[0]] = 100000.
        path_scores.append(one_hot)
    else:
        path_scores.append(tag_sequence[0, :])

    # Evaluate the scores for all possible paths.
    for timestep in range(1, sequence_length):
        # Add pairwise potentials to current scores.
        summed_potentials = path_scores[timestep - 1].unsqueeze(-1) + transition_matrix
        scores, paths = torch.max(summed_potentials, 0)

        # If we have an observation for this timestep, use it
        # instead of the distribution over tags.
        observation = tag_observations[timestep]
        # Warn the user if they have passed
        # invalid/extremely unlikely evidence.
        if tag_observations[timestep - 1] != -1 and observation != -1:
            if transition_matrix[tag_observations[timestep - 1], observation] < -10000:
                logger.warning("The pairwise potential between tags you have passed as "
                               "observations is extremely unlikely. Double check your evidence "
                               "or transition potentials!")
        if observation != -1:
            one_hot = torch.zeros(num_tags)
            one_hot[observation] = 100000.
            path_scores.append(one_hot)
        else:
            path_scores.append(tag_sequence[timestep, :] + scores.squeeze())
        path_indices.append(paths.squeeze())

    # Construct the most likely sequence backwards.
    viterbi_score, best_path = torch.max(path_scores[-1], 0)
    viterbi_path = [int(best_path.numpy())]
    for backward_timestep in reversed(path_indices):
        viterbi_path.append(int(backward_timestep[viterbi_path[-1]]))
    # Reverse the backward path.
    viterbi_path.reverse()

    if has_start_end_restrictions:
        viterbi_path = viterbi_path[1:-1]
    return viterbi_path
