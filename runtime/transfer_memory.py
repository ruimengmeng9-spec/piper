"""Transfer the trained MLP into a 33-state GRU with a near-identity start."""
import torch

def initialize_memory_transfer(policy, source_state):
    target = policy.state_dict()
    compatible = {key:value for key,value in source_state.items()
                  if key in target and target[key].shape==value.shape}
    missing = set(target)-set(compatible)
    if any(not key.startswith(('memory_a.','memory_c.')) for key in missing):
        raise RuntimeError(f'Non-memory parameters cannot be transferred: {sorted(missing)}')
    policy.load_state_dict(compatible,strict=False)
    scale=.1
    with torch.no_grad():
        for memory in [policy.memory_a,policy.memory_c]:
            rnn=memory.rnn
            size=rnn.hidden_size
            if rnn.input_size!=size or rnn.num_layers!=1:
                raise ValueError('Near-identity transfer requires one square GRU layer')
            rnn.weight_ih_l0.zero_()
            rnn.weight_hh_l0.zero_()
            rnn.bias_ih_l0.zero_()
            rnn.bias_hh_l0.zero_()
            rnn.weight_ih_l0[2*size:].copy_(scale*torch.eye(size,device=rnn.weight_ih_l0.device))
            rnn.bias_ih_l0[size:2*size].fill_(-5.)
            rnn.bias_ih_l0[:size].fill_(2.)
        # Normalization precedes the GRU. Compensate its initial small gain in
        # the first MLP layer while allowing the GRU to learn persistent state.
        policy.actor[0].weight.div_(scale)
        policy.critic[0].weight.div_(scale)
    print('MLP_TO_GRU_TRANSFER',len(compatible),'parameters/buffers',flush=True)
