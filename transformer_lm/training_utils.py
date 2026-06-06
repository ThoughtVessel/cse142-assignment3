"""Training utilities: batching and text generation."""

from __future__ import annotations

import torch

from transformer_lm.nn_utils import softmax


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of input-target pairs from a 1-D token array.

    Args:
        data: 1-D tensor of token IDs.
        batch_size: Number of examples per batch.
        context_length: Number of tokens in each sequence.
        device: Device to place tensors on.

    Returns:
        ``(x, y)`` both of shape ``(batch_size, context_length)``.
    """
    
    # Raise ValueError if data is too short
    if len(data) <= context_length:
        raise ValueError("no valid shifted target window exists.")

    # Generate random numbers 0≤start<len(data)−context_length
    random_nums = torch.randint(0, len(data) - context_length, (batch_size,))

    # Use random numbers to get the necessary strings
    big_lists = torch.stack([data[s : s + context_length + 1] for s in random_nums])

    # Take entire first part of list for x
    x = big_lists[:, :context_length].long()
    y = big_lists[:, 1:].long()

    # Add both tensors to device
    x = x.to(device)
    y = y.to(device)

    # Return x and y as a tuple, shape (batch_size, context_length)
    return (x, y)


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    context_length: int | None = None,
) -> list[int]:
    """Autoregressively generate tokens from a language model.

    Args:
        model: Maps ``(B, T)`` integer input to ``(B, T, vocab_size)`` logits.
        prompt_ids: Starting token IDs.
        max_new_tokens: Number of new tokens to generate.
        temperature: Sampling temperature.
        context_length: Maximum context window (defaults to ``model.context_length``).

    Returns:
        List of token IDs (prompt + generated).
    """

    # Get context length
    if context_length is None:
        context_length = model.context_length

    ######################################
    #### Generate the tokens of the future
    ######################################
    token_generated = 0
    # Generate working token list
    prompt = list(prompt_ids)

    # Loop through
    while token_generated < max_new_tokens:
        # Create temp loop prompt of length <= context_length
        loop_prompt = prompt[-context_length:]

        # Generate new token with model
        device = next(model.parameters()).device
        input_tensor = torch.tensor([loop_prompt], device=device)
        model_output = model(input_tensor)
        last_token_output = model_output[:, -1, :]

        # Divide by temp and apply sfotmax
        temp_output = last_token_output / temperature
        soft_output = softmax(temp_output)

        # Sample the distribution to grab the token
        chosen_token = torch.multinomial(soft_output, num_samples=1)

        # Add chosen token to prompt and increment token_generated
        prompt.append(chosen_token.item())
        token_generated += 1
    
    # Return prompt
    return prompt
    
    



