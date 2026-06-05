"""Byte-level BPE tokenizer (no regex pre-tokenization)."""

from __future__ import annotations

from collections import Counter


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[int, int]]]:
    """Train a byte-level BPE tokenizer.

    Merge order: at each step, merge the most frequent adjacent pair.
    Break ties by selecting the pair with the smallest ``(id1, id2)``
    in lexicographic (tuple) order.

    IDs 0–255 are single bytes. Merge tokens get IDs starting from 256.
    Special tokens get the highest IDs in the vocab.

    Args:
        input_path: Path to a UTF-8 text file.
        vocab_size: Target vocabulary size (>= 256 + len(special_tokens)).
        special_tokens: Optional special token strings.

    Returns:
        vocab: ``dict[int, bytes]`` mapping token ID to byte string.
        merges: ``list[tuple[int, int]]`` merge pairs in order.
    """
    
    # Calculate number of tokens to add
    tokens_to_add = vocab_size - (256 + len(special_tokens))

    # Initialize starting vocabulary (special characters are added at end, not here)
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])

    # Initialize merges (list of tuples)
    merges = []

    # Get string for input path
    with open(input_path, "r", encoding="utf-8") as file:
        content = file.read()

    # Convert to integer list
    content = list(content.encode("utf-8"))
    
    # Go through loop for each token to add
    for i in range(tokens_to_add):
        # Track pairs
        pairs = {}

        # Inner loop to go through corpus and add every pair
        for j in range(len(content) - 1):
            # Get new pair
            new_pair = (content[j], content[j+1])

            # Check if in pairs
            if new_pair in pairs:
                pairs[new_pair] += 1
            else:
                pairs[new_pair] = 1

            
        # Sort by occurance and alphabetically as backup
        sorted_list = sorted(pairs.items(), key=lambda item: (-item[1], item[0]))

        # New pair
        best_pair_tuple = sorted_list[0][0]
        vocab[256 + i] = vocab[best_pair_tuple[0]] + vocab[best_pair_tuple[1]]

        # Add to pairs
        merges.append(best_pair_tuple)

        # Finally, change content to add all new combos
        temp_content = []
        k = 0
        while k < len(content):
            if k < len(content) - 1 and (content[k] == best_pair_tuple[0] and content[k + 1] == best_pair_tuple[1]):
                temp_content.append(256 + i)
                k += 2
            else:
                temp_content.append(content[k])
                k += 1
        
        content = temp_content

    
    # Add special tokens to vocab
    if special_tokens is not None:
        for i, token in enumerate(special_tokens):
            vocab[256 + tokens_to_add + i] = token.encode("utf-8")
    

    # Return vocab and merges
    return vocab, merges







class BPETokenizer:
    """Byte-level BPE tokenizer."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[int, int]],
        special_tokens: list[str] | None = None,
    ) -> None:
        # Save vocab, merges, special tokens for later.
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # Build mapping from special token string -> its ID
        self.special_token_ids = {}
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            for id, b in vocab.items():
                if b == token_bytes:
                    self.special_token_ids[token] = id
                    break

        # Sort special tokens longest-first for greedy matching
        self.sorted_special = sorted(self.special_tokens, key=len, reverse=True)

    def _bpe_encode(self, text: str) -> list[int]:
        """Apply BPE to a plain text segment (no special tokens)."""
        # Convert to list
        content = list(text.encode("utf-8"))

        # When merging we find correct new id with merges[0] = vocab 256, etc
        for i in range(len(self.merges)):
            tempcontent = []
            byte1 = self.merges[i][0]
            byte2 = self.merges[i][1]
            k = 0
            while k < len(content):
                if k < len(content) - 1 and content[k] == byte1 and content[k + 1] == byte2:
                    tempcontent.append(256 + i)
                    k += 2
                else:
                    tempcontent.append(content[k])
                    k += 1
            content = tempcontent
        return content

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token IDs."""
        ids = []
        i = 0
        while i < len(text):
            # Try to match a special token at position i (longest first)
            matched = False
            for special in self.sorted_special:
                if text[i:i + len(special)] == special:
                    ids.append(self.special_token_ids[special])
                    i += len(special)
                    matched = True
                    break
            if not matched:
                # Collect text up to the next special token boundary
                j = i + 1
                while j < len(text):
                    if any(text[j:j + len(s)] == s for s in self.sorted_special):
                        break
                    j += 1
                ids.extend(self._bpe_encode(text[i:j]))
                i = j
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back into a string."""
        # Go through list and use vocab to add back
        output_bytes = b""
        for id in ids:
            output_bytes += self.vocab[id]
        return output_bytes.decode("utf-8", errors="replace")
