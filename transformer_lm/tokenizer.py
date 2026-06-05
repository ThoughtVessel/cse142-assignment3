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
        self.special_tokens = special_tokens

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token IDs."""
        # Variable to store special tokens and normal strings divided but in order
        divided_string_locations = []
        sc_ids = []

        # First check to make sure that special tokens exists
        if self.special_tokens is not None:

            # If it does exist, go through every letter in the text
            for i in range(len(text)):

                # For every letter, go through every special character
                for j in range(len(self.special_tokens)):

                    # Make sure the special character isn't longer than the remaining text
                    st_length = len(self.special_tokens[j])
                    if st_length <= (len(text) - i):

                        # If its the token, add its vocab integer id to the list
                        if text[i : i + st_length] == self.special_tokens[j]:
                            sc_ids.append(len(self.vocab) - len(self.special_tokens) + j)
                            divided_string_locations.append((i, i + st_length))


        # Now we have the special token locations and ids, we can add the normal text in between
        normal_string_locations = []

        # If there are no special characters found, then just keep entire list
        if len(divided_string_locations) == 0:
            normal_string_locations.append((0, len(text)))
        else:
            # If they are found, first check if they are the first or not
            if divided_string_locations[0][0] > 0:
                normal_string_locations.append((0, divided_string_locations[0][0]))

            # Use the data gathered from analyzing the special characters, to recaclulate the stops and starts of the normal words
            for i in range(len(divided_string_locations) - 1):
                normal_string_locations.append((divided_string_locations[i][1], divided_string_locations[i + 1][0]))

            # Need to correct if its the last token
            if divided_string_locations[-1][1] < len(text):
                normal_string_locations.append((divided_string_locations[-1][1], len(text)))

        
        # Now, use the normal string locations to get the strings from the text
        normal_strings = []
        for start, end in normal_string_locations:
            normal_strings.append(text[start:end])
        
        # Convert to list of integer values for the normal strings
        content = []
        for string in normal_strings:
            content.append(list(string.encode("utf-8")))

        
        # Loop and merge for evvery one of the normal string, then add the special characters back in
        merged_results = []
        for content_string in content:
            # Add the correct new id when merging (from UTF)
            for i in range(len(self.merges)):
                tempcontent = []
                byte1 = self.merges[i][0]
                byte2 = self.merges[i][1]
                k = 0
                while k < len(content_string):
                    if k < len(content_string) - 1 and content_string[k] == byte1 and content_string[k + 1] == byte2:
                        tempcontent.append(256 + i)
                        k += 2
                    else:
                        tempcontent.append(content_string[k])
                        k += 1
                content_string = tempcontent
            # Save the merged result for normal string
            merged_results.append(content_string)

        # Use the divided and normal string locations to add the special characters and normal strings back together in order
        segments = []

        for idx, (start, end) in enumerate(divided_string_locations):
            segments.append((start, [sc_ids[idx]]))

        for idx, (start, end) in enumerate(normal_string_locations):
            segments.append((start, merged_results[idx]))

        segments.sort(key=lambda x: x[0])

        # Flatten into final result list
        result = []
        for _, tokens in segments:
            result.extend(tokens)
        return result

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back into a string."""
        # Go through list and use vocab to add back
        output_bytes = b""
        for id in ids:
            output_bytes += self.vocab[id]
        return output_bytes.decode("utf-8", errors="replace")
