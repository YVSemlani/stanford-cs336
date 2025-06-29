import os
import sys
import regex as re
import pickle

from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed

def find_chunk_boundaries(file, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pretokenize(text, special_tokens):
    """
    Pretokenize the text into a list of tokens

    Args:
        text (str): The text to pretokenize.

    Returns:
        list: A list of byte tuples
    """

    # regex to escape the special tokens for splitting
    special_token_escape = "|".join(re.escape(token) for token in special_tokens)

    # First split by special tokens
    if special_tokens:
        parts = re.split(f"({special_token_escape})", text)
    else:
        parts = [text]
    
    # Standard GPT-2 regex pattern (without special tokens)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # iterate over the parts and convert pretoken strings to byte tuples
    pretokenized_text = []
    
    for part in tqdm(parts, disable=True):
        if part in special_tokens:
            # if the part is a special token, add it as a single tuple of bytes
            pretokenized_text.append((bytes(part, 'utf-8'), ))
        elif part:  # if part is not empty
            # use regex to find all matches and extract the full matched text
            parsed_text = [match.group(0) for match in re.finditer(PAT, part)]
            
            for word in parsed_text:
                # split the word into a tuple of bytes i.e. (b'h', b'e', b'l', b'l', b'o')
                word = word.encode('utf-8')
                word = tuple(word[i : i+1] for i in range(len(word)))
                pretokenized_text.append(word)

    return pretokenized_text

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.special_tokens = special_tokens
        
        self.vocab = vocab
        
        # add the special tokens to the vocab
        for idx, token in enumerate(special_tokens):
            self.vocab[idx + len(vocab)] = bytes(token, 'utf-8')
        
        # list to store the merges in learning order
        self.merges = merges

    def train(self, input_path, vocab_size, num_processes=12, num_chunks=12):
        # load file
        with open(input_path, 'rb') as f:

            # get chunk boundaries
            chunk_boundaries = find_chunk_boundaries(f, num_chunks, "<|endoftext|>".encode("utf-8"))

            # seperate into chunks of text
            text_chunks = []
            for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
                f.seek(start)
                text_chunks.append(f.read(end - start).decode("utf-8", errors="ignore"))

            # pretokenize the text
            pretokenized_text = []

            # pretokenize each text chunk in parallel
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = []
                for text_chunk in text_chunks:
                    #print(text_chunk)
                    futures.append(executor.submit(pretokenize, text_chunk, self.special_tokens))
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
                    pretokenized_text.extend(future.result())

        # create dictionaries to store the frequency of pairs and tokens
        # create occurrences cache to store indices at which pairs occur
        pairs_freq = {}
        occurrences = {}
        
        for token_idx, token in tqdm(enumerate(pretokenized_text), total=len(pretokenized_text), desc="Building frequencies", leave=False):
            for i in range(len(token) - 1): # iterate over the pairs in the token
                pair = (token[i], token[i + 1]) # get the pair
                pairs_freq[pair] = pairs_freq.get(pair, 0) + 1 # increment the frequency of the pair
                
                # update the occurrences cache
                if pair not in occurrences.keys():
                    occurrences[pair] = set()
                occurrences[pair].add(token_idx)

        total_merges = vocab_size - len(self.vocab)
        with tqdm(total=total_merges, desc="Learning merges") as pbar:
            while len(self.vocab) < vocab_size: # while the vocab is less than the desired size
                pbar.update(1)
                
                # identify a merge candidate
                # sort the pairs by frequency and then by the pair itself
                merge_pair, _ = max(
                    pairs_freq.items(),
                    key=lambda kv: (kv[1], kv[0])
                )

                #print(f"Merge pair: {merge_pair}")
                # create the merged token
                merged_token = merge_pair[0] + merge_pair[1]

                # run the merge with updates to the frequency dictionaries and occurrences cache

                token_transform_cache = {} # cache to store the pretokens we've already applied the merge to
                for token_idx in occurrences[merge_pair].copy(): # iterate over the pretokens that we know contain the merge pair
                    current_token = pretokenized_text[token_idx] # grab the pretoken from the pretokenized text
                    
                    if current_token in token_transform_cache.keys(): # if we've already applied the merge to this pretoken, replace the pretoken with the merged version
                        pretokenized_text[token_idx], original_pairs, new_pairs = token_transform_cache[current_token]
                        
                        for pair in original_pairs:
                            pairs_freq[pair] = pairs_freq.get(pair, 0) - 1
                            if token_idx in occurrences[pair]:
                                occurrences[pair].discard(token_idx)
                        
                        for pair in new_pairs:
                            pairs_freq[pair] = pairs_freq.get(pair, 0) + 1
                            if pair not in occurrences:
                                occurrences[pair] = set()
                            occurrences[pair].add(token_idx)

                    else: # otherwise, perform the merge and update the frequency dictionaries
                        modified_token = list(current_token) # convert the tuple to a list to allow for in place modifications
                        merge_indices = [] # store the indices where we will insert the merged token
                        pop_indices = [] # store the indices where we will remove the current token

                        original_pairs = []
                        new_pairs = []

                        for i in range(len(current_token) - 1):
                            pair = (current_token[i], current_token[i + 1])
                            original_pairs.append(pair)

                            if pair == merge_pair:
                                merge_indices.append(i)
                                pop_indices.append(i + 1)
                        
                        # apply the merges in place
                        for idx in merge_indices:
                            modified_token[idx] = merged_token

                        # pop in reverse order to avoid index shifting
                        for idx in reversed(pop_indices):
                            modified_token.pop(idx)

                        # get the pairs from modified token
                        for i in range(len(modified_token) - 1):
                            new_pairs.append((modified_token[i], modified_token[i + 1]))                     

                        # Update frequencies: remove old pairs, add new pairs
                        for pair in original_pairs:
                            pairs_freq[pair] = pairs_freq.get(pair, 0) - 1
                            if token_idx in occurrences[pair]:  # Safety check
                                occurrences[pair].discard(token_idx)

                        for pair in new_pairs:
                            pairs_freq[pair] = pairs_freq.get(pair, 0) + 1
                            if pair not in occurrences:
                                occurrences[pair] = set()
                            occurrences[pair].add(token_idx)

                        # turn the modified token back into a tuple so it's hashable and we
                        modified_token = tuple(modified_token)
                        
                        # store the new word, and original vs modified pairs in the cache
                        token_transform_cache[current_token] = (tuple(modified_token), original_pairs, new_pairs)
                        
                        # update the pretokenized text
                        pretokenized_text[token_idx] = tuple(modified_token)

                # add the merge token to the vocab
                self.vocab[len(self.vocab)] = merged_token
                
                # add the merge to the merges list
                self.merges.append(merge_pair)

        return pretokenized_text
    
    def encode(self, text):
        # pretokenize the text
        pretokenized_text = pretokenize(text, self.special_tokens)

        # apply the merges in learned order
        for merge in self.merges:
            # cache to store the tokens we've already applied the merge to (resets after each merge)
            token_transform_cache = {}
            for token_idx, token in enumerate(pretokenized_text):
                if token in token_transform_cache:
                    pretokenized_text[token_idx] = token_transform_cache[token]
                else:
                    merge_idx = 0
                    merge_indices = []

                    modified_token = list(token)
                    while merge_idx < len(token) - 1:
                        if token[merge_idx] == merge[0] and token[merge_idx + 1] == merge[1]:
                            merge_indices.append(merge_idx)
                            merge_idx += 2
                        else:
                            merge_idx += 1
                    
                    pop_indices = [idx + 1 for idx in merge_indices]

                    for idx in merge_indices:
                        modified_token[idx] = merge[0] + merge[1]
                    
                    for idx in reversed(pop_indices):
                        modified_token.pop(idx)
                    
                    # update the cache
                    token_transform_cache[token] = tuple(modified_token)

                    # update the pretokenized text
                    pretokenized_text[token_idx] = tuple(modified_token)

        # convert the merged byte values to integer token IDs
        token_ids = []

        # Create a reverse mapping from bytes to token IDs
        vocab_reverse = {v: k for k, v in self.vocab.items()}
        
        # Iteratively map the byte values to token IDs
        for pretoken in pretokenized_text:
            for token in pretoken:
                token_ids.append(vocab_reverse[token])

        return token_ids
    
    def decode(self, tokens):
        decoded_text = []
        for token_id in tokens:
            decoded_text.append(self.vocab[token_id].decode('utf-8'))
        return ''.join(decoded_text)

    def get_vocab(self):
        return self.vocab
    
    def get_merges(self):
        return self.merges

if __name__ == "__main__":
    print("Tokenizer interal testing beginning")
    print("="*100)

    vocab = {i : bytes([i]) for i in range(256)}
    merges = []
    special_tokens = ['<|endoftext|>']

    tokenizer = Tokenizer(vocab, merges, special_tokens)
    pretokenized_text = tokenizer.train('data/TinyStoriesV2-GPT4-valid.txt', vocab_size=270, num_chunks=100)

    vocab = tokenizer.get_vocab()
    merges = tokenizer.get_merges()
    print(f'Vocab \n {"="*100} \n {vocab} \n {"="*100}')
    print(f'Merges \n {"="*100} \n {merges} \n {"="*100}')

    # Save vocab to pickle file
    with open('data/tokenizers/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    print("Vocab saved to data/tokenizers/vocab.pkl")

    # Save merges to pickle file
    with open('data/tokenizers/merges.pkl', 'wb') as f:
        pickle.dump(merges, f)
    print("Merges saved to data/tokenizers/merges.pkl")

    encoded_text = tokenizer.encode("hello, world! <|endoftext|>")
    print(f"Encoded text: {encoded_text}")
    print(f"Decoded text: {tokenizer.decode(encoded_text)}")

    