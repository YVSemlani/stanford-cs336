from concurrent import futures
import os
import sys
import regex as re
from tqdm import tqdm

# Add the parent directory to Python path so cs336_basics package can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs336_basics.pretokenization_example import find_chunk_boundaries

from concurrent.futures import ProcessPoolExecutor, as_completed

def parse_text_worker(file_path, start, end, special_tokens):
    """Standalone function for worker processes - no class instance needed"""
    
    with open(file_path, 'rb') as f:
        # regex to remove special tokens
        delimiter = "|".join(re.escape(token) for token in special_tokens)
        
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        
        sub_chunks = re.split(delimiter, chunk)
        
        pretokenized_chunks = []
        
        for sub_chunk in tqdm(sub_chunks, desc=f"Parsing text...", leave=False):
            if sub_chunk != '':
                parsed_text, tokens = pretokenize_text(sub_chunk)
                pretokenized_chunks.append(tokens)
        return pretokenized_chunks

def pretokenize_text(text):
    """Standalone pretokenization function"""
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    parsed_text = re.findall(PAT, text)
    
    word_frequency = {}
    for word in parsed_text:
        word = word.encode('utf-8')
        word = tuple(word[i : i+1] for i in range(len(word)))
        
        if word in word_frequency:
            word_frequency[word] += 1
        else:
            word_frequency[word] = 1
    
    return parsed_text, word_frequency

class BPETokenizer:
    def __init__(self, vocab_size, special_tokens):
        self.vocab_size = vocab_size

        self.vocab = [bytes([i]) for i in range(256)]
        self.vocab += [bytes(token, 'utf-8') for token in special_tokens]

        self.special_tokens = special_tokens

        self.offset = len(special_tokens)
  
        self.merges = []
    
    def merge(self, corpus, pairs, occurrences):
        """
        Replaces seperated instances of the merge token with a single merge token.

        Args:
            corpus (dict): A dictionary mapping byte tuples to their frequencies. Must span the entire training corpus.
            merge_token (tuple): a tuple containing two byte values that represent chars
        """

        # sort the pairs by frequency and then by the pair itself
        max_pair, max_count = max(
            pairs.items(),
            key=lambda kv: (kv[1], kv[0])
        )

        # create the merged token
        merged_token = max_pair[0] + max_pair[1]

        # now we need to replace instances of the max pair with the merged token
        new_corpus = {}
        for word, freq in corpus.items():
            new_word = []
            idx = 0
            while idx < len(word):
                if idx + 1 < len(word):
                    if (word[idx], word[idx + 1]) == max_pair:
                        new_word.append(merged_token)
                        idx += 2
                    else:
                        new_word.append(word[idx])
                        idx += 1
                else:
                    new_word.append(word[idx])
                    idx += 1

            new_word = tuple(new_word)
            new_corpus[new_word] = freq

        # update the pairs according to the merge token and occurrences cache


        return new_corpus, merged_token, max_pair

    def train(self, input_path, num_processes=12):
        """
        Train the BPE tokenizer on a text corpus.
        
        This method implements the Byte Pair Encoding algorithm to learn merge rules
        from the input text file. It iteratively finds the most frequent pair of 
        adjacent tokens and merges them into a new token until the vocabulary size
        is reached.
        
        Args:
            input_path (str): Path to the input text file containing the training corpus.
                             The file should be a plain text file with UTF-8 encoding.
        
        Returns:
            vocab (list): The final vocabulary including special tokens, base bytes,
                               and learned BPE tokens. This list should contain only bytes
            merges (list): List of merge rules in the order they were learned.
                                Each merge is a tuple of (token1, token2) that were merged.
        
        Raises:
            FileNotFoundError: If the input file does not exist.
            ValueError: If the file is empty or cannot be processed.
        """

        # store pretokenized chunks
        pretokenized_chunks = []

        # load input file
        with open(input_path, 'rb') as f:

            # generate chunk indices for the file
            boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))

            # regex to remove special tokens
            delimiter = "|".join(re.escape(token) for token in self.special_tokens)

            # parallel implementation of the parsing and pretokenization step

            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                tasks = []
                for start, end in zip(boundaries[:-1], boundaries[1:]):
                    print(f"Submitting task for {start} to {end}")
                    tasks.append(executor.submit(parse_text_worker, input_path, start, end, self.special_tokens))

                # Collect results as they complete (non-blocking)
                for task in as_completed(tasks):
                    print(f"Task completed!")
                    chunk_tokens = task.result()
                    pretokenized_chunks.extend(chunk_tokens)  # Add results as they arrive

        # combine the chunks into a single corpus
        corpus = {}

        for chunk in tqdm(pretokenized_chunks, desc="Combining chunks...", unit="chunk"):
            for byte_string, freq in chunk.items():
                if byte_string in corpus.keys():
                    corpus[byte_string] += freq
                else:
                    corpus[byte_string] = freq

        # run one pass to get the pairs
        pairs = {}

        for word, freq in corpus.items():
            for i in range(len(word) - 1): # iterate over the pairs
                pair = (word[i], word[i + 1])
                if pair in pairs.keys(): # if the pair is already in the pairs dictionary, increment the frequency
                    pairs[pair] += freq
                else: # if the pair is not in the pairs dictionary, add it with the frequency
                    pairs[pair] = freq

        # apply merging to the corpus until we hit the vocab size
        total_merges = self.vocab_size - len(self.vocab)
        with tqdm(total=total_merges, desc="Training BPE...", unit="merge") as pbar:
            while len(self.vocab) < self.vocab_size:
                corpus, pairs, merge_token, merge_token_pair = self.merge(corpus, pairs)

                # add the merge token tuple to the merges list
                self.merges.append(merge_token_pair)

                # add the merge token to the vocab
                self.vocab.append(merge_token)
                
                # update progress bar
                pbar.update(1)
    
    def encode(self, text):
        """
        Encode text into a list of token IDs.
        """

        # output tokens
        output_tokens = []

        # replace all special tokens with their corresponding vocab indices
        for special_token in self.special_tokens:
            text = text.replace(special_token, self.vocab[255 + self.offset + self.special_tokens.index(special_token)])

        # regex string provided by the course (matches GPT-2)
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # use regex to parse text
        parsed_text = re.findall(PAT, text)

        # apply merges to each word in the parsed text
        for word in parsed_text:

            if type(word) != str: # if the word is a special token, add it to the output tokens
                output_tokens.append(word)
                continue
            
            word = word.encode('utf-8')
            word_with_merges = list(word)

            for merge_idx, merge in self.merges: # iterate over all the merges
                pops = []
                for idx, pair in enumerate(zip(word_with_merges, word_with_merges[1:])): # iterate over the pairs but backwards to avoid indexing issues
                    if pair == merge:
                        merge_vocab_idx = self.vocab[255 + self.offset + merge_idx]
                        word_with_merges[idx] = merge_vocab_idx
                        pops.append(idx + 1)

                for idx in reversed(pops): # do pops after the loop to avoid indexing issues and reverse so index changes don't matter
                    word_with_merges.pop(idx)
                

            word_with_merges = tuple(word_with_merges)
        
        output_tokens += word_with_merges
        
        return output_tokens
            

    def get_vocab(self):
        """
        Create a dictionary mapping token IDs to tokens from the vocabulary.
        
        Returns:
            dict: A dictionary where keys are token IDs (integers) and values are the corresponding tokens (bytes).
        """
        return {i: token for i, token in enumerate(self.vocab)}
    
    def get_merges(self):
        return self.merges
        


            
    
if __name__ == "__main__":
    print("Tokenizer interal testing beginning\n\n")
    tokenizer = BPETokenizer(256 + 10, ['<|endoftext|>'])
    tokenizer.train('data/TinyStoriesV2-GPT4-train.txt')

    vocab = tokenizer.get_vocab()
    merges = tokenizer.get_merges()
    print(f'Vocab \n\n {vocab} \n\n')
    print(f'Merges \n\n {merges} \n\n')