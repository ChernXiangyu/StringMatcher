import pandas as pd
from pathlib import Path
import os
import re
import dask.dataframe as dd
from tqdm.auto import tqdm
from datasketch import MinHash, MinHashLSH
from rapidfuzz import fuzz
from typing import List, Tuple, Dict, Set

pd.set_option('display.max_columns', None)


class StringMatcher:
    """
    Large-scale string matching class using MinHash LSH and RapidFuzz for efficient fuzzy matching.
    """
    
    def __init__(self, num_perm: int = 128, lsh_threshold: float = 0.6, qgram_size: int = 3):
        """
        Initialize the string matcher.
        
        Args:
            num_perm: Number of MinHash hash functions. Higher means more accurate but slower.
            lsh_threshold: LSH similarity threshold. Higher means more precision but possibly lower recall.
            qgram_size: Q-gram length, used to split strings into tokens.
        """
        self.num_perm = num_perm
        self.lsh_threshold = lsh_threshold
        self.qgram_size = qgram_size
    
    def get_qgrams(self, text: str) -> Set[str]:
        """
        Convert a string into a set of Q-grams (Q-shingles) with basic cleaning.
        
        Args:
            text: Input string
            
        Returns:
            Set of Q-grams
        """
        # Normalize case and remove spaces
        text = text.lower().replace(' ', '')
        # Remove special characters but keep important digits and letters
        text = re.sub(r'[^\w]', '', text)
        
        if len(text) < self.qgram_size:
            # For very short strings, just return the whole string as a token
            return {text}
        
        return {text[i:i+self.qgram_size] for i in range(len(text) - self.qgram_size + 1)}
    
    def generate_minhash(self, text: str) -> MinHash:
        """
        Generate a MinHash signature for a given string.
        
        Args:
            text: Input string
            
        Returns:
            MinHash signature
        """
        m = MinHash(num_perm=self.num_perm)
        qgrams = self.get_qgrams(text)
        
        for qgram in qgrams:
            # datasketch requires input as bytes
            m.update(qgram.encode('utf8'))
        return m
    
    def find_candidate_pairs(self, arr_A: List[str], arr_B: List[str]) -> Set[Tuple[str, str]]:
        """
        Use MinHashLSH to find candidate similar pairs between two arrays.
        
        Args:
            arr_A: First string array
            arr_B: Second string array
            
        Returns:
            Set of candidate similar pairs
        """
        # Build LSH index (using array B's strings as index)
        lsh = MinHashLSH(threshold=self.lsh_threshold, num_perm=self.num_perm)
        
        print("--- Stage 1: Generate MinHash signatures and index array B ---")
        
        # Batch insertion for performance
        with lsh.insertion_session() as session:
            for idx, item in enumerate(tqdm(arr_B, desc="Indexing array B")):
                # Use index as key to ensure uniqueness
                key = f"B_{idx}"
                m = self.generate_minhash(item)
                session.insert(key, m)
        
        print(f"Array B indexing complete. Total {len(arr_B)} items.")
        
        # Query array A
        candidate_pairs: Set[Tuple[str, str]] = set()
        
        print("\n--- Stage 2: Query array A and find candidate pairs ---")
        
        for idx_A, item_A in enumerate(tqdm(arr_A, desc="Querying array A")):
            m_A = self.generate_minhash(item_A)
            # Query LSH index to find all possibly similar keys (i.e., indices from array B)
            result_keys = lsh.query(m_A)
            
            if result_keys:
                for key_B in result_keys:
                    # Retrieve original string from array B using the key
                    item_B = arr_B[int(key_B.split('_')[1])]
                    # Store (A string, B string) pair
                    candidate_pairs.add((item_A, item_B))
        
        return candidate_pairs
    
    def compute_fuzzy_scores(self, candidates: Set[Tuple[str, str]], 
                            name_A_col: str = 'name_A', 
                            name_B_col: str = 'name_B') -> pd.DataFrame:
        """
        Compute various similarity metrics for candidate pairs.
        
        Args:
            candidates: Set of candidate similar pairs
            name_A_col: Column name for array A
            name_B_col: Column name for array B
            
        Returns:
            DataFrame containing similarity metrics
        """
        print("\n--- Stage 3: Compute fuzzy matching scores ---")
        
        fuzzy_results = []
        for candidate in tqdm(candidates, desc="Computing similarity"):
            name_A = candidate[0]
            name_B = candidate[1]
            
            # Compute various similarity metrics
            partial_ratio = fuzz.partial_ratio(name_A, name_B)
            ratio = fuzz.ratio(name_A, name_B)
            token_sort_ratio = fuzz.token_sort_ratio(name_A, name_B)
            token_set_ratio = fuzz.token_set_ratio(name_A, name_B)
            partial_token_sort_ratio = fuzz.partial_token_sort_ratio(name_A, name_B)
            partial_token_set_ratio = fuzz.partial_token_set_ratio(name_A, name_B)
            
            fuzzy_results.append({
                name_A_col: name_A,
                name_B_col: name_B,
                'partial_ratio': partial_ratio,
                'ratio': ratio,
                'token_sort_ratio': token_sort_ratio,
                'token_set_ratio': token_set_ratio,
                'partial_token_sort_ratio': partial_token_sort_ratio,
                'partial_token_set_ratio': partial_token_set_ratio
            })
        
        fuzzy_df = pd.DataFrame(fuzzy_results)
        fuzzy_df.sort_values(by='partial_ratio', ascending=False, inplace=True)
        fuzzy_df.reset_index(drop=True, inplace=True)
        
        return fuzzy_df
    
    def match(self, arr_A: List[str], arr_B: List[str],
             name_A_col: str = 'name_A',
             name_B_col: str = 'name_B') -> pd.DataFrame:
        """
        Execute the full string matching pipeline.
        
        Args:
            arr_A: First string array
            arr_B: Second string array
            name_A_col: Column name for array A
            name_B_col: Column name for array B
            
        Returns:
            DataFrame containing match results and similarity metrics
        """
        # Step 1: Find candidate pairs
        candidates = self.find_candidate_pairs(arr_A, arr_B)
        print(f"\nFound {len(candidates)} candidate match pairs")
        
        # Step 2: Compute similarity
        fuzzy_df = self.compute_fuzzy_scores(candidates, name_A_col, name_B_col)
        
        return fuzzy_df


def main():
    """
    Main function: Load Boardex and Earnings Call data and perform matching.
    """
    # Set data paths
    DATA_DIR = Path('/Users/chenxiangyu/Library/CloudStorage/Dropbox-ChicagoBooth/Xiangyu Chen/datasets/')
    BOARDEX_DIR = DATA_DIR / 'boardex'
    EARNINGS_CALL_DIR = DATA_DIR / 'earnings_call' / 'processed' / 'streetevents_parsed'
    
    print("=== Loading data ===")
    
    # Load Earnings Call data
    ec_df = dd.read_parquet(EARNINGS_CALL_DIR / 'participant').compute()
    ec_df = ec_df.drop_duplicates(['name', 'org', 'position'])[['name', 'org', 'position']].reset_index(drop=True)
    print(f"Earnings Call participants: {len(ec_df)}")
    
    # Load Boardex data
    dir_df = pd.read_parquet(BOARDEX_DIR / 'na_dir_profile_details.pqt')
    print(f"Boardex directors: {len(dir_df)}")
    
    # Extract unique names
    boardex_names = list(dir_df['directorname'].unique())
    ec_names = list(ec_df['name'].unique())
    
    print(f"Unique Boardex names: {len(boardex_names)}")
    print(f"Unique Earnings Call names: {len(ec_names)}")
    
    # Free memory
    del dir_df, ec_df
    
    # Create matcher and perform matching
    print("\n=== Start string matching ===")
    matcher = StringMatcher(num_perm=128, lsh_threshold=0.6, qgram_size=3)
    
    fuzzy_df = matcher.match(
        boardex_names, 
        ec_names,
        name_A_col='boardex_name',
        name_B_col='ec_name'
    )
    
    # Save results
    output_file = 'fuzzy_df.pqt'
    fuzzy_df.to_parquet(output_file)
    print(f"\n=== Matching complete ===")
    print(f"Total matched pairs: {len(fuzzy_df)}")
    print(f"Results saved to: {output_file}")
    
    return fuzzy_df


if __name__ == '__main__':
    fuzzy_df = main()

