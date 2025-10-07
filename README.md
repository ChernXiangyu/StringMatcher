# StringMatcher

A high-performance Python toolkit for large-scale fuzzy string matching using MinHash LSH (Locality-Sensitive Hashing) and multiple similarity metrics.

## Overview

StringMatcher is designed to efficiently match similar strings across large datasets by combining:
- **MinHash LSH** for fast candidate pair generation with O(n) time complexity
- **RapidFuzz** for computing multiple fuzzy matching scores on candidate pairs
- **Q-gram tokenization** for robust string comparison

This two-stage approach dramatically reduces computation time compared to naive pairwise comparison, making it ideal for matching names, addresses, or any text data across millions of records.

## Features

- ⚡ **Scalable**: Handles millions of string comparisons efficiently using LSH indexing
- 📊 **Multiple Metrics**: Computes 6 different similarity scores (ratio, partial ratio, token sort/set ratios)
- 🎯 **Configurable**: Adjustable LSH threshold, number of permutations, and Q-gram size
- 🔄 **Two-Stage Pipeline**: Fast candidate generation followed by precise similarity scoring
- 📈 **Progress Tracking**: Built-in progress bars with tqdm
- 🧹 **Text Normalization**: Automatic case normalization and special character handling

## Installation

### Prerequisites

```bash
pip install pandas dask[dataframe] datasketch rapidfuzz tqdm
```

### Dependencies

- `pandas`: Data manipulation and analysis
- `dask`: Large-scale data processing
- `datasketch`: MinHash and LSH implementation
- `rapidfuzz`: Fast fuzzy string matching
- `tqdm`: Progress bars

## Quick Start

### Basic Usage

```python
from string_matcher import StringMatcher

# Initialize the matcher
matcher = StringMatcher(
    num_perm=128,           # Number of hash functions (higher = more accurate)
    lsh_threshold=0.6,      # LSH similarity threshold (0.0-1.0)
    qgram_size=3            # Q-gram size for tokenization
)

# Two arrays of strings to match
array_A = ["John Smith", "Jane Doe", "Robert Johnson"]
array_B = ["Jon Smith", "Jane D.", "Bob Johnson", "Alice Williams"]

# Perform matching
results = matcher.match(
    array_A, 
    array_B,
    name_A_col='name_A',
    name_B_col='name_B'
)

# View results sorted by similarity
print(results.head())
```

### Output Format

The `match()` method returns a pandas DataFrame with the following columns:

| Column | Description |
|--------|-------------|
| `name_A` | String from array A |
| `name_B` | Matched string from array B |
| `partial_ratio` | Partial string matching score (0-100) |
| `ratio` | Basic Levenshtein distance ratio (0-100) |
| `token_sort_ratio` | Token-sorted comparison (0-100) |
| `token_set_ratio` | Token-set comparison (0-100) |
| `partial_token_sort_ratio` | Partial token-sorted score (0-100) |
| `partial_token_set_ratio` | Partial token-set score (0-100) |

## How It Works

### Stage 1: Candidate Generation with MinHash LSH

1. Convert strings to Q-grams (character n-grams)
2. Generate MinHash signatures for each string
3. Build LSH index with array B
4. Query the index with array A to find candidate pairs

**Time Complexity**: O(n + m) instead of O(n × m) for naive comparison

### Stage 2: Fuzzy Matching

1. Compute multiple similarity metrics for each candidate pair
2. Return sorted results with all similarity scores

## API Reference

### `StringMatcher`

Main class for string matching operations.

#### `__init__(num_perm=128, lsh_threshold=0.6, qgram_size=3)`

Initialize the string matcher.

**Parameters:**
- `num_perm` (int): Number of MinHash hash functions. Higher values increase accuracy but reduce speed. Default: 128
- `lsh_threshold` (float): LSH similarity threshold (0.0-1.0). Higher values increase precision but may reduce recall. Default: 0.6
- `qgram_size` (int): Q-gram length for tokenization. Default: 3

#### `match(arr_A, arr_B, name_A_col='name_A', name_B_col='name_B')`

Execute the full string matching pipeline.

**Parameters:**
- `arr_A` (List[str]): First string array
- `arr_B` (List[str]): Second string array
- `name_A_col` (str): Column name for array A in output DataFrame
- `name_B_col` (str): Column name for array B in output DataFrame

**Returns:**
- `pd.DataFrame`: DataFrame with matched pairs and similarity scores

#### `find_candidate_pairs(arr_A, arr_B)`

Find candidate similar pairs using MinHash LSH.

**Parameters:**
- `arr_A` (List[str]): First string array
- `arr_B` (List[str]): Second string array

**Returns:**
- `Set[Tuple[str, str]]`: Set of candidate pairs

#### `compute_fuzzy_scores(candidates, name_A_col='name_A', name_B_col='name_B')`

Compute fuzzy matching scores for candidate pairs.

**Parameters:**
- `candidates` (Set[Tuple[str, str]]): Set of candidate pairs
- `name_A_col` (str): Column name for array A
- `name_B_col` (str): Column name for array B

**Returns:**
- `pd.DataFrame`: DataFrame with similarity metrics

## Configuration Guide

### Choosing `num_perm`

- **32-64**: Fast but less accurate, suitable for very large datasets
- **128** (default): Good balance of speed and accuracy
- **256+**: High accuracy for critical applications

### Choosing `lsh_threshold`

- **0.3-0.5**: High recall, catches more potential matches (more candidates)
- **0.6** (default): Balanced precision and recall
- **0.7-0.9**: High precision, fewer false positives (fewer candidates)

### Choosing `qgram_size`

- **2**: Better for very short strings or when high sensitivity is needed
- **3** (default): Standard choice, works well for most text
- **4-5**: Better for longer strings with less noise

## Performance Tips

1. **Adjust LSH threshold**: Lower threshold = more candidates = higher recall but slower
2. **Filter results**: Use a score threshold on the output DataFrame (e.g., `partial_ratio >= 85`)
3. **Deduplicate input**: Remove duplicates from input arrays to reduce comparisons
4. **Batch processing**: For extremely large datasets, process in chunks and concatenate results

## Example: Filtering Results

```python
# Get only high-confidence matches
results = matcher.match(array_A, array_B)
high_confidence = results[results['partial_ratio'] >= 85]

# Get best match for each string in array A
best_matches = results.sort_values('partial_ratio', ascending=False).groupby('name_A').first()
```

## Use Cases

- **Entity Resolution**: Match company names, person names, or addresses across datasets
- **Data Deduplication**: Find and merge duplicate records
- **Record Linkage**: Connect records from different data sources
- **Name Matching**: Match variations of names (nicknames, misspellings, different formats)
- **Data Integration**: Align entities across multiple databases

## Example Output

```
   name_A           name_B              partial_ratio  ratio  token_sort_ratio  ...
0  John Smith       Jon Smith           95.0           90.0   90.0              ...
1  Robert Johnson   Bob Johnson         85.0           75.0   75.0              ...
2  Jane Doe         Jane D.             88.0           70.0   70.0              ...
```

## License

This project is available for use under your chosen license.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{stringmatcher,
  author = {Xiangyu Chen},
  title = {StringMatcher: Large-Scale Fuzzy String Matching with MinHash LSH},
  year = {2025},
  url = {https://github.com/yourusername/StringMatcher}
}
```

## Contact

For questions or feedback, please open an issue on GitHub.
