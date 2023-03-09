# GitModel
GitModel is for dynamically generating high quality heirarchical topic tree representations of github repos using customizable GNN layers. 

- Highly customizable philospophy. Goal to support huggingface, openai, cohere, etc. python, js, c, c++, C#, etc
- BERTopic is highly customizable and can compose several different clustering and dimensionality reduction techniques.


"Memory" Tree representations can be dynamically selected and added to the system prompt augmenting text generation. 




## Examples
### DeepMind CLRS 
- https://github.com/deepmind/clrs

```
.
├─Purpose and Attributes of the `Net` Class in Graph Neural Networks____
│    ├─Graph Attention Networks and DAG Shortest Paths in JAX.____
│    │    ├─Graph Attention Networks (GAT and GATv2) code implementation____
│    │    │    ├─Code for DAG shortest path and depth-first search algorithms____
│    │    │    │    ├─■──String Matching and Maximum Subarray____ ── Topic: 10
│    │    │    │    └─Depth-First Search and DAG Shortest Path Algorithms implemented in Python____
│    │    │    │         ├─■──Description of string probe functions in probing.py file for Hash Table probing.____ ── Topic: 1
│    │    │    │         └─Graph Algorithms - DFS and DAG Shortest Paths____
│    │    │    │              ├─■──Graph algorithms (DFS and DAG shortest path) in Python____ ── Topic: 0
│    │    │    │              └─■──Functions for decoding diff and graph features in PyTorch graph neural networks.____ ── Topic: 6
│    │    │    └─■──Graph Attention Networks (GAT and GATv2)____ ── Topic: 20
│    │    └─■──Message Passing with _MessagePassingScanState, _MessagePassingOutputChunked and MessagePassingStateC ── Topic: 17
│    └─Implementing a Baseline Model with Selectable Message Passing Algorithm and its Dataset Sampler.____
│         ├─Handling of untrained parameters in optimization updates____
│         │    ├─■──Updating parameters with filtered gradients from multiple algorithms.____ ── Topic: 8
│         │    └─■──Processing trajectory hints with variable-length time dimension using batching.____ ── Topic: 9
│         └─Processing time-chunked data with batched samplers and message passing nets.____
│              ├─Model processing of time-chunked data with dataset sampling and batch processing____
│              │    ├─■──CLRS dataset download and URL retrieval on Google Cloud Platform____ ── Topic: 13
│              │    └─Chunked data and dataset sampling with JAX.____
│              │         ├─■──JAX functions for reshaping and restacking data for pmap computation____ ── Topic: 4
│              │         └─Data chunking with batched sampling and message passing in neural networks.____
│              │              ├─Processing time-chunked data with batch samplers and a NetChunked class____
│              │              │    ├─■──Time-chunked data processing using BaselineModelChunked and NetChunked in TensorFlow.____ ── Topic: 2
│              │              │    └─■──Creating samplers for training data.____ ── Topic: 11
│              │              └─■──Documented code for sampling algorithms using randomized position generation.____ ── Topic: 3
│              └─■──Point Sampling and Convex Hull Computation____ ── Topic: 18
└─Loss functions for training with time-chunked data____
     ├─Loss calculation for time-chunked and full-sample training.____
     │    ├─Code functions for evaluating predictions using permutations and masking____
     │    │    ├─■──Functions for Evaluating Predictions in Probing Tasks.____ ── Topic: 7
     │    │    └─■──permutation pointer manipulation and reduction in predictions____ ── Topic: 16
     │    └─Loss calculation and decoder output postprocessing in neural networks.____
     │         ├─■──Postprocessing with Sinkhorn operator in log space____ ── Topic: 15
     │         └─■──Loss calculation methods for training with time and full samples____ ── Topic: 19
     └─Functions for expanding and broadcasting JAX arrays____
          ├─Description and input/output parameters of _expand_to and _is_not_done_broadcast functions____
          │    ├─■──Array expansion and broadcasting techniques____ ── Topic: 21
          │    └─■──Purpose and Functionality of _is_not_done_broadcast Function____ ── Topic: 14
          └─Sampler classes and associated data generation types____
               ├─■──Understanding Parameters and Expected Input/Output of Various Functions (including mst_prim, floyd_w ── Topic: 5
               └─■──Sampling classes and their data generation purpose____ ── Topic: 12
```
With topic tree in system prompt 
```
.
├─DFS and DAG Shortest Paths Algorithm Implementation with Probing____
│    ├─■──Bipartite matching-based flow networks____ ── Topic: 34
│    └─Search and Shortest Path Algorithms____
│         ├─DAG shortest path algorithm with probing and initialization____
│         │    ├─■──Strongly Connected Components Algorithm with Kosaraju's Implementation____ ── Topic: 37
│         │    └─Graph Sampling and DAG Shortest Path Algorithm____
│         │         ├─■──Bipartite matching using Edmonds-Karp algorithm____ ── Topic: 18
│         │         └─■──Random graph generation using Bellman-Ford algorithm in Python____ ── Topic: 0
│         └─Graham scan convex hull algorithm implementation in Python____
│              ├─■──Maximum subarray algorithm implementation____ ── Topic: 6
│              └─■──Graham scan convex hull algorithm implementation____ ── Topic: 12
└─Postprocessing Decoder Output for Chunked Data Processing Net____
     ├─Postprocessing Decoder Output with Chunked Data in JAX____
     │    ├─Functions and Files in Probing.py Explained____
     │    │    ├─Functions and techniques for data splitting and replication in probing and pmap computation.____
     │    │    │    ├─Understanding the strings_pair_cat function and split_stages function in probing.py file____
     │    │    │    │    ├─TFDS CLRSDataset Command-Line Tool for Sampling Datasets____
     │    │    │    │    │    ├─■──CLRS30 dataset and related functions explanation____ ── Topic: 5
     │    │    │    │    │    └─■──TFDS CLRSDataset Builder Implementation____ ── Topic: 16
     │    │    │    │    └─Functions and Probing in Python Code____
     │    │    │    │         ├─Purpose of the `split_stages` function in `probing.py` and related functions for evaluating `ProbesD
     │    │    │    │         │    ├─Functions for evaluating hint and output predictions using permutation objects and dictionaries.____
     │    │    │    │         │    │    ├─Processing randomized `pos` input in a sampler with pointers and permutations.____
     │    │    │    │         │    │    │    ├─■──Process randomization of `pos` input in algorithms including string algorithms____ ── Topic: 29
     │    │    │    │         │    │    │    └─■──A function to replace should-be permutations with proper permutation pointers using a sample iterato ── Topic: 19
     │    │    │    │         │    │    └─Function for Evaluating Permutation Predictions using Hint Data____
     │    │    │    │         │    │         ├─■──Function to Reduce Permutations in a Dictionary of Result Objects____ ── Topic: 11
     │    │    │    │         │    │         └─■──Function to evaluate hint predictions with tuple and list inputs____ ── Topic: 17
     │    │    │    │         │    └─Understanding probing functions in Hash Table implementation____
     │    │    │    │         │         ├─Hash Table Probing Functions in probing.py File____
     │    │    │    │         │         │    ├─■──Splitting ProbesDict into DataPoints by stage in Python____ ── Topic: 14
     │    │    │    │         │         │    └─■──Understanding Hash Table Probing Functions (strings_pi, strings_pos, strings_pair_cat) in Python's ` ── Topic: 1
     │    │    │    │         │         └─■──Functions for Checking Input Dimensions in Machine Learning Models____ ── Topic: 15
     │    │    │    │         └─JAX pmap reshaping and computation functions (_pmap_reshape, _maybe_pmap_reshape, _maybe_pmap_data)_
     │    │    │    │              ├─JAX pmap computation and pytree reshaping____
     │    │    │    │              │    ├─■──Purpose and attributes of the Stage and OutputClass classes____ ── Topic: 22
     │    │    │    │              │    └─■──JAX tree reshaping for pmap computation with _pmap_reshape and _maybe_pmap_reshape functions____ ── Topic: 3
     │    │    │    │              └─Numpy array copying functions with assertions____
     │    │    │    │                   ├─■──Functions for copying data between numpy arrays in Python____ ── Topic: 21
     │    │    │    │                   └─■──Function Purpose and Parameters Analysis in Codebase____ ── Topic: 9
     │    │    │    └─Trajectory Batching with Variable-Length Time Dimension____
     │    │    │         ├─■──Trajectory Batching and Concatenation____ ── Topic: 35
     │    │    │         └─■──Batch processing of variable-length hint trajectories.____ ── Topic: 31
     │    │    └─Understanding the `_is_not_done_broadcast` function and its input/output parameters.____
     │    │         ├─■──Understanding the _is_not_done_broadcast function in JAX array for sequence completion.____ ── Topic: 8
     │    │         └─■──Array broadcasting and expansion with _expand_and_broadcast_to and _expand_to functions____ ── Topic: 27
     │    └─Postprocessing Decoder Output with Sinkhorn Algorithm and Hard Categorization____
     │         ├─Node Feature Decoding with Encoders and Decoders____
     │         │    ├─■──Position Encoding Function for Natural Language Processing____ ── Topic: 23
     │         │    └─Node feature decoding using decoders and edge features____
     │         │         ├─■──Creating Encoders with Xavier Initialization and Truncated Normal Distribution for Encoding Categori ── Topic: 33
     │         │         └─Node feature decoding with decoders and edge features____
     │         │              ├─■──Node feature decoding and encoding with decoders and edge features____ ── Topic: 2
     │         │              └─■──Graph diff decoders____ ── Topic: 32
     │         └─Postprocessing of decoder output in graph neural networks.____
     │              ├─Decoder Output Postprocessing with Sinkhorn Algorithm and Cross-Entropy Loss____
     │              │    ├─Message Passing Net with Time-Chunked Data Processing____
     │              │    │    ├─■──Python Class for Message Passing Model with Selectable Algorithm____ ── Topic: 26
     │              │    │    └─■──NetChunked message passing operation with LSTM states for time-chunked data____ ── Topic: 7
     │              │    └─Loss calculation for time-chunked training with scalar truth data.____
     │              │         ├─Loss calculation function for time-chunked training with scalar truth data.____
     │              │         │    ├─■──Loss calculation for time-chunked training data____ ── Topic: 4
     │              │         │    └─■──Logarithmic Sinkhorn Operator for Permutation Pointer Logits____ ── Topic: 10
     │              │         └─■──Decoder postprocessing with Sinkhorn operator____ ── Topic: 28
     │              └─Gradient Filtering for Optimizer Updates____
     │                   ├─■──Filtering processor parameters in Haiku models____ ── Topic: 30
     │                   └─■──Filtering null gradients for untrained parameters during optimization.____ ── Topic: 24
     └─PGN with Jax implementation and NeurIPS 2020 paper____
          ├─Message-Passing Neural Network (MPNN) for Graph Convolutional Networks (GCNs)____
          │    ├─■──"Applying Triplet Message Passing with HK Transforms in MPNN for Graph Neural Networks"____ ── Topic: 20
          │    └─■──Implementation of Deep Sets (Zaheer et al., NeurIPS 2017) using adjacency matrices and memory networ ── Topic: 13
          └─GATv2 Graph Attention Network with adjustable sizes of multi-head attention and residual connections
               ├─■──Graph Attention Network v2 architecture with adjustable head number and output size.____ ── Topic: 36
               └─■──Processor factory with various models and configurations____ ── Topic: 25
```
