# GitModel 🥳
GitModel is for dynamically generating high quality heirarchical topic tree representations of github repos using customizable GNN layers. 

- Highly customizable philospophy. Goal to support huggingface, openai, cohere, etc. python, js, c, c++, C#, etc
- BERTopic is highly customizable and can compose several different clustering and dimensionality reduction techniques.


"Memory" Tree representations can be dynamically selected and added to the system prompt augmenting text generation. 




## Examples
## Gitmodel
```
.
├─Outlier Reduction with Topic Modeling Libraries____
│    ├─Document reduction with hierarchical clustering and TF-IDF feature extraction____
│    │    ├─■──Hierarchical Topic Modeling with Document Embeddings____ ── Topic: 2
│    │    └─■──Document processing with NLP techniques____ ── Topic: 10
│    └─Reducing Outliers in BERTopic Generated Topics using a Memory Tree Hierarchical Approach____
│         ├─■──BERTopic model reduces outliers in generated topics____ ── Topic: 5
│         └─■──Memory tree generation with hierarchical topic modeling using BERTopic in GitModel____ ── Topic: 9
└─Embedding text data using MiniLM language model.____
     ├─Text embedding with MiniLM language model____
     │    ├─■──MiniLM language model for sentence embedding____ ── Topic: 12
     │    └─Computing Gaussian kernel matrix between two sets of embeddings using PyTorch____
     │         ├─Computing Gaussian kernel matrix between two sets of embeddings using PyTorch____
     │         │    ├─■──Computing Gaussian kernel matrix using PyTorch on two sets of embeddings____ ── Topic: 1
     │         │    └─■──Computing graph Laplacian and degree matrix from pairwise distances using numpy____ ── Topic: 4
     │         └─■──Probability density estimation using Gaussian KDE____ ── Topic: 13
     └─Extracting code summary prompts from GitHub repositories using ChatGPT engine____
          ├─ChatGPT-based Code Summarization for GitHub Repositories____
          │    ├─■──OpenAI GPT-3.5-Turbo model for chat response inference with `chat_gpt_inference` function.____ ── Topic: 11
          │    └─■──Generating System Prompts for ChatGPT____ ── Topic: 3
          └─Code decomposition and system prompts formatting in a Git repository____
               ├─Code decomposition and system prompts formatting in Python____
               │    ├─■──Function purpose and step-by-step guide for repo graph generation in Python file.____ ── Topic: 6
               │    └─Issues with a function for decomposing code in a git repository and formatting system prompts.____
               │         ├─■──Understanding the MemoryTreeManager and MessageTreeManagerConfiguration classes in the GitModel repo ── Topic: 8
               │         └─■──Issues with a function for decomposing a GitHub repository into topics____ ── Topic: 0
               └─■──Running SQLAlchemy Migrations in Online Mode with Alembic____ ── Topic: 7
```

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

## Langchain
- https://github.com/hwchase17/langchain
```
.
├─Combining documents with different chain types and LLM chains____
│    ├─MapReduce Chain Loading and Combining____
│    │    ├─Question answering chain with sources loading and combining____
│    │    │    ├─■──Loading question answering with sources chain with multiple loader mappings and chains.____ ── Topic: 53
│    │    │    └─■──Loading and Combining Documents with Language Models for Summarizing and QA____ ── Topic: 71
│    │    └─Map Reduce Chain Loading Function____
│    │         ├─Document Refinement using LLM Chains____
│    │         │    ├─■──Combining Documents with Stuffing and LLM Chain in Python____ ── Topic: 97
│    │         │    └─BaseQAWithSourcesChain document handling and processing.____
│    │         │         ├─■──Question Answering with Sources over Documents Chain____ ── Topic: 60
│    │         │         └─■──Python class for chatbot with vector database and question generation____ ── Topic: 16
│    │         └─MapReduce chain implementation____
│    │              ├─■──MapReduceDocumentsChain document combination with chaining and mapping____ ── Topic: 12
│    │              └─■──MapReduce Chain Loading Function____ ── Topic: 95
│    └─LLMBashChain document examples and related keywords____
│         ├─Bash operations and language modeling chain implementation____
│         │    ├─LLMSummarizationCheckerChain document samples____
│         │    │    ├─■──Working with SQL databases in Python using SQLDatabaseChain____ ── Topic: 46
│         │    │    └─Document processing with LLMSummarizationCheckerChain____
│         │    │         ├─■──Implementation of Program-Aided Language Models with PALChain class and related prompts and assertio ── Topic: 31
│         │    │         └─■──LLMSummarizationCheckerChain class and its functionality____ ── Topic: 93
│         │    └─LLMBashChain - interpreting prompts and executing bash code____
│         │         ├─■──LLMMathChain - Python code execution for math prompts____ ── Topic: 92
│         │         └─■──Bash execution with LLMBashChain____ ── Topic: 80
│         └─■──MRKLChain implementation with ChainConfig and API integration____ ── Topic: 59
└─Code organization and structure in Python including several classes related to self-hosted embedding
     ├─Code organization and improvement suggestions for a class definition.____
     │    ├─Code Loading and Organization Best Practices____
     │    │    ├─Web scraping Hacker News webpage titles____
     │    │    │    ├─Loading files using unstructured in Python____
     │    │    │    │    ├─Unstructured file loading with retry and partitioning capabilities.____
     │    │    │    │    │    ├─■──Retry Decorator for OpenAI API Calls____ ── Topic: 45
     │    │    │    │    │    └─Unstructured File Loading and Partitioning____
     │    │    │    │    │         ├─■──Unstructured File Loader for Partitioning Files in Various Formats____ ── Topic: 25
     │    │    │    │    │         └─■──Loading files with Unstructured package in different modes (Python code).____ ── Topic: 26
     │    │    │    │    └─PDF manipulation in Python with pypdf, pdfminer, fitz and pymupdf libraries____
     │    │    │    │         ├─■──PDF file loading and text extraction using PyMuPDF and PDFMiner____ ── Topic: 69
     │    │    │    │         └─■──Extracting Text from Paged PDF using PyPDF and PDFMiner____ ── Topic: 96
     │    │    │    └─Extracting Hacker News Webpage Information using WebBaseLoader and BeautifulSoup.____
     │    │    │         ├─■──Web scraping Hacker News with BeautifulSoup and WebBaseLoader____ ── Topic: 21
     │    │    │         └─■──Web Scraping for College Confidential and Lyrics Websites____ ── Topic: 76
     │    │    └─Code organization and structure in various Python modules____
     │    │         ├─Compliments on clear and structured codebase with good use of type hints for memory handling and con
     │    │         │    ├─Implementation of ReAct paper using ReActChain with examples in Python____
     │    │         │    │    ├─■──Implementation of ReAct paper in ReActChain agent with OpenAI LLC model and tools____ ── Topic: 101
     │    │         │    │    └─In-memory Docstore for Efficient Lookup and Exploration____
     │    │         │    │         ├─■──Document Store Exploration with DocstoreExplorer____ ── Topic: 87
     │    │         │    │         └─■──InMemoryDocstore for Storing and Searching Documents with AddableMixin____ ── Topic: 61
     │    │         │    └─Compliments on Code Readability and Organization in Python Codebase.____
     │    │         │         ├─Memory Handling and Conversation Management____
     │    │         │         │    ├─Memory Conversation Summarizer Implementation____
     │    │         │         │    │    ├─Memory and Conversation Summarization in AI-assisted dialogues.____
     │    │         │         │    │    │    ├─■──Purpose of ChatPromptValue class in chat.py____ ── Topic: 30
     │    │         │         │    │    │    └─■──Memory management and conversation summarization in AI chatbot system.____ ── Topic: 6
     │    │         │         │    │    └─■──Implementation of Chain class with CallbackManager and Memory attributes.____ ── Topic: 52
     │    │         │         │    └─Potential bugs and suggestions for loading LLM, few-shot prompts, and examples from JSON and YAML fi
     │    │         │         │         ├─Code structure and organization tips for loading examples and templates from files in Python.____
     │    │         │         │         │    ├─Compliments on code structure and organization____
     │    │         │         │         │    │    ├─■──Loading few-shot prompts from config with prefix and suffix templates____ ── Topic: 34
     │    │         │         │         │    │    └─Code organization and structure for creating chat prompt templates____
     │    │         │         │         │    │         ├─■──Chat prompt template and message prompt templates for generating chatbot prompts.____ ── Topic: 8
     │    │         │         │         │    │         └─■──Purpose of `_load_prompt_from_file` function in loading.py module.____ ── Topic: 13
     │    │         │         │         │    └─■──Function for Loading a Chain of LLM Checkers from a Configuration Dictionary.____ ── Topic: 3
     │    │         │         │         └─Documented class definitions for tools used in handling API requests, including OpenSearchVectorSear
     │    │         │         │              ├─Handling API requests using tools such as RequestsPostTool and OpenSearchVectorSearch____
     │    │         │         │              │    ├─Python requests wrapper for making HTTP requests with various tools and methods____
     │    │         │         │              │    │    ├─■──DeepInfra API token and text generation model wrapper____ ── Topic: 41
     │    │         │         │              │    │    └─RequestsWrapper and BaseRequestsTool for making HTTP requests (POST, GET, PATCH, DELETE) to API endp
     │    │         │         │              │    │         ├─■──Checking Validity of Template Strings with Input Variables and Formatter Mapping____ ── Topic: 14
     │    │         │         │              │    │         └─■──Requests tools for making HTTP requests with Python____ ── Topic: 10
     │    │         │         │              │    └─Code organization and positive feedback____
     │    │         │         │              │         ├─Bing Search API Wrapper and Handler Classes____
     │    │         │         │              │         │    ├─■──Langchain callback manager and codebase organization____ ── Topic: 2
     │    │         │         │              │         │    └─■──Bing Search API Wrapper and SERP API Usage in Python____ ── Topic: 1
     │    │         │         │              │         └─Handling iFixit devices with models and remote hardware____
     │    │         │         │              │              ├─■──Loading iFixit repair guides and device wikis with transformer model inference.____ ── Topic: 0
     │    │         │         │              │              └─■──Potential Issues with Modifying Input Dictionary in a Prompt Loading Function____ ── Topic: 9
     │    │         │         │              └─Implementation and Usage of SearxSearchWrapper with Environment Variables and SSL Support____
     │    │         │         │                   ├─Python Libraries for API Wrappers and Search Engines____
     │    │         │         │                   │    ├─Python packages for integrating with search engines: SearxSearchWrapper and QdrantClient.____
     │    │         │         │                   │    │    ├─■──Implementation of Searx API Wrapper (SearxSearchWrapper) using Python's BaseModel with QdrantClient  ── Topic: 33
     │    │         │         │                   │    │    └─■──Handling environment variables and dictionaries with get_from_dict_or_env function____ ── Topic: 72
     │    │         │         │                   │    └─Purpose and Issues with `print_text` Function in `langchain` Repository's `input.py` File____
     │    │         │         │                   │         ├─■──Printing Highlighted Text with Options in Python____ ── Topic: 51
     │    │         │         │                   │         └─■──Converting Python Objects to String Representation with Nested Structures and Joining on Newline Cha ── Topic: 66
     │    │         │         │                   └─GitbookLoader class and its methods____
     │    │         │         │                        ├─■──Handling newlines recursively in data structures using pandas____ ── Topic: 29
     │    │         │         │                        └─GitBookLoader class for loading web pages with options to load all or single pages____
     │    │         │         │                             ├─■──GitbookLoader class for loading single or multiple pages from GitBook with relative paths in the nav ── Topic: 28
     │    │         │         │                             └─■──Length-Based Example Selection and Text Length Calculation____ ── Topic: 57
     │    │         │         └─Ngram overlap score using sentence_bleu and method1 smoothing function____
     │    │         │              ├─Ngram overlap score using sentence_bleu method1 smoothing function and auto reweighting____
     │    │         │              │    ├─■──Code structure and organization in langchain document loaders with support for parsing comma-separat ── Topic: 70
     │    │         │              │    └─Ngram overlap score using sentence_bleu and method1 smoothing function with auto reweighting in nltk
     │    │         │              │         ├─■──Compliments on well-structured and organized code in different classes and methods____ ── Topic: 65
     │    │         │              │         └─■──Sentence BLEU score and ngram overlap computation with method1 smoothing function and auto reweighti ── Topic: 49
     │    │         │              └─Model Definition and Experimentation with Datetime and UTCNow Attributes____
     │    │         │                   ├─■──Data Modeling with Time Zones in Python____ ── Topic: 91
     │    │         │                   └─■──Constitutional Principles and Tracing in Python____ ── Topic: 68
     │    │         └─Text splitting for knowledge triple extraction____
     │    │              ├─Text Splitting Toolkit____
     │    │              │    ├─Text splitting interface and implementation____
     │    │              │    │    ├─Python REPL Tool and AST Implementation____
     │    │              │    │    │    ├─Python REPL Tool Implementation____
     │    │              │    │    │    │    ├─SQL database metadata retrieval tool____
     │    │              │    │    │    │    │    ├─■──Python function to concatenate cell information for AI and human usage____ ── Topic: 44
     │    │              │    │    │    │    │    └─SQL database metadata tool for listing table schema and metadata____
     │    │              │    │    │    │    │         ├─■──SQL database metadata extraction tool for specified tables____ ── Topic: 75
     │    │              │    │    │    │    │         └─■──JSON and SQL database tools for listing and getting values____ ── Topic: 15
     │    │              │    │    │    │    └─Python REPL Tool using AST and Coroutine____
     │    │              │    │    │    │         ├─■──Tool implementation with direct function or coroutine input and error handling.____ ── Topic: 99
     │    │              │    │    │    │         └─■──Python REPL Tool with AST and version validation____ ── Topic: 74
     │    │              │    │    │    └─Implementing API wrappers for news, movie information, and weather using APIChain____
     │    │              │    │    │         ├─Implementing APIs for News, Weather, and Movie Information in LangChain's Load Tools Module____
     │    │              │    │    │         │    ├─■──Language model for reasoning about position and color attributes of objects in weather forecasting w ── Topic: 73
     │    │              │    │    │         │    └─Implementing APIs for fetching news and movies using Python____
     │    │              │    │    │         │         ├─■──well-structured and readable implementation of API initialization functions in load_tools.py for Too ── Topic: 85
     │    │              │    │    │         │         └─■──Working with API authentication and chaining for news and movie information retrieval (using news_ap ── Topic: 100
     │    │              │    │    │         └─■──Wolfram Alpha SDK querying using WolframAlphaQueryRun class and api_wrapper attribute____ ── Topic: 89
     │    │              │    │    └─TextSplitting for Vector Storage with Overlapping Chunks____
     │    │              │    │         ├─Python's StrictFormatter class and its check_unused_args method for formatting and validation of inp
     │    │              │    │         │    ├─L2 distance search using ndarray in Python____
     │    │              │    │         │    │    ├─■──L2 search for nearest neighbors with np.linalg.norm____ ── Topic: 32
     │    │              │    │         │    │    └─■──Parsing and Organizing Notes with Hashing and Embeddings____ ── Topic: 67
     │    │              │    │         │    └─Python Class for Strict Formatter with Check on Unused Args____
     │    │              │    │         │         ├─Vector Store Toolkit and Deployment____
     │    │              │    │         │         │    ├─■──Vector Store Toolkit and Deployment with OpenAI LLM____ ── Topic: 35
     │    │              │    │         │         │    └─■──Working with AirbyteJSONLoader to load local Airbyte JSON files____ ── Topic: 47
     │    │              │    │         │         └─Python Formatter class with check_unused_args method and strict validation____
     │    │              │    │         │              ├─Python's StrictFormatter class and its check_unused_args method for validating unused and extra argu
     │    │              │    │         │              │    ├─■──Finding TODO Tasks in Code Snippets____ ── Topic: 4
     │    │              │    │         │              │    └─Python Formatter and StrictFormatter with check_unused_args method____
     │    │              │    │         │              │         ├─■──Color Mapping Function for Prompt Inputs with Exclusions____ ── Topic: 88
     │    │              │    │         │              │         └─■──Implementing strict checking of unused and extra keys in a subclass of formatter____ ── Topic: 48
     │    │              │    │         │              └─Python module for loading and manipulating language chain data with verbosity control.____
     │    │              │    │         │                   ├─■──Python function for getting verbosity from language chaining with Azure OpenAI and difference from O ── Topic: 64
     │    │              │    │         │                   └─■──Purpose of functions in loading.py and csv toolkit of langchain repository____ ── Topic: 42
     │    │              │    │         └─Text splitting using chunk size and overlap with various libraries and interfaces.____
     │    │              │    │              ├─Text splitting and chunking with overlap and length functions____
     │    │              │    │              │    ├─■──Developing and Maintaining Docker Compose Modules in Python____ ── Topic: 79
     │    │              │    │              │    └─Text splitting and chunking using TextSplitter interface____
     │    │              │    │              │         ├─Text Splitting Interface and Implementation____
     │    │              │    │              │         │    ├─■──Text splitting using TokenTextSplitter class.____ ── Topic: 7
     │    │              │    │              │         │    └─■──Document Loading and Splitting with Text Splitting and Callback Management.____ ── Topic: 84
     │    │              │    │              │         └─■──Python code for initializing an agent with various optional arguments____ ── Topic: 18
     │    │              │    │              └─Loading Google Docs from Google Drive using Credentials and Tokens with Python____
     │    │              │    │                   ├─Document Loading from Cloud Storage (GCS and S3) using BaseLoader Class____
     │    │              │    │                   │    ├─■──Online PDF loading and caching using SQLite and temporary directories____ ── Topic: 98
     │    │              │    │                   │    └─■──Loading documents from cloud storage using GCSFileLoader and S3FileLoader classes.____ ── Topic: 36
     │    │              │    │                   └─■──Google Drive Loader and Credentials for Loading Google Docs____ ── Topic: 86
     │    │              │    └─StreamlitCallbackHandler for logging to streamlit in Python code____
     │    │              │         ├─Streaming with LLMs and Callback Handlers____
     │    │              │         │    ├─Networkx wrapper for entity graph operations with Redis caching.____
     │    │              │         │    │    ├─NetworkX Entity Graph with Missing Tables and Callback Manager____
     │    │              │         │    │    │    ├─■──Graph Index Creation and Operations using NetworkX Library in Python____ ── Topic: 58
     │    │              │         │    │    │    └─■──NetworkxEntityGraph and entity graph operations.____ ── Topic: 20
     │    │              │         │    │    └─Redis cache implementation in Python____
     │    │              │         │    │         ├─■──Implementing a SQAlchemy-based cache system with missing and existing prompts for better performance ── Topic: 17
     │    │              │         │    │         └─■──Implementation of a Redis cache as a backend in Python____ ── Topic: 39
     │    │              │         │    └─Python Callback Handler for Streamlit Logging____
     │    │              │         │         ├─■──Callback handlers for printing to standard output.____ ── Topic: 43
     │    │              │         │         └─■──StreamlitCallbackHandler for logging prompts and actions to Streamlit____ ── Topic: 90
     │    │              │         └─ZeroShotAgent class and observation prefix property in Python____
     │    │              │              ├─Creating a JSON agent using a toolkit for zeroshot agent execution with format instructions and inpu
     │    │              │              │    ├─■──Creating Pandas DataFrames using Agent Scratchpad and Python AST REPL Tool.____ ── Topic: 82
     │    │              │              │    └─Creating a JSON agent with toolkit, format instructions, and prefix/suffix____
     │    │              │              │         ├─■──SQL agent creation with SQLDatabaseToolkit, BaseLLM and BaseCallbackManager____ ── Topic: 11
     │    │              │              │         └─■──Creating a JSON agent with OpenAPI toolkit and interacting with it using JSON tools____ ── Topic: 56
     │    │              │              └─Classes for language model-driven decision making and use of "agent_scratchpad" in LLMChain prompts_
     │    │              │                   ├─■──Agent class and entity extraction using "agent_scratchpad" variable____ ── Topic: 38
     │    │              │                   └─■──Code for a text-based game-playing agent using self-ask-with-search approach in TextWorld environmen ── Topic: 102
     │    │              └─Text Mapping for Approximate k-NN Search using nmslib in Python____
     │    │                   ├─Script Scoring with KNN Search____
     │    │                   │    ├─■──Document bulk-ingest function for embeddings in Elasticsearch index____ ── Topic: 23
     │    │                   │    └─■──Script Scoring Search with Cosine Similarity and k-Nearest Neighbors (k-NN) Algorithm____ ── Topic: 19
     │    │                   └─Default text mapping for Approximate k-NN Search in dense vector fields using NMSLIB engine____
     │    │                        ├─■──Default Mapping for Approximate k-NN Search using NMSLIB Engine____ ── Topic: 81
     │    │                        └─■──Elasticsearch indexing and scripting with default mappings and painless scripting____ ── Topic: 94
     │    └─Tracing and Recording Runs with SharedTracer and TracerStack____
     │         ├─Python classes ToolRun and ChainRun in schemas.py file with additional attributes and their purpose.
     │         │    ├─■──Extracting information about ElementInViewPort instances in chainrun toolrun runs.____ ── Topic: 77
     │         │    └─■──Purpose and attributes of the ChainRun class in schemas.py file____ ── Topic: 78
     │         └─Tracing and thread-safe execution with SharedTracer Singleton class____
     │              ├─■──Tracing Execution Order with BaseTracer in a Thread-Safe Manner____ ── Topic: 55
     │              └─■──TracerStack and SharedTracer Implementation in Python____ ── Topic: 63
     └─Python wrapper for OpenAI and Hugging Face language models____
          ├─Self-Hosted Hugging Face Instructor Embedding Models on Remote Hardware____
          │    ├─HuggingFace and Sentence-Transformers Embeddings for Cohere____
          │    │    ├─■──Output parsing using regular expressions and the BaseOutputParser class____ ── Topic: 54
          │    │    └─NLP Embeddings using Hugging Face and Sentence Transformers____
          │    │         ├─■──Neural Embeddings with Hugging Face and Cohere API____ ── Topic: 24
          │    │         └─■──Loading sentence embedding model with sentence_transformers library.____ ── Topic: 27
          │    └─Self-hosted HuggingFace pipeline API for running models on remote hardware____
          │         ├─Self-hosted HuggingFace pipeline for remote GPU hardware inference with autolaunched instances on va
          │         │    ├─■──Self-hosted HuggingFace pipeline for remote hardware with HuggingFace Transformers and AutoTokenizer ── Topic: 40
          │         │    └─■──Self-hosted embeddings for sentence_transformers with remote hardware support.____ ── Topic: 22
          │         └─■──Self-hosted embeddings for running custom embedding models on remote hardware____ ── Topic: 62
          └─Python wrapper for OpenAI language model with API key authentication and model parameters configurat
               ├─OpenAI Language Model Wrapper Class with API Key Authentication and Model Parameters Configuration__
               │    ├─■──StochasticAI Wrapper for Large Language Models with Environment Key Validation and PDF Partitioning_ ── Topic: 50
               │    └─Integration of OpenAI Language Model with GooseAI class for Text Generation____
               │         ├─■──OpenAI Chat Model Implementation____ ── Topic: 37
               │         └─■──Python Wrapper for OpenAI Language Models____ ── Topic: 5
               └─■──Anthropic Large Language Models and API Usage in AtlasDB Project Management____ ── Topic: 83

```

## Pyknotid 
- https://pyknotid.readthedocs.io/en/latest/
```

└─Calculation of Alexander polynomial for knots in Python and Mathematica.____
     ├─Alexander polynomial calculation using Mathematica process and knot routing algorithm with various p
     │    ├─Calculation of Alexander polynomial for knots using Python and Mathematica representations____
     │    │    ├─CellKnot object initialization and properties with sin, cos, linspace, phi, psi, theta, rotation, pe
     │    │    │    ├─Mollweide projection and spherical coordinates____
     │    │    │    │    ├─Rotation of Spheres using Rotation Matrices____
     │    │    │    │    │    ├─■──Rotation of sphere to align given positions at the top____ ── Topic: 41
     │    │    │    │    │    └─■──Rotation matrix computation and manipulation using iterable angles.____ ── Topic: 18
     │    │    │    │    └─Mollweide projection and conversion of spherical coordinates____
     │    │    │    │         ├─Mollweide projection and spherical coordinates conversion____
     │    │    │    │         │    ├─■──Vector magnitude calculation, Mollweide projection, and well-written code in Python.____ ── Topic: 51
     │    │    │    │         │    └─■──"Mollweide projection and spherical coordinate conversion"____ ── Topic: 30
     │    │    │    │         └─■──Verbose printing function for Pyknotid counters.____ ── Topic: 10
     │    │    │    └─CellKnot class and points folding____
     │    │    │         ├─CellKnot and Knot Folding____
     │    │    │         │    ├─■──Understanding the "cell_trefoil" function and the "aperiodic_trefoil" function for creating interpol ── Topic: 37
     │    │    │         │    └─■──CellKnot class and related methods____ ── Topic: 33
     │    │    │         └─3D geometric scaling with numpy and crossing signs____
     │    │    │              ├─Geometric Transformation with Crossing Signs____
     │    │    │              │    ├─■──Numpy arrays for creating and perturbing a simple link using sin and cos in Python code.____ ── Topic: 15
     │    │    │              │    └─■──Geometric transformation with crossing signs and np array____ ── Topic: 3
     │    │    │              └─■──3D point scaling helper functions in p4_3__1, p4_4__1, p4_5__1_false, p5_3__1 and p5_4__1.____ ── Topic: 47
     │    │    └─Knot representations and calculation of Alexander polynomial using Python and Mathematica____
     │    │         ├─Line Segment Open by Distance Generator____
     │    │         │    ├─Issues with missing function definitions and potential bugs in serialisation functions.____
     │    │         │    │    ├─■──JSON and polynomial serialisation with potential implementation issues____ ── Topic: 17
     │    │         │    │    └─■──Issues with incomplete function for serialising Jones polynomials in Python____ ── Topic: 36
     │    │         │    └─Line vectors open by distance fraction with seed and number of segments as parameters.____
     │    │         │         ├─Line segment manipulation and generation____
     │    │         │         │    ├─Line Segments and Open/Closed Loop Detection____
     │    │         │         │    │    ├─■──Open and closed line segments generation with distance constraint.____ ── Topic: 5
     │    │         │         │    │    └─■──Writing Mathematica code to file and running it using MathKernel____ ── Topic: 28
     │    │         │         │    └─Loading and manipulating CSV files with Pandas and saving to JSON.____
     │    │         │         │         ├─■──Writing and loading data in json format with numpy and handling file paths (filenotfounderror explan ── Topic: 14
     │    │         │         │         └─■──Parsing CSV data using pandas in Python____ ── Topic: 19
     │    │         │         └─Downloading Knots Database with Pyknotid Library.____
     │    │         │              ├─Knots database download and management____
     │    │         │              │    ├─■──Downloading Knots Database using Pyknotid Library____ ── Topic: 23
     │    │         │              │    └─■──Deleting old versions of database files in specific format using Python.____ ── Topic: 44
     │    │         │              └─■──Recursive file inclusion using fnmatch patterns in Python____ ── Topic: 43
     │    │         └─Alexander polynomial computation using Mathematica for knot representations____
     │    │              ├─Calculation of Alexander polynomial using Python and Mathematica code snippets.____
     │    │              │    ├─MeshCollectionVisual class and associated methods for vertex colors and shading in mesh visualizatio
     │    │              │    │    ├─Code Refactoring and Todo Tasks with Comments and Unit Tests____
     │    │              │    │    │    ├─■──Classes and functionality for handling periodic boundary conditions in a 2D space.____ ── Topic: 39
     │    │              │    │    │    └─■──Code Refactoring and Unit Testing____ ── Topic: 4
     │    │              │    │    └─MeshCollectionVisual class and vertex colors in 3D mesh visualization.____
     │    │              │    │         ├─Signal Smoothing with Window Functions____
     │    │              │    │         │    ├─■──Testing vector intersection in a dp/dq region using do_vectors_intersect function and obtaining bool ── Topic: 20
     │    │              │    │         │    └─signal smoothing with different windows and sizes____
     │    │              │    │         │         ├─■──Signal Smoothing using Different Windows____ ── Topic: 49
     │    │              │    │         │         └─■──Code organization and readability of periodic_vassiliev_degree_2 function in adherence with PEP 8 gu ── Topic: 26
     │    │              │    │         └─MeshCollectionVisual class and related methods____
     │    │              │    │              ├─■──MeshCollectionVisual class and its methods for mesh visualization and handling vertex colors and sha ── Topic: 9
     │    │              │    │              └─■──Cell object for lines with periodic boundary conditions____ ── Topic: 45
     │    │              │    └─Alexander polynomial calculation using Mathematica____
     │    │              │         ├─Calculating the Alexander polynomial of knots using various representations____
     │    │              │         │    ├─Gauss code conversion to crossing indices____
     │    │              │         │    │    ├─Recommendations for the "mag" function implementation in periodic.py____
     │    │              │         │    │    │    ├─■──Implementing vector magnitude using dot product in Pyknotid____ ── Topic: 8
     │    │              │         │    │    │    └─■──Improving code with imports and using numpy.zeros instead of n.zeros.____ ── Topic: 50
     │    │              │         │    │    └─■──Converting Gauss code to crossings in a crossing object____ ── Topic: 22
     │    │              │         │    └─Calculation of Alexander polynomial using Mathematica for knot representations____
     │    │              │         │         ├─Knot theory and Alexander polynomial calculation using Mathematica____
     │    │              │         │         │    ├─■──BoundingBox class implementation in Python with numpy and axis manipulation____ ── Topic: 1
     │    │              │         │         │    └─■──Calculation of Alexander polynomial for knot representations using Mathematica____ ── Topic: 0
     │    │              │         │         └─3D sphere plotting with Mollweide projection using VisPy____
     │    │              │         │              ├─■──"3D visualization of spherical data using VisPy and Mollweide projection"____ ── Topic: 2
     │    │              │         │              └─■──Class definition of MeshCollectionVisual that creates a mesh by concatenating visuals' vertices, ind ── Topic: 12
     │    │              │         └─Database objects matching invariants using Python code____
     │    │              │              ├─Database objects and invariants in knot theory____
     │    │              │              │    ├─■──"Database storage and manipulation of knots using Peewee and optimized Cython routines"____ ── Topic: 35
     │    │              │              │    └─■──Database searching with invariants in Python____ ── Topic: 7
     │    │              │              └─■──OpenKnot class for holding vertices of an open curve in spacecurves module____ ── Topic: 38
     │    │              └─Parsing data files and visualizing with matplotlib and mayavi/vispy.____
     │    │                   ├─Code for drawing bounding boxes in 3D using VisPy____
     │    │                   │    ├─Plotting Lissajous Conformation with Vispy and Mayavi Toolkits____
     │    │                   │    │    ├─■──Plotting Lissajous conformations with plot_cell using Vispy and Mayavi toolkits____ ── Topic: 13
     │    │                   │    │    └─■──Understanding the plot_line() function in pyknotid's visualise.py and its 3D plotting toolkits (Maya ── Topic: 27
     │    │                   │    └─■──Bounding box visualization with VisPy____ ── Topic: 32
     │    │                   └─Analyzing knot type of curve in a data file using argparse parser and VisPy canvas plotting.____
     │    │                        ├─■──Plotting 2D projections with optional markers in Python using pyplot____ ── Topic: 48
     │    │                        └─Analysis of knot types in data files using argparse and vispy_canvas.____
     │    │                             ├─■──Working with VisPy Canvas and Scene objects____ ── Topic: 40
     │    │                             └─■──Parsing and analyzing knot types in data files using argparse____ ── Topic: 42
     │    └─Alternative Periodic Vassiliev Function with Conway Notation (Degree 4, Z4 Coefficients) and Core Cr
     │         ├─■──Calculation of writhing numbers using Arrow diagrams and Gauss codes____ ── Topic: 16
     │         └─Alternative periodic Vassiliev function for Conway notation with z4 coefficients and related invaria
     │              ├─■──Arnold's invariants and their calculation by transforming representation into an unknot____ ── Topic: 6
     │              └─■──Alternative periodic Vassiliev function in pyknotid with Conway notation and Z4 coefficients____ ── Topic: 11
     └─"Calculating higher order writhe integrals using numpy and cython"____
          ├─Calculation of Higher Order Writhe Integral with NumPy and Cython Implementation.____
          │    ├─Calculation of higher order writhe integral using points and order contributions____
          │    │    ├─■──higher_order_writhe_integral function implementation with numpy____ ── Topic: 29
          │    │    └─■──Writhing matrix and coefficient calculations for points in 3-dimensional space____ ── Topic: 46
          │    └─■──Calculation of Writhe and Average Crossing Number using Integral____ ── Topic: 21
          └─■──Distance Quantity Calculation from Curve Integral____ ── Topic: 31
```

## PyReason
- https://github.com/lab-v2/pyreason
```
.
├─Updating Interpretations Graph with Nodes and Edges____
│    ├─Analysis of Python code implementing a graph data structure and functions to add nodes and edges, us
│    │    ├─■──Code Refactoring and Commenting, Handling None Values in Python Code____ ── Topic: 10
│    │    └─Code analysis and review of a function for adding edges and nodes to a graph, including checking for
│    │         ├─■──Positive aspects of a codebase with clear and descriptive function names and variable names.____ ── Topic: 4
│    │         └─■──Methods for adding edges to a graph with considerations for existing nodes and labels.____ ── Topic: 11
│    └─Updating nodes and edges in a graph with rule trace and atom trace, using Numba for optimization____
│         ├─Python functions for updating rule traces with graph attributes____
│         │    ├─■──Updating nodes with rules and traces in a converging system.____ ── Topic: 15
│         │    └─■──Interpretation of Graph Labels Using Numba in Python____ ── Topic: 5
│         └─analysis of profiling data for program optimization____
│              ├─■──Parsing YAML to create a list of rules____ ── Topic: 9
│              └─■──Parsing and Profiling Data from CSV files in Python____ ── Topic: 1
└─Python Object Getter Implementation for Fact Objects____
     ├─Python code structure and implementation in pyreason's numba_wrapper lib____
     │    ├─Functions and objects in pyreason's numba_wrapper module that return nested functions and implement 
     │    │    ├─■──Function for getting time lower and upper bounds from a fact object____ ── Topic: 14
     │    │    └─Higher-order functions in rule_type.py for accessing fields of a rule object in pyreason/pyreason/sc
     │    │         ├─Python classes and object-oriented programming concepts with "Fact" class examples.____
     │    │         │    ├─■──Nested function type checking with isinstance in Python____ ── Topic: 8
     │    │         │    └─■──Class Fact and its attributes and methods.____ ── Topic: 7
     │    │         └─Numba implementation functions for label and world types in PyReason____
     │    │              ├─Higher-order functions for getting attributes of rules in PyReason's numba_types module____
     │    │              │    ├─■──Code structure and naming conventions in pyreason's numba_wrapper/numpy_types directory____ ── Topic: 6
     │    │              │    └─■──Implementation details of the `get_target_criteria` function and `unbox_rule` in rule_type.py____ ── Topic: 0
     │    │              └─■──Implementation of interval methods in pyreason using numba_wrapper____ ── Topic: 2
     │    └─Compliments on codebase functions for calculating minimum, maximum, average, and average lower using
     │         ├─■──Implementing a reset function to update the bounds of an interval object in Python____ ── Topic: 18
     │         └─■──Compliments on Function Names and Docstrings for Array Calculation Functions____ ── Topic: 13
     └─Working with pyapi and objects in Python code____
          ├─Understanding a Python code snippet for unboxing facts with structmodels____
          │    ├─■──Object Unboxing and Attribute Retrieval in Python with PyAPI____ ── Topic: 17
          │    └─Numba code for creating and boxing a struct model instance____
          │         ├─■──Code snippet for creating a struct proxy and boxing its components in Numba.____ ── Topic: 3
          │         └─■──Python class RuleModel with labeled attributes and types____ ── Topic: 12
          └─■──Functions for creating and boxing objects in a Python-C++ interface____ ── Topic: 16
```
