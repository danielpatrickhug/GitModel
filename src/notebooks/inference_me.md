```python
from ast_parsers.python_ast_parser import parse_python_file, get_methods, parse_github_repo
from system_prompts.format_system_prompts import format_system_prompts, format_system_prompts_with_tree
from ..ml_models.transformers.chatgpt_api_inference import process_transcript
from ..repo_graph_generation import decompose_repo
from ..ml_models.graph_networks.kernels import compute_kernel_by_type, graph_laplacian, compute_kernel
from ..ml_models.graph_networks.message_passing import k_hop_message_passing_sparse
from ..ml_models.transformers.sentence_embeddings import embed_data
from ..ml_models.topic_modeling.umap_hdbscan_pipeline import (
    load_topic_model,
    get_topic_model,
    get_representative_docs,
    reduce_outliers,
    fit_topic_model,
    compute_hierarchical_topic_tree,
    get_topic_info,
)


from getpass import getpass

openai_secret = getpass("Enter the secret value: ")
# Set up OpenAI API credentials
openai.api_key = openai_secret


!git clone https://github.com/danielpatrickhug/GitModel.git
```
TODO. standardize this
```python
git_repo_path = "/content/GitModel"
out_path = "/content/gitmodel_sum"
name_id = "gitmodel"
contents = parse_github_repo(git_repo_path)
print(len(contents))
pruned_contents = []
for cont in contents:
    fp = cont["file_name"]
    fn = fp.split("/")[-1]
    fn_ = fn.split(".")[0]
    if fn_ in ["__init__"] or fn_.split("_")[-1] in ["test"]:
        continue
    else:
        print(cont["file_name"])
        pruned_contents.append(cont)
        
decompose_repo()
```
```python
def load_jsonl(filepaths):
    data = []
    for filepath in filepaths:
        with open(filepath, "r") as f:
            for line in f:
                data.append(json.loads(line))
    return data


root_dir = "/content/gitmodel_sum"
repo = "gitmodel"
repo_files = [
    f"{root_dir}/{repo}_summary.jsonl",
    f"{root_dir}/{repo}_question_asking.jsonl",
]  

res = load_jsonl(repo_files)
sents = []
for r in res:
    messages = r["conversation_history"]
    reply = r["assistant_reply"]
    sents.append(reply)
    sents.append(messages[-2]["content"])
```

```python
data = pd.DataFrame(sents, columns=["query"])
data["_id"] = data.index
```
```python
MODEL_NAME = "all-MiniLM-L6-v2"  # "allenai-specter"#
embs = embed_data(data, model_name=MODEL_NAME)
A = compute_kernel_by_type(embs, threshold=0.6, kernel_type="cosine")
k=2
A_k, agg_features = k_hop_message_passing_sparse(A, embs, k)
```
Graph Laplacian
```python
L, D = graph_laplacian(A)
L_k, D_k = graph_laplacian(A_k)
```
SVD for when the heads

```python
U, S, VT = np.linalg.svd(A)
print(f"U: {U.shape}\n")
print(f"S: {S.shape}\n")
print(f"VT: {VT.shape}\n")
plt.plot(np.diag(S))
plt.xlabel("Singular value index")
plt.ylabel("Singular value")
plt.title("Singular values of A")
plt.show()
```

```python
U_k, S_k, VT_k = np.linalg.svd(A_k)
print(f"U_{k}: {U_k.shape}\n")
print(f"S_{k}: {S_k.shape}\n")
print(f"VT_{k}: {VT_k.shape}\n")
plt.plot(np.diag(S_k))
plt.xlabel("Singular value index")
plt.ylabel("Singular value")
plt.title("Singular values of A_k")
plt.show()
```

```python
topic_model = load_topic_model(nr_topics="auto")
topics, probs = fit_topic_model(topic_model, data, agg_features)
freq = get_topic_info(topic_model)
rep_docs = topic_model
hr, tree = compute_hierarchical_topic_tree(topic_model=topic_model, data=data)
```





