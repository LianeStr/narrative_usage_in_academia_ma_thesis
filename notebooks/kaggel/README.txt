#####################################
getembeddings_bert-base.ipynb 
input: data/mentions/narrative_mentions.jsonl
output in: data/embeddings/bert-base/
run on 24.08.2026 
runtime & processors: 51m 49s · GPU T4 x2
#####################################
getembeddings_scibert.ipynb 
input: data/mentions/narrative_mentions.jsonl
output in: data/embeddings/scibert/
run on 27.08.2026 
runtime & processors: 59m 17s · GPU T4 x2
#####################################
create-embeddings-samples-bertbase.ipynb
input: 
data/sample/sample_tuning_10per.json
data/embeddings/bert-base/chunks

#####################################
create-embeddings-samples-scibert.ipynb
input: 
data/sample/sample_tuning_10per.json
data/embeddings/scibert/chunks

#####################################
bert-base-sample-tuning-kdist.ipynb
input: 
data/sample/sample_tuning_10per.json
data/embeddings/bert-base/chunks

#####################################
all the clustering runs on the sample:
sample-tuning-dbscan-bb-12.ipynb
sample-tuning-dbscan-bb-mean.ipynb
sample-tuning-dbscan-sb-11.ipynb
sample-tuning-dbscan-sb-12.ipynb
sample-tuning-dbscan-sb-mean.ipynb
sample-tuning-dbscan-bb-11.ipynb

input: 
data/sample/sample_tuning_10per.json
for bb (bert-base):
data/embeddings/bert-base/chunks
for sb (scibert):
data/embeddings/sci_bert/chunks/

#####################################
