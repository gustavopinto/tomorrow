from sentence_transformers import SentenceTransformer

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# The sentences to encode
chunks = ["pizza", # imagine que essa lista é um banco de dados
             "man", 
             "king", 
             "google", 
             "federal university of bahia", # ruido
             "lasagna"]

embeddings = model.encode(chunks)

user_query = "what kind of food I eat in Italy?"
query = model.encode([user_query])


for i in range(len(embeddings)):
    similarities = model.similarity(query[0], embeddings[i])
    print(chunks[i], user_query, similarities)


# 1. os embeddings tem sempre o mesmo tamanho
# 2. os valores dos embeddings são deterministicos
# 3. nao é somente criar os embeddings, devo tambem testar o vocabulario