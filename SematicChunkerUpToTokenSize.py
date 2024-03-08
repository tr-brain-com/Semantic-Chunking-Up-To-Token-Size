from langchain.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from transformers import AutoTokenizer
from langchain.docstore.document import Document

#text'i yükle
text = """Akkuyu Nükleer AŞ Üretim ve İnşaat Organizasyon Direktörü Denis Sezemin, 8 Mart Dünya Kadınlar Günü kapsamında düzenlenen etkinlikte bu yılın sonuna doğru 1. ünitede test amaçlı ilk elektriğin verilmesinin hedeflendiğini söyledi. Santralin 1. ünitesinin reaktör binasına monte edilen kutup vincinin reaktörün kurulumunda önemli rol oynadığını belirten Sezemin, reaktörün monte edilebilirlik kontrolünün tamamlandığını (...), bir sonraki aşama olan sıcak ve soğuk test denemelerine geçilmesinin önünün açıldığını kaydetti. 1. ünitede aynı zamanda turbo jeneratör kurulumu çalışmalarının devam ettiğini vurgulayan Sezemin, söz konusu jeneratörlerin ana bileşenlerinin kurulduğunu ve bu kapsamda stator adı verilen parçanın da yerine yerleştirildiğini anlattı."""

#Embedding modeli yükle
embedding_model_name = "sentence-transformers/quora-distilbert-multilingual"
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)

#Tokenizer tanımla
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/quora-distilbert-multilingual')

#SemantikChunker oluştur
text_splitter = SemanticChunker(embeddings)

#Recursive metodumuz
def semantic_text_chunking(docs, size=100, status=True):
    if not status:
        return docs
    for idx, doc in enumerate(docs):

        if not doc.metadata["status"]:
            temp = text_splitter.create_documents([doc.page_content])

            if len(temp) == 1:
                docs[idx].metadata["status"] = True
                return semantic_text_chunking(docs, size, status)

            for t in temp:
                t_token_size = len(tokenizer(t.page_content).tokens())
                t_doc = Document(page_content=t.page_content,
                                 metadata={"status": True if t_token_size < size else False})
                docs.append(t_doc)
            docs.pop(idx)
            return semantic_text_chunking(docs, size, True)
    return docs, size, False

#Tokenizer model max token size al
model_max_token_size = tokenizer.model_max_length

#Başlangıç metin parçalarını oluştur
documentS = []

for doc in text_splitter.create_documents([text]):
    doc_t_token_size = len(tokenizer(doc.page_content).tokens())
    doc_t =  Document(page_content=doc.page_content, metadata={"status": True if doc_t_token_size < model_max_token_size else False})
    documentS.append(doc_t)

#Metin parçalarını recursive metoda gönder
docs, tokensize, status = semantic_text_chunking(documentS, model_max_token_size, True)

#Sonuçları göster
print("token size : ", model_max_token_size)
print("doc size : ", len(docs))
print("docs")
for idx, doc in enumerate(docs) :
    print(f"Chunk {idx} :\n", doc.page_content)

#Her bir oluşan chunk için token büyüklüğüne bak
for idx, doc in enumerate(docs) :
    print(f"Chunk {idx} : ",len(tokenizer(doc.page_content).tokens()))