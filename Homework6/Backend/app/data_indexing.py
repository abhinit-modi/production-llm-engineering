import os
import chromadb
import uuid
from pathlib import Path
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from pydantic import BaseModel
from clients import openai_client
from langchain_community.vectorstores import Chroma


current_dir = Path(__file__).resolve().parent


class CodeDoc(BaseModel):
    content: str
    source: str



class DataIndexer:

    source_file =  os.path.join(current_dir, 'sources.txt')

    def __init__(self, index_name='langchain-repo') -> None:
        # self.embedding_client = InferenceClient(
        #     "dunzhang/stella_en_1.5B_v5",
        # )
        self.embedding_client = openai_client
        self.index_name = index_name
        self.pinecone_client = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))

        if index_name not in self.pinecone_client.list_indexes().names():
            # TODO: create your index if it doesn't exist. Use the create_index function. 
            # Make sure to choose the dimension that corresponds to your embedding model
            self.pinecone_client.create_index(
                name=self.index_name,
                dimension=len(self.embedding_client.embeddings.create("test").data[0].embedding),  # type: ignore
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                ))

        self.index = self.pinecone_client.Index(self.index_name)
        # TODO: make sure to build the index.
        self.source_index = self.get_source_index()

    def get_source_index(self):

        client_db = chromadb.PersistentClient()
        index = client_db.get_or_create_collection(
            name=self.index_name
        )
        if index.count() != 0:
            return index
        
        if not os.path.isfile(self.source_file):
            print('No source file')
            return None
                
        with open(self.source_file, 'r') as file:
            sources = file.readlines()
            
        sources = list(set([s.rstrip('\n') for s in sources if s.rstrip('\n')]))
        batch_size = 1000
        for i in range(0, len(sources), batch_size):
            batch = sources[i:i+batch_size]
            response = openai_client.embeddings.create(
                input=batch,
                model="text-embedding-3-small"
            )
            embeddings = [res.embedding for res in response.data]
            index.add(
                documents=batch,
                embeddings=embeddings,
                ids=batch
            )
        return index
    
    def embed_data(self,  data: list[CodeDoc]) -> list[list[float]]:
        # TODO: implement the function. The function takes a list of CodeDoc and returns 
        # a list of the related embeddings
        response = self.embedding_client.embeddings.create(
            input=[doc.content for doc in data],
            model="text-embedding-3-small"
        )
        embeddings =[res.embedding for res in response.data]
        return embeddings
        
    def index_data(self, docs: list[CodeDoc], batch_size:int = 32):

        with open(self.source_file, 'a') as file:
            for doc in docs:
                file.writelines(doc.source + '\n')

        for i in range(0, len(docs), batch_size):
            batch = docs[i: i + batch_size]

            # TODO: create a list of the vector representations of each text data in the batch
            values = self.embed_data(batch)

            # TODO: create a list of unique identifiers for each element in the batch with the uuid package.
            vector_ids = [str(uuid.uuid4()) for _ in range(len(batch))]

            # TODO: create a list of dictionaries representing the metadata. You can use the model_dump() on 
            # a CodeDoc instance
            metadatas = [{
                    **doc.model_dump(),
                } for doc in batch]

            # create a list of dictionaries with keys "id" (the unique identifiers), "values"
            # (the vector representation), and "metadata" (the metadata).
            vectors = [{
                'id': vector_id,
                'values': value,
                'metadata': metadata
            } for vector_id, value, metadata in zip(vector_ids, values, metadatas)]

            try: 
                # TODO: Use the function upsert to upload the data to the database.
                upsert_response = self.index.upsert(
                    vectors=vectors,  # type: ignore
                    namespace='langchain_repo'
                )
                print(upsert_response)
            except Exception as e:
                print(e)

    def search(self, text_query, top_k=5, hybrid_search=False) -> list[CodeDoc]:
        # TODO: embed the text_query by using the embedding model
        vector = self.embedding_client.embeddings.create(input=text_query, model="text-embedding-3-small").data[0].embedding

        filter = None
        if hybrid_search and self.source_index:
            # I implemented the filtering process to pull the 50 most relevant file names
            # to the question. Make sure to adjust this number as you see fit.
            results = self.source_index.query(
                query_embeddings=[vector],
                n_results=50,
                include=["documents"]
            )
            sources = results["documents"][0]
            filter = {"source": {"$in": sources}}
            print(f"Sources: {sources}")

         # TODO: use the vector representation of the text_query to 
         # search the database by using the query function.
        result = self.index.query(
            vector=vector,
            top_k=top_k,
            namespace='langchain_repo',
            include_metadata=True,
            filter=filter, # type: ignore
        )

        docs = []
        for res in result["matches"]:
            # TODO: use the model_validate() function to create a 
            # CodeDoc instance from the result's metadata.
            # e.g. doc = CodeDoc.model_validate(res["metadata"])
            doc = CodeDoc.model_validate(res["metadata"])
            docs.append(doc)

        return docs

    

if __name__ == '__main__':

    from langchain_community.document_loaders import GitLoader
    from langchain_text_splitters import (
        Language,
        RecursiveCharacterTextSplitter,
    )

    loader = GitLoader(
        clone_url="https://github.com/langchain-ai/langchain",
        repo_path="./code_data/langchain_repo/",
        branch="master",
    )

    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=10000, chunk_overlap=100
    )

    docs = loader.load()
    docs = [doc for doc in docs if doc.metadata['file_type'] in ['.py', '.md']]
    docs = [doc for doc in docs if len(doc.page_content) < 50000]
    docs = python_splitter.split_documents(docs)
    code_docs = []
    for doc in docs:
        doc.page_content = '# {}\n\n'.format(doc.metadata['source']) + doc.page_content
        code_doc = CodeDoc(
            content=doc.page_content,
            source=doc.metadata['source']
        )
        code_docs.append(code_doc)

    indexer = DataIndexer()
    with open(os.path.join(current_dir, 'sources.txt'), 'a') as file:
        for doc in code_docs:
            file.writelines(doc.source + '\n')

    indexer.index_data(code_docs)