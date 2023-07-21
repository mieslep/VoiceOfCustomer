
from cassandra.cluster import Session
from langchain.docstore.document import Document
from datetime import datetime

class Cassandra():
    def __init__(self, embedding, session: Session, keyspace: str, table_name: str):
        self.embedding = embedding
        self.session = session
        self.keyspace = keyspace
        self.table_name = table_name
        
        self.pstmt = self.session.prepare(
                f"SELECT asin, overall_rating, review_text, reviewer_name, unix_review_time FROM {self.keyspace}.{self.table_name} WHERE asin = ? ORDER BY embedding_vector ann of ? LIMIT ?")
        
        self.pstmt_by_rating = self.session.prepare(
                f"SELECT asin, overall_rating, review_text, reviewer_name, unix_review_time FROM {self.keyspace}.{self.table_name} WHERE asin = ? AND overall_rating = ? ORDER BY embedding_vector ann of ? LIMIT ?")

    def similarity_search(self, search_string: str, k: int, asin: str, overall_rating: int = None):
        # Generate the embedding for the search string
        search_embedding = self.embedding.embed_query(search_string)

        if overall_rating is None:
            rows = self.session.execute(self.pstmt, [asin, search_embedding, k])
        else:
            rows = self.session.execute(self.pstmt_by_rating, [asin, overall_rating, search_embedding, k])

        documents = []

        for row in rows:
            documents.append(Document(page_content=row.review_text, metadata={'asin': row.asin, 'overall_rating': row.overall_rating, 'reviewer_name': row.reviewer_name, 'review_time': row.unix_review_time.strftime("%d %B %Y")}))

        return documents