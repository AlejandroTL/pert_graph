from neo4j import GraphDatabase


class GRNToNeo4J:

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def delete_all(self):
        return """  MATCH (n) DETACH DELETE n"""

    def insert_inhibits_genes(self):
        return """  LOAD CSV WITH HEADERS FROM 'file:///Neo4j_triples_inhibits_gene.csv' AS line FIELDTERMINATOR ',' 
                    MERGE (g:Gene {name: line.subject})
                    MERGE (d:Gene {name: line.object})
                    WITH g,d
                    MATCH (g), (d)
                    MERGE (g)-[:INHIBITS]->(d)"""

    def insert_stimulates_genes(self):
        return """  LOAD CSV WITH HEADERS FROM 'file:///Neo4j_triples_stimulates_gene.csv' AS line FIELDTERMINATOR ',' 
                    MERGE (g:Gene {name: line.subject})
                    MERGE (d:Gene {name: line.object})
                    WITH g,d
                    MATCH (g), (d)
                    MERGE (g)-[:STIMULATES]->(d)"""

    def insert_stimulates_drugs(self):
        return """  LOAD CSV WITH HEADERS FROM 'file:///Neo4j_triples_stimulates_drug.csv' AS line FIELDTERMINATOR ',' 
                    MERGE (g:Drug {name: line.subject})
                    MERGE (d:Gene {name: line.object})
                    WITH g,d
                    MATCH (g), (d)
                    MERGE (g)-[:STIMULATES]->(d)"""

    def insert_inhibits_drugs(self):
        return """  LOAD CSV WITH HEADERS FROM 'file:///Neo4j_triples_inhibits_drug.csv' AS line FIELDTERMINATOR ',' 
                    MERGE (g:Drug {name: line.subject})
                    MERGE (d:Gene {name: line.object})
                    WITH g,d
                    MATCH (g), (d)
                    MERGE (g)-[:INHIBITS]->(d)"""

    def run(self):
        session = self.driver.session()

        queries = [self.delete_all(), self.insert_inhibits_genes(), self.insert_stimulates_genes(),
                   self.insert_stimulates_drugs(), self.insert_inhibits_drugs()]
        for query in queries:
            session.run(query)

        self.close()


if __name__ == '__main__':
    db = GRNToNeo4J(uri='bolt://localhost:7687', user='neo4j', password='grn')
    db.run()
