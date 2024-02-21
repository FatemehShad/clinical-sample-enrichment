# Reactome
## Reactome via Graphite
See [here](https://bioconductor.org/packages/devel/bioc/vignettes/graphite/inst/doc/graphite.pdf)

## Reactome via Neo4J
### Starting Reacome in Docker
To start Reactome in Neo4J in Docker:
 ```bash
 docker run -p 7474:7474 -p 7687:7687 -e NEO4J_dbms_memory_heap_maxSize=8g reactome/graphdb:latest
 ```

### Some queries for Reactome
* Find the number of nodes:
```cypher
MATCH(n) RETURN count(n) AS node_count
```
(answer: 2 439 640)
* Find the number of edges:
```cypher
MATCH ()-[r]->()
RETURN count(r) AS edge_count
```
(answer: 10 239 890)
* Find the nodes connected to the mouse:
```cypher
MATCH (n {abbreviation: 'MMU'}) -- (m) RETURN m LIMIT 15
```
* Retrieves the info about the node with UniProt ID: A0A075B5J3
```cypher
MATCH (n {identifier: 'A0A075B5J3'}) -- (m) RETURN *
```
* Retrieves the info about the node with ReactToMe ID: R-MMU-198955
```cypher
MATCH (n {stId: 'R-MMU-198955'}) RETURN n LIMIT 25
```
* Retrieves all the path between excluding the _created_ edges
```cypher
MATCH p = (r {stId: 'R-MMU-198955'})-[*..3]-(u {identifier: 'A0A075B5J3'}) 
WITH *, relationships(p) AS e 
WHERE NONE( rel in e WHERE type(rel) = 'created') 
RETURN *
```
* Retrieves all the path between excluding the _created_ and *species* edges
```cypher
MATCH p = (r {stId: 'R-MMU-198955'})-[*..4]-(u {identifier: 'A0A075B5J3'}) 
WITH *, relationships(p) AS rs
WHERE NONE(rel in rs WHERE type(rel) = 'created' OR type(rel) = 'species') 
RETURN p 
LIMIT 1
```

```cypher
MATCH (o)-[r:species]->(m:Taxon{abbreviation:'MMU'})
RETURN o
LIMIT 10
UNION
MATCH (o)-[s]-(n)-[r:species]->(m:Taxon{abbreviation:'MMU'})
WHERE type(s) <> 'created'
AND type(s) <> "reference_Database"
RETURN o
LIMIT 10
```
