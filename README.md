To start Reactome in Neo4J in Docker:
 ```bash
 docker run -p 7474:7474 -p 7687:7687 -e NEO4J_dbms_memory_heap_maxSize=8g reactome/graphdb:latest
 ```
