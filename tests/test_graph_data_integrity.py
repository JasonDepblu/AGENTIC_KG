"""
Test script to verify data integrity between extracted CSV files and Neo4j graph.
Checks that all entities and relationships from targeted_*.csv files are properly imported.
"""

import pandas as pd
import sys
import os
from pathlib import Path
from neo4j import GraphDatabase

# Project root
project_root = Path(__file__).parent.parent


class SimpleGraphDB:
    """Simple Neo4j client for testing."""

    def __init__(self):
        # Load config from .env file
        env_file = project_root / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        os.environ.setdefault(key, value)

        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = os.environ.get("NEO4J_USERNAME", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD", "password123")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def send_query(self, query: str, params: dict = None) -> dict:
        try:
            with self.driver.session() as session:
                result = session.run(query, params or {})
                records = [dict(record) for record in result]
                return {"status": "success", "query_result": records}
        except Exception as e:
            return {"status": "error", "error_message": str(e)}

    def close(self):
        self.driver.close()


def get_graphdb():
    return SimpleGraphDB()


def load_csv_data(data_dir: Path) -> dict:
    """Load all targeted CSV files into a dictionary."""
    csv_files = {
        "respondent_entities": "targeted_respondent_entities.csv",
        "aspect_entities": "targeted_aspect_entities.csv",
        "brand_entities": "targeted_brand_entities.csv",
        "model_entities": "targeted_model_entities.csv",
        "store_entities": "targeted_store_entities.csv",
        "rates_relationships": "targeted_rates_relationships.csv",
        "evaluated_brand_relationships": "targeted_evaluated_brand_relationships.csv",
        "evaluated_model_relationships": "targeted_evaluated_model_relationships.csv",
        "visited_store_relationships": "targeted_visited_store_relationships.csv",
    }

    data = {}
    for key, filename in csv_files.items():
        filepath = data_dir / filename
        if filepath.exists():
            data[key] = pd.read_csv(filepath)
            print(f"Loaded {filename}: {len(data[key])} rows")
        else:
            print(f"Warning: {filename} not found")
            data[key] = pd.DataFrame()

    return data


def check_entities_in_neo4j(graphdb, csv_data: dict) -> dict:
    """Check if all entities from CSV exist in Neo4j."""
    results = {}

    entity_configs = [
        ("respondent_entities", "Respondent", "respondent_id"),
        ("aspect_entities", "Aspect", "aspect_name"),
        ("brand_entities", "Brand", "brand_name"),
        ("model_entities", "Model", "model_name"),
        ("store_entities", "Store", "store_name"),
    ]

    for csv_key, label, id_column in entity_configs:
        df = csv_data.get(csv_key, pd.DataFrame())
        if df.empty:
            continue

        csv_ids = set(df[id_column].astype(str).tolist())

        # Query Neo4j for all entities of this type
        query = f"MATCH (n:{label}) RETURN n.{id_column} as id"
        result = graphdb.send_query(query)

        if result.get("status") == "success":
            neo4j_ids = set(str(r["id"]) for r in result.get("query_result", []) if r.get("id"))
        else:
            neo4j_ids = set()
            print(f"Error querying {label}: {result.get('error_message')}")

        missing = csv_ids - neo4j_ids
        extra = neo4j_ids - csv_ids

        results[label] = {
            "csv_count": len(csv_ids),
            "neo4j_count": len(neo4j_ids),
            "missing_in_neo4j": list(missing)[:20],  # Limit to first 20
            "missing_count": len(missing),
            "extra_in_neo4j": list(extra)[:20],
            "extra_count": len(extra),
        }

        status = "✅" if len(missing) == 0 else "❌"
        print(f"{status} {label}: CSV={len(csv_ids)}, Neo4j={len(neo4j_ids)}, Missing={len(missing)}")
        if missing:
            print(f"   Missing samples: {list(missing)[:5]}")

    return results


def check_relationships_in_neo4j(graphdb, csv_data: dict) -> dict:
    """Check if all relationships from CSV exist in Neo4j."""
    results = {}

    relationship_configs = [
        ("rates_relationships", "RATES", "from_id", "to_id", "Respondent", "Aspect", "respondent_id", "aspect_name"),
        ("evaluated_brand_relationships", "EVALUATED_BRAND", "from_id", "to_id", "Respondent", "Brand", "respondent_id", "brand_name"),
        ("evaluated_model_relationships", "EVALUATED_MODEL", "from_id", "to_id", "Respondent", "Model", "respondent_id", "model_name"),
        ("visited_store_relationships", "VISITED_STORE", "from_id", "to_id", "Respondent", "Store", "respondent_id", "store_name"),
    ]

    for csv_key, rel_type, from_col, to_col, from_label, to_label, from_prop, to_prop in relationship_configs:
        df = csv_data.get(csv_key, pd.DataFrame())
        if df.empty:
            continue

        # Build set of expected relationships from CSV
        csv_rels = set()
        for _, row in df.iterrows():
            from_id = str(row[from_col])
            to_id = str(row[to_col])
            csv_rels.add((from_id, to_id))

        # Query Neo4j for all relationships of this type
        query = f"""
        MATCH (a:{from_label})-[r:{rel_type}]->(b:{to_label})
        RETURN a.{from_prop} as from_id, b.{to_prop} as to_id
        """
        result = graphdb.send_query(query)

        if result.get("status") == "success":
            neo4j_rels = set()
            for r in result.get("query_result", []):
                from_id = str(r.get("from_id", ""))
                to_id = str(r.get("to_id", ""))
                if from_id and to_id:
                    neo4j_rels.add((from_id, to_id))
        else:
            neo4j_rels = set()
            print(f"Error querying {rel_type}: {result.get('error_message')}")

        missing = csv_rels - neo4j_rels
        extra = neo4j_rels - csv_rels

        results[rel_type] = {
            "csv_count": len(csv_rels),
            "neo4j_count": len(neo4j_rels),
            "missing_in_neo4j": list(missing)[:20],
            "missing_count": len(missing),
            "extra_in_neo4j": list(extra)[:20],
            "extra_count": len(extra),
        }

        status = "✅" if len(missing) == 0 else "❌"
        print(f"{status} {rel_type}: CSV={len(csv_rels)}, Neo4j={len(neo4j_rels)}, Missing={len(missing)}")
        if missing:
            print(f"   Missing samples: {list(missing)[:5]}")

    return results


def check_specific_relationship(graphdb, from_id: str, to_id: str, rel_type: str):
    """Check if a specific relationship exists."""
    query = f"""
    MATCH (a)-[r:{rel_type}]->(b)
    WHERE a.respondent_id = $from_id
    RETURN a, r, b
    """
    result = graphdb.send_query(query, {"from_id": from_id})

    print(f"\n=== Checking {from_id} -[{rel_type}]-> {to_id} ===")

    if result.get("status") == "success":
        records = result.get("query_result", [])
        print(f"Found {len(records)} {rel_type} relationships from {from_id}")
        for rec in records[:5]:
            print(f"  -> {rec}")
    else:
        print(f"Error: {result.get('error_message')}")


def check_api_sample_coverage(csv_data: dict):
    """Check if API sample endpoint returns complete data."""
    import requests

    try:
        response = requests.get("http://localhost:8000/api/graph/sample?limit=1000", timeout=10)
        if response.status_code != 200:
            print(f"API Error: {response.status_code}")
            return

        data = response.json()
        nodes = data.get("nodes", [])
        links = data.get("links", [])

        print("\n=== API Sample Coverage ===")

        # Count nodes by label
        node_counts = {}
        for n in nodes:
            label = n.get("label", "Unknown")
            node_counts[label] = node_counts.get(label, 0) + 1

        for label, count in sorted(node_counts.items()):
            print(f"  {label}: {count}")

        # Count relationships by type
        rel_counts = {}
        for l in links:
            rel_type = l.get("type", "Unknown")
            rel_counts[rel_type] = rel_counts.get(rel_type, 0) + 1

        print("\nRelationship counts in sample:")
        for rel_type, count in sorted(rel_counts.items()):
            print(f"  {rel_type}: {count}")

        # Check specific relationship coverage
        rates_links = [l for l in links if l.get("type") == "RATES"]

        # Build a map from node ID to respondent_id
        node_id_to_respondent = {}
        node_id_to_aspect = {}
        for n in nodes:
            if n.get("label") == "Respondent":
                rid = n.get("properties", {}).get("respondent_id")
                if rid:
                    node_id_to_respondent[n["id"]] = rid
            elif n.get("label") == "Aspect":
                aid = n.get("properties", {}).get("aspect_name")
                if aid:
                    node_id_to_aspect[n["id"]] = aid

        # Check if Respondent_43 -> Aspect_4 is in sample
        sample_rates = set()
        for l in rates_links:
            from_rid = node_id_to_respondent.get(l.get("source"))
            to_aid = node_id_to_aspect.get(l.get("target"))
            if from_rid and to_aid:
                sample_rates.add((from_rid, to_aid))

        print(f"\nTotal RATES in sample: {len(sample_rates)}")

        # Check specific relationship
        test_rel = ("Respondent_43", "Aspect_4")
        if test_rel in sample_rates:
            print(f"✅ {test_rel} found in API sample")
        else:
            print(f"❌ {test_rel} NOT found in API sample")
            # Check if it exists at all in CSV
            rates_df = csv_data.get("rates_relationships", pd.DataFrame())
            csv_has = not rates_df[(rates_df["from_id"] == "Respondent_43") & (rates_df["to_id"] == "Aspect_4")].empty
            print(f"   CSV has this relationship: {csv_has}")

    except Exception as e:
        print(f"Error checking API: {e}")


def main():
    print("=" * 60)
    print("Graph Data Integrity Test")
    print("=" * 60)

    # Load CSV data
    data_dir = project_root / "data" / "extracted_data"
    print(f"\nLoading CSV files from: {data_dir}\n")
    csv_data = load_csv_data(data_dir)

    # Connect to Neo4j
    print("\n" + "=" * 60)
    print("Checking Neo4j Data")
    print("=" * 60)

    graphdb = get_graphdb()

    # Check entities
    print("\n--- Entity Checks ---")
    entity_results = check_entities_in_neo4j(graphdb, csv_data)

    # Check relationships
    print("\n--- Relationship Checks ---")
    rel_results = check_relationships_in_neo4j(graphdb, csv_data)

    # Check specific problematic relationship
    check_specific_relationship(graphdb, "Respondent_43", "Aspect_4", "RATES")

    # Check API sample coverage
    print("\n" + "=" * 60)
    print("Checking API Sample Coverage")
    print("=" * 60)
    check_api_sample_coverage(csv_data)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_ok = True
    for label, result in entity_results.items():
        if result["missing_count"] > 0:
            all_ok = False
            print(f"❌ {label}: {result['missing_count']} entities missing in Neo4j")

    for rel_type, result in rel_results.items():
        if result["missing_count"] > 0:
            all_ok = False
            print(f"❌ {rel_type}: {result['missing_count']} relationships missing in Neo4j")

    if all_ok:
        print("✅ All entities and relationships are properly imported!")

    return entity_results, rel_results


if __name__ == "__main__":
    main()
