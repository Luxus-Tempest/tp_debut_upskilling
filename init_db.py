import time
import psycopg
from psycopg.errors import DuplicateObject

DB_CONFIG = {
    "host": "localhost",
    "port": 5532,
    "dbname": "ai",
    "user": "ai",
    "password": "ai",
}

def wait_for_postgres(max_retries=10, delay=3):
    """Attend que PostgreSQL soit disponible avant de continuer."""
    for i in range(max_retries):
        try:
            with psycopg.connect(**DB_CONFIG) as conn:
                print("✅ PostgreSQL est prêt.")
                return True
        except Exception as e:
            print(f"⏳ Tentative {i+1}/{max_retries} : PostgreSQL pas encore prêt ({e})")
            time.sleep(delay)
    print("❌ Impossible de se connecter à PostgreSQL.")
    return False

def init_pgvector():
    """Crée l'extension pgvector si elle n'existe pas."""
    try:
        with psycopg.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                conn.commit()
                print("✅ Extension 'pgvector' vérifiée ou installée avec succès.")
    except DuplicateObject:
        print("ℹ️  L’extension 'vector' existe déjà.")
    except Exception as e:
        print(f"❌ Erreur lors de la création de l’extension vector : {e}")

def main():
    print("🚀 Initialisation de la base PostgreSQL pour Agno...")
    if wait_for_postgres():
        init_pgvector()
    else:
        print("⚠️ PostgreSQL n’est pas prêt. Vérifie ton conteneur Docker.")

if __name__ == "__main__":
    main()
