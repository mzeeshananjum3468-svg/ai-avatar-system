import psycopg2
from psycopg2.extras import RealDictCursor

# -----------------------------
# Database Configuration
# -----------------------------
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "avatar_db",
    "user": "avatar_user",
    "password": "avatar_user",
}

# -----------------------------
# Connect
# -----------------------------
conn = psycopg2.connect(**DB_CONFIG)
conn.autocommit = True

print("✅ Connected to PostgreSQL")
print("Type SQL below.")
print("Type 'exit' or 'quit' to close.\n")

cursor = conn.cursor(cursor_factory=RealDictCursor)

while True:
    try:
        query = input("SQL> ").strip()

        if query.lower() in ("exit", "quit"):
            break

        # Reset avatar data while keeping users
        if query.lower() == "reset":
            cursor.execute("""
                DELETE FROM messages;
                DELETE FROM conversations;
                DELETE FROM sessions;
                DELETE FROM avatars;
            """)
            print("✅ Deleted all avatars, sessions, messages, and conversations.\n")
            continue

        if not query.endswith(";"):
            query += ";"

        cursor.execute(query)

        if cursor.description:
            rows = cursor.fetchall()

            if not rows:
                print("No rows returned.\n")
                continue

            print()
            for i, row in enumerate(rows, 1):
                print(f"----- Row {i} -----")
                for k, v in row.items():
                    print(f"{k}: {v}")
                print()

        else:
            print(f"✅ Query OK ({cursor.rowcount} rows affected)\n")

    except Exception as e:
        print("\n❌ ERROR")
        print(e)
        print()