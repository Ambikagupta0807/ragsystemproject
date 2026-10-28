from assistant.repository import connection, create_schema

print("connecting to database...")

with connection() as connection:
    print("connected")
    create_schema(connection)
    print("database ok")