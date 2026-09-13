from sqlalchemy import create_engine, text

DB_USER = "postgres"
DB_PASSWORD = "jayasree18"
DB_HOST = "172.17.160.1"
DB_PORT = "5432"
DB_NAME = "HEA_MPEA_DB"

DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(DATABASE_URL)

try:
    with engine.connect() as connection:
        result = connection.execute(text("SELECT version();"))

        print("PostgreSQL connected successfully!")
        print(result.fetchone()[0])

except Exception as e:
    print("PostgreSQL connection failed:")
    print(e)