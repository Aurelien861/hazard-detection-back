from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from alertModel import Base

DATABASE_URL = "postgresql://user:password@db:5432/alerts_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

def get_db_session():
    return SessionLocal()

# À appeler une fois au démarrage pour créer les tables
def init_db():
    Base.metadata.create_all(bind=engine)
