from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String, index=True)
    camera_name = Column(String)
    timestamp = Column(Integer)
    image_url = Column(String)
    distance = Column(Float)
    description = Column(String)
