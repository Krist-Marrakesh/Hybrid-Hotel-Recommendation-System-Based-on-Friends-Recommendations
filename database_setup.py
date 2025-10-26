import os
import sys
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, PrimaryKeyConstraint, JSON, DateTime, Boolean
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("CRITICAL ERROR: DATABASE_URL variable not found. Please create a .env file.")
    sys.exit(1)
DATA_FILE = "data/hackathon_augmented_data.csv"
FRIENDS_FILE = "data/friendships.csv"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)

class Hotel(Base):
    __tablename__ = 'hotels'
    id = Column(Integer, primary_key=True, index=True)
    city = Column(String, index=True)
    hotel_type = Column(String)
    price_rub = Column(Float)
    stars = Column(Integer)
    user_reviews_count = Column(Integer)

class Review(Base):
    __tablename__ = 'reviews'
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), index=True)
    hotel_id = Column(Integer, ForeignKey('hotels.id'), index=True)
    rating_overall = Column(Float)
    rating_location = Column(Float)
    rating_cleanliness = Column(Float)
    rating_food = Column(Float)
    rating_service = Column(Float)
    was_booked = Column(Integer)

class Friendship(Base):
    __tablename__ = 'friendships'
    user_id_1 = Column(Integer, ForeignKey('users.id'))
    user_id_2 = Column(Integer, ForeignKey('users.id'))
    __table_args__ = (PrimaryKeyConstraint('user_id_1', 'user_id_2'),)

class MLModel(Base):
    __tablename__ = 'ml_models'
    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(String, unique=True, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_active = Column(Boolean, default=False, index=True)
    metrics = Column(JSON)
    hyperparameters = Column(JSON)
    model_weights_path = Column(String, nullable=False)
    item_embeddings_path = Column(String, nullable=False)
    artifacts_path = Column(String, nullable=False)

def seed_database(db: Session):
    try:
        print("--- Starting database seeding process ---")
        print("Step 1/5: Dropping old tables and creating new ones...")
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        print("  ...tables created successfully.")
        print("Step 2/5: Reading CSV files...")
        df = pd.read_csv(DATA_FILE)
        friends_df = pd.read_csv(FRIENDS_FILE)
        df.rename(columns={'guest_id': 'user_id'}, inplace=True)
        print("  ...CSV files read successfully.")
        print("Step 3/5: Seeding unique users...")
        all_user_ids = pd.concat([
            df['user_id'],
            friends_df['user_id_1'],
            friends_df['user_id_2']
        ]).unique()
        users_to_add = [User(id=int(user_id)) for user_id in all_user_ids]
        db.add_all(users_to_add)
        db.commit()
        print(f"  ...seeded {len(users_to_add)} users.")
        print("Step 4/5: Seeding unique hotels...")
        hotels_df = df[[
            'hotel_id', 'city', 'hotel_type', 'price_rub', 'stars', 'user_reviews_count'
        ]].drop_duplicates(subset=['hotel_id'])
        hotels_to_add = [Hotel(id=int(row.hotel_id), **row.drop('hotel_id').to_dict()) for _, row in
                         hotels_df.iterrows()]
        db.add_all(hotels_to_add)
        db.commit()
        print(f"  ...seeded {len(hotels_to_add)} hotels.")
        print("Step 5/5: Seeding reviews and friendships...")
        reviews_df = df[[
            'user_id', 'hotel_id', 'rating_overall', 'rating_location',
            'rating_cleanliness', 'rating_food', 'rating_service', 'was_booked'
        ]]
        reviews_to_add = [Review(**row.to_dict()) for _, row in reviews_df.iterrows()]
        db.add_all(reviews_to_add)

        friendships_processed = set()
        for _, row in friends_df.iterrows():
            u1, u2 = sorted((int(row.user_id_1), int(row.user_id_2)))
            if (u1, u2) not in friendships_processed:
                db.add(Friendship(user_id_1=u1, user_id_2=u2))
                friendships_processed.add((u1, u2))
        db.commit()
        print(f"  ...seeded {len(reviews_to_add)} reviews and {len(friendships_processed)} friendships.")
        print("\n[SUCCESS] Database has been successfully initialized and seeded!")
        
    except FileNotFoundError as e:
        print(f"\n[CRITICAL ERROR] Data file not found: {e}. Make sure the CSV files are in the correct directory.")
        db.rollback()
        sys.exit(1)
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] An unexpected error occurred: {e}")
        db.rollback()
        sys.exit(1)
    finally:
        db.close()

if __name__ == "__main__":
    db_session = SessionLocal()
    seed_database(db=db_session)
