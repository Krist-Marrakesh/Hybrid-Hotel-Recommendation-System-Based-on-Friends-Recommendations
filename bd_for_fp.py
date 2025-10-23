import os
import sys
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, PrimaryKeyConstraint
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("Критическая ошибка: переменная DATABASE_URL не найдена. Создайте файл .env.")
    sys.exit(1)

DATA_FILE = "hackathon_augmented_data.csv"
FRIENDS_FILE = "friendships.csv"



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


def seed_database(db: Session):
    try:
        print("--- Начало процесса заполнения БД ---")

        print("Шаг 1/5: Удаление старых и создание новых таблиц...")
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        print("  ...таблицы успешно созданы.")


        print("Шаг 2/5: Чтение CSV-файлов...")
        df = pd.read_csv(DATA_FILE)
        friends_df = pd.read_csv(FRIENDS_FILE)
        df.rename(columns={'guest_id': 'user_id'}, inplace=True)
        print("  ...CSV-файлы успешно прочитаны.")

        print("Шаг 3/5: Загрузка уникальных пользователей (users)...")
        all_user_ids = pd.concat([
            df['user_id'],
            friends_df['user_id_1'],
            friends_df['user_id_2']
        ]).unique()
        users_to_add = [User(id=int(user_id)) for user_id in all_user_ids]
        db.add_all(users_to_add)
        db.commit()
        print(f"  ...загружено {len(users_to_add)} пользователей.")


        print("Шаг 4/5: Загрузка уникальных отелей (hotels)...")
        hotels_df = df[[
            'hotel_id', 'city', 'hotel_type', 'price_rub', 'stars', 'user_reviews_count'
        ]].drop_duplicates(subset=['hotel_id'])
        hotels_to_add = [Hotel(id=int(row.hotel_id), **row.drop('hotel_id').to_dict()) for _, row in
                         hotels_df.iterrows()]
        db.add_all(hotels_to_add)
        db.commit()
        print(f"  ...загружено {len(hotels_to_add)} отелей.")

        print("Шаг 5/5: Загрузка отзывов и дружеских связей...")
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
        print(f"  ...загружено {len(reviews_to_add)} отзывов и {len(friendships_processed)} дружеских связей.")

        print("\n[SUCCESS] База данных успешно инициализирована и заполнена!")

    except FileNotFoundError as e:
        print(f"\n[CRITICAL ERROR] Не найден файл данных: {e}. Убедитесь, что CSV лежат в том же каталоге.")
        db.rollback()
        sys.exit(1)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Произошла непредвиденная ошибка: {e}")
        db.rollback()
        sys.exit(1)
    finally:
        db.close()



if __name__ == "__main__":


    db_session = SessionLocal()

    seed_database(db=db_session)