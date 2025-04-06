import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import sys


class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.movie_embedding.weight)

        self.fc1 = nn.Linear(embedding_dim * 2, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.1467)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.1018)

        self.fc3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, user_ids, movie_ids):
        u = self.user_embedding(user_ids)
        m = self.movie_embedding(movie_ids)
        x = torch.cat([u, m], dim=1)
        x = self.dropout1(F.leaky_relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.leaky_relu(self.bn2(self.fc2(x))))
        x = F.leaky_relu(self.fc3(x))
        return self.output(x).squeeze()


# --------------------
# Función de recomendación
# --------------------
def recommend_movies(user_original_id, model, movies_df, ratings_df, user2idx, movie2idx, top_k=10):
    model.eval()
    user_idx = user2idx.get(user_original_id)
    if user_idx is None:
        print("⚠️ User not found.")
        return

    seen_movies = ratings_df[ratings_df["UserID"] == user_idx]["MovieID"].values
    unseen_movies = [mid for mid in movie2idx.values() if mid not in seen_movies]

    user_tensor = torch.tensor([user_idx] * len(unseen_movies), dtype=torch.long)
    movie_tensor = torch.tensor(unseen_movies, dtype=torch.long)

    with torch.no_grad():
        predictions = model(user_tensor, movie_tensor)
        predictions = torch.clamp(predictions, 0.0, 1.0)
        predicted_scores = predictions.numpy() * 4 + 1

    top_indices = np.argsort(predicted_scores)[-top_k:][::-1]
    top_movie_ids = [
        list(movie2idx.keys())[list(movie2idx.values()).index(unseen_movies[i])]
        for i in top_indices
    ]
    top_scores = [predicted_scores[i] for i in top_indices]

    print(f"\n🎬 Top {top_k} recommendations for user {user_original_id}:\n")
    for title, score in zip(movies_df[movies_df["MovieID"].isin(top_movie_ids)]["Title"], top_scores):
        print(f"⭐ {title} — Predicted Rating: {score:.2f}")


# --------------------
# Main script
# --------------------
def main():
    # Rutas de los archivos
    users = pd.read_csv("ml-1m/users.dat", sep="::", engine="python",
                        names=["UserID", "Gender", "Age", "Occupation", "Zip-code"], encoding="latin-1")
    movies = pd.read_csv("ml-1m/movies.dat", sep="::", engine="python",
                         names=["MovieID", "Title", "Genres"], encoding="latin-1")
    ratings = pd.read_csv("ml-1m/ratings.dat", sep="::", engine="python",
                          names=["UserID", "MovieID", "Rating", "Timestamp"], encoding="latin-1")

    ratings["Rating"] = (ratings["Rating"] - 1.0) / 4.0
    movies["Genres"] = movies["Genres"].apply(lambda x: x.split("|"))

    user2idx = {uid: idx for idx, uid in enumerate(users["UserID"].unique())}
    movie2idx = {mid: idx for idx, mid in enumerate(movies["MovieID"].unique())}
    ratings["UserID"] = ratings["UserID"].map(user2idx)
    ratings["MovieID"] = ratings["MovieID"].map(movie2idx)

    num_users = len(user2idx)
    num_movies = len(movie2idx)

    # Modelo
    model = RecommenderNet(num_users, num_movies, embedding_dim=64)
    model.load_state_dict(torch.load("best_model.pth", map_location=torch.device("cpu")))
    model.eval()

    # Leer ID de usuario desde línea de comandos o usar uno fijo
    if len(sys.argv) > 1:
        try:
            user_id = int(sys.argv[1])
        except ValueError:
            print("Please provide a valid integer user ID.")
            return
    else:
        user_id = 75  # test

    recommend_movies(user_id, model, movies, ratings, user2idx, movie2idx)

if __name__ == "__main__":
    main()
