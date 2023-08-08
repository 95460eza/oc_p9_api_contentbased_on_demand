
import os
import glob
import pickle
import pandas as pd
import flask
from sklearn.metrics.pairwise import cosine_similarity



# API setup
app_flask = flask.Flask(__name__)
app_flask.config["DEBUG"] = True

# Create URL of Main Page:  http://127.0.0.1:5000 IF locally
@app_flask.route("/index", methods=["GET"])
def home():
    return "<h1>FLASK CONTENT-BASED PREDICTION API</h1><p>ENDPOINT for Predictions is under the /predict url route.</p>"


# Create URL of Predict Endpoint:http://127.0.0.1:5000/predict IF locally
@app_flask.route("/predict", methods=["POST"])
def predict_top_items():

    # Convert JSON received data from the request to Dictionary
    data_as_dict = flask.request.get_json()

    # Read-In Users Interactions Log files
    def read_in_user_intersactions(files_path):
        file_list = glob.glob(os.path.join(files_path, "*.csv"))

        list_of_dataframes = []

        for file in file_list:
            df = pd.read_csv(file)
            list_of_dataframes.append(df)

        df_combined = pd.concat(list_of_dataframes, ignore_index=True)

        return df_combined

    users_interactions_folder = "./clicks"
    df_users_interactions = read_in_user_intersactions(users_interactions_folder)

    # Function to Find last article clicked by a user
    def find_last_article_clicked(df, user_id):
        user_history = df[df["user_id"] == user_id]
        last_clicked_time = user_history["click_timestamp"].max()
        last_clicked_item_id = \
        user_history[user_history["click_timestamp"] == last_clicked_time]["click_article_id"].values[0]

        return last_clicked_item_id

    # Value Received from POST Request
    uid = data_as_dict["id"]
    user_last_clicked_item_id = find_last_article_clicked(df_users_interactions, uid)


    # Articles Embeddings Info
    def get_article_embeddings_vector(item_id):

        #Read embeddings matrix
        with open("./embeddings_matrix/articles_embeddings.pkl", "rb") as file:
            array_loaded = pickle.load(file)
        df_no_ids = pd.DataFrame(array_loaded)

        #Articles metadata
        df_articles_metadata = pd.read_csv("./articles_info/articles_metadata.csv")

        df_embeddings_with_full_metadata = pd.concat([df_articles_metadata, df_no_ids], axis=1)
        del df_articles_metadata, df_no_ids
        
        columns_to_drop = df_embeddings_with_full_metadata.columns[1:5]
        df_embeddings_with_ids = df_embeddings_with_full_metadata.drop(columns=columns_to_drop)
        del df_embeddings_with_full_metadata

        vector = df_embeddings_with_ids[df_embeddings_with_ids["article_id"]==item_id].drop(columns=["article_id"]).values

        return vector, array_loaded

    item_vector, embeddings_matrix = get_article_embeddings_vector(user_last_clicked_item_id)


    # Functions for COSINE Similarity
    def calculate_cosine_similarity(single_vector, matrix):

        # Reshape the single vector to (1, n) or (n,) if necessary
        single_vector = single_vector.reshape(1, -1) if len(single_vector.shape) == 1 else single_vector

        # Compute the cosine similarity using the cosine_similarity function
        similarity_scores = cosine_similarity(single_vector, matrix)

        return similarity_scores

    df_item_similarity_scores = pd.DataFrame(calculate_cosine_similarity(item_vector, embeddings_matrix)).drop(columns=[user_last_clicked_item_id])

    top_values_to_keep = 5
    melted_df = df_item_similarity_scores.melt(var_name="article_id", value_name="similarity_score")
    top_n_values = melted_df.nlargest(top_values_to_keep, "similarity_score")
    top_n_values.index = range(1, top_values_to_keep+1)
    print(top_n_values.shape)

    json_string = top_n_values.to_json()


    return flask.jsonify(response=json_string, message=user_last_clicked_item_id)
    # return flask.jsonify(response=1, message=shape_of_predicted)



#************Launch API LOCALLY*************
if __name__ == '__main__':
    app_flask.run(host='0.0.0.0', port=8000,  debug=True)