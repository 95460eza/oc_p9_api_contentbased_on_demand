
import os
import glob
import pandas as pd
import flask



# API setup
app_flask = flask.Flask(__name__)
app_flask.config["DEBUG"] = True

# Create URL of Main Page:  http://127.0.0.1:5000 IF locally
@app_flask.route("/", methods=["GET"])
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

    uid = data_as_dict["id"]
    user_last_clicked_item_id = find_last_article_clicked(df_users_interactions, uid)
    print(type(user_last_clicked_item_id), user_last_clicked_item_id)



    return flask.jsonify(response=1)
    # return flask.jsonify(response=1, message=shape_of_predicted)



#************Launch API LOCALLY*************
if __name__ == '__main__':
    app_flask.run(host='0.0.0.0', port=8000,  debug=True)