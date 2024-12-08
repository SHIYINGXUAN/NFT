from flask import Flask, jsonify, request
from flask_cors import CORS

from DataCluster import DataCluster
# Import the DataLoader class
from DataLoader import DataLoader

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
loader = DataLoader()

@app.route('/test', methods=['GET'])
def test_server():
    return "hello"

@app.route('/correlation_matrix', methods=['GET'])
def get_correlation_matrix():
    """
    API endpoint to fetch the correlation matrix.

    Query Parameters:
    - sub_ind (optional): Index of the subspace. If not provided, computes for the full dataset.

    Response:
    - JSON object containing the correlation matrix.
    """
    sub_ind = request.args.get("sub_ind", type=int)
    try:
        corr_matrix = loader.corr_matrix(sub_ind)
        return jsonify({"correlation_matrix": corr_matrix.to_dict()})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route('/create_subspace', methods=['POST'])
def create_subspace():
    """
    API endpoint to create a subspace with filtered features.

    Request Body (JSON)
    ------------
    - features: List of feature names to filter on.
    - ranges:
        for quantitative attributes: List of [min, max] ranges corresponding to the features.
        for categorical attributes: list of chosen values

    Response
    -------------
    - JSON object containing the index of the created subspace.
    """
    data = request.get_json()
    features = data.get("features")
    ranges = data.get("ranges")

    try:
        subspace_index = loader.create_subspace(features, ranges)
        return jsonify({"subspace_index": subspace_index})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route('/feature_ranges', methods=['GET'])
def get_feature_ranges():
    """
    API endpoint to fetch feature ranges.
    ---------------
    Query Parameters:
    -------------
    - sub_ind (optional): Index of the subspace. If not provided, fetches for the full dataset.

    Response:
    -------------
    - JSON object containing feature ranges.
    """
    sub_ind = request.args.get("sub_ind", type=int)
    try:
        feature_ranges = loader.get_feature_ranges(sub_ind)
        return jsonify({"feature_ranges": feature_ranges})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route('/fetch_data_with_features', methods=['GET'])
def fetch_data_with_features():
    """
    API endpoint to fetch data with features.

    Query Parameters
    -----------
    - sub_ind (optional): Index of the subspace. If not provided, fetches for the full dataset.
    - features (optional): List of feature names to filter on. If not provided, fetches for the full subspace/dataset

    Response
    -----------
    :returns: JSON object containing the data with features and their corresponding ranges.
    """
    features_lst = request.args.get("features", type=str)
    sub_ind = request.args.get("sub_ind", type=int)
    if features_lst is not None:
        features_lst = features_lst.split(",")

    try:
        fetched_data = loader.fetch_data_with_features(sub_ind, features_lst).to_dict(orient="list")
        return jsonify({"data": fetched_data})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route('/hierarchy_cluster', methods=['GET'])
def hierarchy_cluster():
    """
    API endpoint to build hierarchy tree from selected subspace

    Query Parameters:
    ---------------
    - sub_ind (optional): Index of the subspace. If not provided, cluster for the full dataset.

    Response:
    -----------
    :returns: JSON object containing the hierarchy tree. Refer to the tree for more details
    """

    sub_ind = request.args.get("sub_ind", type=int)

    try:

        sub_ind_df = loader.fetch_data_with_features(sub_ind)
        cluster = DataCluster(sub_ind_df)
        cluster_tree = cluster.build_iterative_tree()
        return jsonify(cluster_tree)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route('/distribution_by_feature', methods=['GET'])
def distribution_by_feature():
    sub_ind = request.args.get("sub_ind", type=int)
    feature = request.args.get("feature", type=str)
    by_label = request.args.get("by_label", type=bool)
    try:
        distribution = loader.distribution_by_feature(feature, sub_ind, by_label)
        return jsonify(distribution)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route('/update_subspace', methods=['POST'])
def update_subspace():
    data = request.get_json()
    sub_ind = data.get("sub_ind", type=int)
    features = data.get("features")
    ranges = data.get("ranges")

    try:
        res = loader.update_subspace(features, ranges, sub_ind)
        return jsonify(res)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route('/dimension_reduce', methods=['GET'])
def dimension_reduce():
    sub_ind = request.args.get("sub_ind", type=int)
    n_components = request.args.get("n_components", type=int)
    try:
        reduced_df = loader.dimension_reduce(sub_ind, n_components).to_dict(orient="list")
        return jsonify(reduced_df)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route('/get_feature_metric', methods=['GET'])
def get_feature_metric():
    sub_ind = request.args.get("sub_ind", type=int)
    metric = request.args.get("metric", type=str)

    try:
        result = loader.get_feature_metric(sub_ind, metric)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
