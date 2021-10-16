import pandas as pd
import numpy as np
import traceback
from flask import Flask, Response
from recomendations import als_matrix, user_based, item_based

# init parametres
user_item_df = pd.DataFrame({"userid"   : [ 0, 0, 1, 1, 2],
                             "item_id" : [ 0, 3, 0, 1, 3]})

user_item   = np.array( [[1,0,0,1],
                         [1,1,0,0],
                         [0,0,0,1]] )

rec_matrix = als_matrix( user_item, 2, 400, 1)

def create_app( user_item_df, user_item, rec_matrix):
    """Create flask server

    Args:
        user_item_df ([pandas.DataFrame]): [interaction data frame for user-item, columns=(user_id, item_id)]
        user_item ([numpy.array int, float]): [interaction matrix for user-item]
        rec_matrix ([numpy.array int, float]): [estimated user_item matrix]

    Returns:
        [app]: [flask application]
    """

    app = Flask(__name__)

    @app.route('/recomenadation/user_based/<user_id>')
    def recomenadation_user_based(user_id):
        """Recomend items for some user

        Args:
            user_id ([int]): [user id]

        Returns:
            [array int]: [ids of items]
            [error string]: [if something went wrong, but it was expected]
            [0 string]: [if something went wrong, but it was not expected]
        """
        try:
            user_id = int(user_id)
            return str( user_based( rec_matrix, user_item, user_id ) )
        except AssertionError as ae:
            # if index + 1 not in df.userid.value_counts().index:
            #     return {"sort_ids": [int(item) for item in df.item_id.value_counts().index[:3]]}
            return str(ae)
        except Exception as e:
            traceback.print_exc()
            return "0"


    @app.route('/recomendation/item_based/<user_id>/<item_id>')
    def recommendation_item_based(user_id, item_id):
        """Recommend items for the selected item

        Args:
            user_id ([int]): [user id]
            item_id ([int]): [the id of the item he selected]

        Returns:
            [array int]: [ids of items]
            [error string]: [if something went wrong, but it was expected]
            [0 string]: [if something went wrong, but it was not expected]
        """
        try:
            user_id = int(user_id)
            item_id = int(item_id)
            return str( item_based( user_item, user_id, item_id) )
        except AssertionError as ae:
            return str(ae)
        except Exception as e:
            traceback.print_exc()
            return "0"

    return app

if __name__ == "__main__":
    app = create_app( user_item_df, user_item, rec_matrix)
    app.run()
