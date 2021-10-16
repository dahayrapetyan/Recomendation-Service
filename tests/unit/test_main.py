import os
import sys
import pandas as pd
import numpy as np

file_name = str( os.path.basename(__file__) )
file_name = file_name.replace("_test", '')

src_dir = str( os.getcwd() )
src_dir = src_dir.replace( r'tests\unit', "src" )
# src_dir = os.path.join( src_dir, file_name )
print( src_dir )
sys.path.insert( 1, src_dir)

from main import create_app
import pytest

@pytest.fixture(scope="module")
def client():
    user_item_df = pd.DataFrame({"userid"   : [ 0, 0, 1, 1, 2],
                                 "courseid" : [ 0, 3, 0, 1, 3]})

    user_item   = np.array( [[1,0,0,1],
                             [1,1,0,0],
                             [0,0,0,1]] )

    rec_matrix = np.array( [[1  ,0.5,0,1],
                            [1  ,1,0,0.5],
                            [0.5,0.3,0,1]] )

    app = create_app( user_item_df, user_item, rec_matrix)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

@pytest.mark.parametrize(
    ("user_id", "expected"),
    [
        ( 5,   "user_id out of bounds"),
        ( "a", "0")
    ]
)
def test_recomenadation_user_based( user_id, expected, client):
    req = client.get('/recomenadation/user_based/'+ str( user_id ) )
    assert req.data.decode("utf-8")  == expected 



@pytest.mark.parametrize(
    ("user_id", "item_id", "expected"),
    [
        ( 5, 1, "user_id out of bounds"),
        ( 1, 5, "item_id out of bounds"),
        ("a", 5, "0"),
        ( 1, "a", "0")
    ]
)
def test_recommendation_item_based( user_id, item_id, expected, client):

    req = client.get("/recomendation/item_based/{}/{}".format(user_id, item_id))
    assert req.data.decode("utf-8") == expected