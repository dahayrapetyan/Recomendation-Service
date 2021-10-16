import numpy as np
import pytest

def user_based( rec_matrix, initial_matrix, user_id, rec_cnt=3 ):
    """Return several index of items, based on similar users 

    Args:
        rec_matrix ([np.array]): [estimated matrix]
        initial_matrix ([np.array]): [initial matrix user-item]
        user_id ([int]): [id of the user we recommended]
        rec_cnt (int, optional): [count of recommdations]. Defaults to 3.

    Returns:
        [array int]: [indexs of recommdations]

    Examples:
        >>> import numpy as np 
        >>> user_item   = np.array( [[1,0,0,1], [1,1,0,0], [0,0,0,1]] )
        >>> rec_matrix = np.array( [[0.98981623, 0.74652651, 0.25153983, 1.00197617], [0.99896439, 0.98994769, 0.32799586, 0.86552836], [0.90507976, 0.56661684, 0.19364865, 0.98766064]])
        >>> user_based( rec_matrix, user_item, 1, 2 )
        [3, 2]
    """

    index = int(user_id)
    assert index < initial_matrix.shape[0], "user_id out of bounds"

    # оставляем объекты которые не оценены
    index_null = np.where(initial_matrix[index] == 0)
    sort_index = np.argsort(rec_matrix[index])[::-1]

    del_index = np.isin(sort_index, index_null)

    sort_index = sort_index[del_index]
    indexs = [int(item) for item in sort_index[:rec_cnt]]

    return indexs

@pytest.mark.parametrize(
    ("rec_matrix", "initial_matrix", "user_id", "expected"), 
    [
        ( np.array( [[1,1,0,1], [1,1,0,1], [1,1,0,1]]), np.array( [[1,0,0,1], [1,1,0,0], [0,0,0,1]]), 1, [3, 2] ),
        ( np.array( [[1,0.8,1], [1,1,0.8], [0.8,1,0.5]]), np.array( [[1,0,1], [1,1,0], [0,1,0]]), 2, [0, 2]),
    ]
)
def test_user_based( rec_matrix, initial_matrix, user_id, expected):
    assert user_based( rec_matrix, initial_matrix, user_id ) == expected
