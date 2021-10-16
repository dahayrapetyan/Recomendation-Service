import numpy as np
import pytest

def item_based( matrix, user_id, item_id, rec_cnt=3):
    """Return several index of items, based on similar items and items frequency. If user_1 buys item_1, and we know that user_2 bought item_1 and also item_2, 
    then we can recommend item_2 for user_1

    Args:
        matrix ([np.array]): [interaction matrix for user-item]
        user_id ([int]): [the id of the user who choose the item]
        item_id ([int]): [the id of the item selected by the user]

    Returns:
        [array int]: [ids of items for recommendation]

    Examples:
        >>> import numpy as np
        >>> matrix = np.array( [[1,0,0,0], [0,1,1,0], [1,1,1,1]] )
        >>> item_based( matrix, 0, 3 )
        [1, 2]
    """

    assert user_id < matrix.shape[0], "user_id out of bounds"
    assert item_id < matrix.shape[1], "item_id out of bounds"

    sh = matrix.shape[1]
    time_weght_matrix = np.ones((sh, sh))

    # index строк где есть этот обьект ( индекс людей котррый уже купили этот товар )
    not_null_values = np.where(matrix[:, item_id] != 0)
    # товры которые пользователь уже купил, и текущий элемент
    user_not_null_index = np.where(matrix[user_id] != 0)
    user_not_null_index = np.append( user_not_null_index, item_id )

    # сколько раз встречаются соседи у выбранног эллемента
    element_neighbours = np.sum(matrix[not_null_values], axis=0)
    near_element_args = np.argsort(element_neighbours)[::-1]

    # удалем элементы которые уже оценены
    del_index = np.isin(near_element_args, user_not_null_index, invert=True)
    near_element_args = near_element_args[del_index]

    # умножает рекомендации близких объектов к выбранному из матрица, на разность времени между выбранным и рекомендациями
    near_element_args_dist = np.array( [element_neighbours[arg] * time_weght_matrix[item_id, arg] for arg in near_element_args])

    # берем модуль, сортируем 
    near_element_args_dist = np.absolute(near_element_args_dist)
    near_element_args_dist_sort = np.argsort( near_element_args_dist )[::-1]

    # берем элементы по убыванию коэффициента
    near_element_args_dist_index = near_element_args[near_element_args_dist_sort]

    indexs = [int(item) for item in near_element_args_dist_index[:rec_cnt]]


    return indexs

@pytest.mark.parametrize(
    ("matrix", "user_id", "item_id", "expected"),
    [
        (np.array( [[1,0,0,0], [0,1,1,0], [1,1,1,1]] ), 0, 3, [1, 2]),
        (np.array( [[1, 1],[0, 0]] ), 1, 0, [1])
    ]
)
def test_item_based( matrix, user_id, item_id, expected ):
    assert item_based( matrix, user_id, item_id ) == expected