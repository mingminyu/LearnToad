# coding: utf-8
import toad
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.utils import shuffle


def generate_train_data() -> pd.DataFrame:
    """生成鸢尾花二分类数据集"""
    data_iris = load_iris(as_frame=True)
    df_iris = data_iris['frame']
    df_iris["target"] = data_iris['target']
    # 只筛选 2 个类别的数据
    df_iris = df_iris.loc[df_iris['target'].isin((0, 1))]
    df_iris = shuffle(df_iris)
    return df_iris


if __name__ == '__main__':
    df = generate_train_data()
    # toad.detect(df)
    # toad.quality(df, iv_only=True)
    toad.selection.select()