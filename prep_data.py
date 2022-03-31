import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def main():
    # Load a standard machine learning dataset
    cancer = load_breast_cancer()

    df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
    df['target'] = cancer['target']

    # Optionally write out a subset of the data, used in this tutorial for inference with the API
    train, test = train_test_split(df, test_size=0.2)
    del test['target']
    test.to_csv('data/test.csv', index=False)
    train.to_csv('data/train.csv', index=False)
    print(train)


if __name__ == "__main__":
    main()