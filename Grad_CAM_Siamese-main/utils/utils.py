import os
import pandas as pd
import datetime
from tqdm import tqdm

def get_dataset(params:dict=None)->pd.DataFrame:
    # 데이터 디렉토리를 스캔, 각 이미지 파일의 경로와 클래스를 담은 DataFrame을 만듦
    '''
        Import data

        Parameters
        ----------
        params: dict
            dictionary with parameters i.e paths, ...

        Returns
        -------
        DataFrame containing the files with the corresponding class
    '''


    # Get directory names ie classes
    directories_list = [f for f in os.listdir(params['data_path']) if os.path.isdir(os.path.join(params['data_path'], f))]

    # Parse all directory and get the contained files
    files, labels = [], []
    for directory in directories_list:
        # Get files
        path = os.path.join(params['data_path'], directory)
        files_list = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))] 

        files += [os.path.join(path, file) for file in files_list]
        labels += [ directory ] * len(files_list)

    # Store information in DataFrame
    df = pd.DataFrame({})
    df['Files'] = files
    df['Labels'] = labels

    return df


def create_instances(df:pd.DataFrame=None, number_of_iterations:int=None)->list:
    # Siamese 학습을 위해 이미지 쌍 생성하는 함수
    # positive와 negative를 균형있게 섞어서 생성
    '''
        Training instances (positive, negative) for training and evaluating
        Siamese network

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame containing the filenames along with their corresponsing class
        number_of_iterations: int
            Number of iteration for creating positive & negative instances


        Returns
        -------
        list with positive and negative examples
    '''
    # Create dataset for Siamese network
    # ---------------------------------------------------------
    # Each instance of the dataset has the form: [anchor image, positive/negative image, class]

    # Sanity check
    
    data = []
    for iterations in tqdm(range(number_of_iterations)): # number_of_iterations: 1이면 p/n 하나씩 총 2쌍 생성 (값이 크면 더 다양한 쌍을 여러 번 샘플링)
        for label in df['Labels'].unique(): # 모든 클래스에 대해 반복(각 클래스별로 돌아가면서 한 번씩 반복)
            sample_size = min(df[df['Labels'] == label].shape[0], df[df['Labels'] != label].shape[0])
            # min(df[class], df[not class])로 클래스별 sample 수 결정 -> 쌍 수가 많거나 적은 클래스로 인한 쏠림 현상 방지
            
            anchors = list( df[df['Labels'] == label].sample(sample_size)['Files'] )
            # 각 클래스에서 랜덤으로 anchor 이미지 선택
            positives = list( df[df['Labels'] == label].sample(sample_size)['Files'] )
            # anchor와 같은 클래스에서 하나 더 뽑아 positive (라벨 0)
            negatives = list( df[df['Labels'] != label].sample(sample_size)['Files'] )
            # anchor와 다른 클래스에서 하나 뽑아 negative (라벨 1)

            for anchor, positive, negative in zip(anchors, positives, negatives):
                data.append( [anchor, positive, 0.0] ) # (라벨 0)
                data.append( [anchor, negative, 1.0] ) # (라벨 1)

    return data




def format_time(elapsed: float):
    '''
        Function for presenting time in 
        
        Parameters
        ----------
        elapsed: time in seconds
        Returns
        -------
        elapsed as string
    '''
    return str(datetime.timedelta(seconds=int(round((elapsed)))))