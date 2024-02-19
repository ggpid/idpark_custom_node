import numpy as np


def run_length_encode(mask):
    """2차원 ndarray 마스크 데이터를 런랭스 압축하는 최적화된 함수"""
    # 마스크를 1차원 배열로 변환
    pixels = mask.flatten()
    # 배열의 시작에 가짜 변화를 추가하여 첫 번째 값도 처리하도록 함
    pixels_padded = np.pad(pixels, (1, 0), mode='constant', constant_values=not pixels[0])
    # 값이 변화하는 지점 찾기
    changes = np.diff(pixels_padded)
    change_indices = np.where(changes)[0]
    # 런 길이 계산
    lengths = np.diff(np.append(change_indices, pixels.size))

    # 첫 번째 값이 True인지 False인지에 따라 시작 값 조정
    first_value = bool(pixels[0])  # numpy.bool_를 Python의 bool로 변환
    lengths_list = lengths.tolist()  # numpy 배열을 Python 리스트로 변환
    height, width = int(mask.shape[0]), int(mask.shape[1])  # numpy 정수를 Python 정수로 변환

    return first_value, lengths_list, height, width


