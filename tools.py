import numpy as np


def run_length_encode(mask):
    """ 2차원 ndarray 마스크 데이터를 런랭스 압축하는 함수 """
    # 마스크를 1차원 배열로 변환
    pixels = mask.flatten()

    # 첫 번째 값, 압축 결과, 그리고 마스크의 차원을 저장
    first_value = pixels[0]
    lengths = []
    height, width = mask.shape

    # 현재 실행 중인 값 및 길이를 추적
    current_run_length = 1

    for pixel in pixels[1:]:
        if pixel == first_value:
            # 현재 값과 동일한 경우, 런 길이 증가
            current_run_length += 1
        else:
            # 현재 실행 길이를 기록하고 새 실행 시작
            lengths.append(current_run_length)
            first_value = not first_value  # 값 토글
            current_run_length = 1

    # 마지막 실행 길이 기록
    lengths.append(current_run_length)

    return first_value, lengths, height, width


def run_length_decode(first_value, lengths, height, width):
    """ 런랭스 인코딩된 데이터를 2차원 배열로 복원 """
    # 모든 값이 False인 2차원 배열 생성
    decoded_image = np.zeros(height * width, dtype=np.bool_)

    current_position = 0
    current_value = first_value
    for length in lengths:
        end_position = current_position + length
        decoded_image[current_position:end_position] = current_value
        current_position = end_position
        current_value = not current_value  # 값 토글

    return decoded_image.reshape(height, width)


