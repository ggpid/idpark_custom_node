import numpy as np
import torch
from scipy.ndimage import zoom
import torchvision.transforms.functional as TF

def resize_mask_centered(mask, new_width, new_height):
    """
    이미지 마스크의 크기를 조정합니다. 중앙에서부터 이미지를 잘라내거나 0으로 채웁니다.
    이를 통해 높이와 너비 모두 이미지의 중심에서 조정됩니다.

    Parameters:
    - mask (ndarray): 원본 이미지 마스크 (2차원 ndarray).
    - new_width (int): 조정된 이미지의 새 너비.
    - new_height (int): 조정된 이미지의 새 높이.

    Returns:
    - ndarray: 크기가 조정된 이미지 마스크.
    """
    current_height, current_width = mask.shape
    resized_mask = np.zeros((new_height, new_width), dtype=mask.dtype)

    # 시작점 계산 (원본 이미지 중앙에서 리사이징된 이미지의 중앙까지의 거리를 고려)
    start_y = max((current_height - new_height) // 2, 0)
    start_x = max((current_width - new_width) // 2, 0)

    # 종료점 계산
    end_y = start_y + min(current_height, new_height)
    end_x = start_x + min(current_width, new_width)

    # 리사이징된 이미지의 중앙에 원본 데이터를 삽입
    resized_mask_start_y = max((new_height - current_height) // 2, 0)
    resized_mask_start_x = max((new_width - current_width) // 2, 0)

    resized_mask[resized_mask_start_y:resized_mask_start_y+min(current_height, new_height),
    resized_mask_start_x:resized_mask_start_x+min(current_width, new_width)] = \
        mask[start_y:end_y, start_x:end_x]

    return resized_mask


def resize_mask(mask, new_width, new_height):
    """
    2차원 ndarray 이미지 마스크의 너비와 높이를 원하는 값으로 조정합니다.
    이미지의 원본 형태는 보존되어야 합니다.

    Parameters:
    - mask (ndarray): 원본 이미지 마스크 (2차원 ndarray).
    - new_width (int): 조정된 이미지의 새 너비.
    - new_height (int): 조정된 이미지의 새 높이.

    Returns:
    - ndarray: 조정된 이미지 마스크.
    """
    # 원본 이미지의 현재 너비와 높이를 계산
    current_height, current_width = mask.shape

    # 원하는 너비와 높이로 조정하기 위한 비율 계산
    width_ratio = new_width / current_width
    height_ratio = new_height / current_height

    # 'nearest' 보간법을 사용하여 이미지 리사이징
    resized_mask = zoom(mask, (height_ratio, width_ratio), order=0)

    return resized_mask

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

def tensor2mask(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if (len(size) < 4):
        return t
    if size[3] == 1:
        return t[:,:,:,0]
    elif size[3] == 4:
        # Not sure what the right thing to do here is. Going to try to be a little smart and use alpha unless all alpha is 1 in case we'll fallback to RGB behavior
        if torch.min(t[:, :, :, 3]).item() != 1.:
            return t[:,:,:,3]

    return TF.rgb_to_grayscale(tensor2rgb(t).permute(0,3,1,2), num_output_channels=1)[:,0,:,:]

def tensor2rgb(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if (len(size) < 4):
        return t.unsqueeze(3).repeat(1, 1, 1, 3)
    if size[3] == 1:
        return t.repeat(1, 1, 1, 3)
    elif size[3] == 4:
        return t[:, :, :, :3]
    else:
        return t

def tensor2rgba(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if (len(size) < 4):
        return t.unsqueeze(3).repeat(1, 1, 1, 4)
    elif size[3] == 1:
        return t.repeat(1, 1, 1, 4)
    elif size[3] == 3:
        alpha_tensor = torch.ones((size[0], size[1], size[2], 1))
        return torch.cat((t, alpha_tensor), dim=3)
    else:
        return t

def tensor2batch(t: torch.Tensor, bs: torch.Size) -> torch.Tensor:
    if len(t.size()) < len(bs):
        t = t.unsqueeze(3)
    if t.size()[0] < bs[0]:
        t.repeat(bs[0], 1, 1, 1)
    dim = bs[3]
    if dim == 1:
        return tensor2mask(t)
    elif dim == 3:
        return tensor2rgb(t)
    elif dim == 4:
        return tensor2rgba(t)

def tensors2common(t1: torch.Tensor, t2: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    t1s = t1.size()
    t2s = t2.size()
    if len(t1s) < len(t2s):
        t1 = t1.unsqueeze(3)
    elif len(t1s) > len(t2s):
        t2 = t2.unsqueeze(3)

    if len(t1.size()) == 3:
        if t1s[0] < t2s[0]:
            t1 = t1.repeat(t2s[0], 1, 1)
        elif t1s[0] > t2s[0]:
            t2 = t2.repeat(t1s[0], 1, 1)
    else:
        if t1s[0] < t2s[0]:
            t1 = t1.repeat(t2s[0], 1, 1, 1)
        elif t1s[0] > t2s[0]:
            t2 = t2.repeat(t1s[0], 1, 1, 1)

    t1s = t1.size()
    t2s = t2.size()
    if len(t1s) > 3 and t1s[3] < t2s[3]:
        return tensor2batch(t1, t2s), t2
    elif len(t1s) > 3 and t1s[3] > t2s[3]:
        return t1, tensor2batch(t2, t1s)
    else:
        return t1, t2
