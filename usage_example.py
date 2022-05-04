import torch
from scores import get_cmc_k_score, get_map_score

if __name__ == '__main__':

    labels = torch.IntTensor([1, 4])
    g_labels = torch.IntTensor([1, 2, 3, 4, 5])
    predictions = torch.IntTensor([[2, 4, 1, 5, 3], [1, 3, 2, 5, 4]])

    cmc_1 = get_cmc_k_score(labels, predictions, k=1)
    cmc_3 = get_cmc_k_score(labels, predictions, k=3)
    cmc_5 = get_cmc_k_score(labels, predictions, k=5)
    mAP = get_map_score(labels, predictions, g_labels)

    print(cmc_1, cmc_3, cmc_5, mAP)
