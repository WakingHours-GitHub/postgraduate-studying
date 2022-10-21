import time
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


def calculate_svd(A: np.ndarray):
    m, n = A.shape
    eigenvalue_v, V = np.linalg.eig(A.T @ A)
    sorted_id = sorted(range(len(eigenvalue_v)), key=lambda k: eigenvalue_v[k], reverse=True)
    eigenvalue_v = np.sqrt(eigenvalue_v)
    V[:] = V[:, sorted_id]

    eigenvalue_u, U = np.linalg.eig(A @ A.T)
    sorted_id = sorted(range(len(eigenvalue_u)), key=lambda k: eigenvalue_u[k], reverse=True)
    eigenvalue_u = np.sqrt(eigenvalue_u)

    U[:] = U[:, sorted_id]
    if m > n:
        return U, eigenvalue_v.reshape((-1, 1)), V.T
    else:
        return U, eigenvalue_u.reshape((-1, 1)), V.T

    # if m > n:
    #     sigma, V = np.linalg.eig(A.T @ A)
    #     # 将sigma 和V 按照特征值从大到小排列
    #     arg_sort = np.argsort(sigma)[::-1]
    #     sigma = np.sort(sigma)[::-1]
    #     V = V[:, arg_sort]
    #
    #     # 对sigma进行平方根处理
    #     sigma_matrix = np.diag(np.sqrt(sigma))
    #
    #     sigma_inv = np.linalg.inv(sigma_matrix)
    #
    #     U = A @ V.T @ sigma_inv
    #     U = np.pad(U, pad_width=((0, 0), (0, m - n)))
    #     sigma_matrix = np.pad(sigma_matrix, pad_width=((0, m - n), (0, 0)))
    #     return (U, sigma_matrix, V)
    # else:
    #     # 同m>n 只不过换成从U开始计算
    #     sigma, U = np.linalg.eig(A @ A.T)
    #     arg_sort = np.argsort(sigma)[::-1]
    #     sigma = np.sort(sigma)[::-1]
    #     U = U[:, arg_sort]
    #
    #     sigma_matrix = np.diag(np.sqrt(sigma))
    #     sigma_inv = np.linalg.inv(sigma_matrix)
    #     V = sigma_inv @ U.T @ A
    #     V = np.pad(V, pad_width=((0, n - m), (0, 0)))
    #
    #     sigma_matrix = np.pad(sigma_matrix, pad_width=((0, 0), (0, n - m)))
    #     return (U, sigma_matrix, V)


    # A_t = A.T
    #
    # eigenvalue, U = np.linalg.eig(np.matmul(A, A_t))
    # sorted_id = sorted(range(len(eigenvalue)), key=lambda k: eigenvalue[k], reverse=True)
    # # 将range()序列, 根据eigenvalue的值, 进行排序, 得到的就是index
    # U[:] = U[:, sorted_id]
    # # print("fix", U)
    # eigenvalue, V = np.linalg.eig(np.matmul(A_t, A))
    # sorted_id = sorted(range(len(eigenvalue)), key=lambda k: eigenvalue[k], reverse=True)
    # V[:] = V[:, sorted_id]
    # # print("fix", V)
    # eigenvalue = abs(eigenvalue) ** 0.5
    #
    # singular_value = sorted(eigenvalue, reverse=True)
    # print(singular_value)
    #
    # return U, np.array(singular_value), eigenvalue




def get_sigma_matrix(signal_value: np.ndarray, k: int) -> np.ndarray:
    assert k <= signal_value.shape[0]
    # print(signal_value.shape) # (len, )
    # n = signal_value.shape[0]
    # result = np.zeros((n, n), signal_value.dtype)
    # result[:n-k].flat[0::n + 1] = signal_value # 理解np.diag()源码. 本质上还是索引
    # 只不过将result view 1D vector, 然后利用切片, 进行插值而已.
    # np.diag()
    # return result

    # 另一种方法:
    return np.eye(k) * signal_value[:k] # broadcast and multiply by each corresponding element


def test() -> None:
    A = np.array([
        [2, 4],
        [1, 3],
        [0, 0],
        [0, 0]
    ])

    U, signal_value, V = np.linalg.svd(A)
    print(U)
    print(signal_value)
    print(V)
    print("===="*100)

    calculate_svd(A)
    #
    # print(get_signal_matrix(signal_value, len(signal_value)))
    # 将奇异值处理成矩阵, 测试速度。 :
    # start_time = time.time()
    # for i in range(1000):
    #     get_signal_matrix(signal_value)
    # endtime = time.time()

    # print(endtime-start_time)

    print("===="*100)



    # print(A @ A.T)



def main() -> None:

    # plt.figure(0, figsize=(16, 9), dpi=300)
    K = 256



    img = cv.imread("./lenna.jpg")
    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # print(img)
    h, w, c = img.shape

    # plt.subplot(2, 4, 1)
    # plt.imshow(img[:, :, ::-1]) # plt读取是RGB
    # plt.title("source img")
    size_list = list()
    size_list.append(0)
    K_list = list(range(0, K+1, 1))
    zip_rate_list = list((120, ))
    # print(zip_rate_list)

    for idx, k in enumerate(range(1, K+1, 1)):
        # k = k * 4
        svd_zip_img = np.zeros_like(img, dtype=np.float32)
        size_img = 0
        for channel in range(c):
            # U, singular_value, V_T = calculate_svd(img[:, :, channel])
            # print(singular_value)
            U, singular_value, V_T = np.linalg.svd(img[:, :, channel])
            # print(singular_value)
            sigma_matrix = get_sigma_matrix(singular_value, k)
            svd_zip_img[:, :, channel] = U[:, :k] @ sigma_matrix @ V_T[:k, :]

            size = U.shape[0] * k + sigma_matrix.shape[0] * sigma_matrix.shape[1] + k * V_T.shape[1]
            size_img += size

        size_list.append(size_img)
        zip_rate_list.append(img.size / size_img)

    plt.plot(K_list, size_list)
    plt.title("Image size in different singular value")
    plt.show()


    plt.plot(K_list, zip_rate_list)
    plt.title("Image compress rate in different singular value")

    plt.show()

"""
        # 归一化, 平均色调。
        for i in range(c):
            MAX = np.max(svd_zip_img[:, :, i])
            MIN = np.min(svd_zip_img[:, :, i])
            svd_zip_img[:, :, i] = (svd_zip_img[:, :, i] - MIN) / (MAX - MIN)

        zip_img = np.round(svd_zip_img * 255).astype("uint8")

        plt.subplot(2, 4, idx+1)
        plt.title(f"num of singular: {k}")
        # print(222 + idx)
        plt.imshow(zip_img[:, :, ::-1])

    # plt.savefig("./svd_subtle_contrast_image.png")
    plt.show()
"""
        # cv.imshow("img", np.hstack([img, zip_img]))
        #
        # cv.imshow("B", svd_zip_img)
        # cv.waitKey(0)





if __name__ == '__main__':
    main()
    # test()