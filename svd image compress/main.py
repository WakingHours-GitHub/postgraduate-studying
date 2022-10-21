import time
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


def calculate_svd(A: np.ndarray):
    print(A)
    A_t = A.T

    eigenvalue, U = np.linalg.eig(np.matmul(A, A_t))

    eigenvalue, V = np.linalg.eig(np.matmul(A_t, A))

    eigenvalue = abs(eigenvalue) ** 0.5
    print("U", U)

    print("V", V)
    print("eigenvalue", eigenvalue)





def get_signal_matrix(signal_value: np.ndarray, k: int) -> np.ndarray:
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

    # calculate_svd(A)

    print(get_signal_matrix(signal_value, len(signal_value)))
    # 将奇异值处理成矩阵, 测试速度。 :
    # start_time = time.time()
    # for i in range(1000):
    #     get_signal_matrix(signal_value)
    # endtime = time.time()

    # print(endtime-start_time)

    print("===="*100)



    # print(A @ A.T)



def main() -> None:

    plt.figure(0, figsize=(16, 9), dpi=600)
    K = 256



    img = cv.imread("./lenna.jpg")
    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # print(img)
    h, w, c = img.shape

    # plt.subplot(2, 4, 1)
    # plt.imshow(img[:, :, ::-1]) # plt读取是RGB
    # plt.title("source img")

    for idx, k in enumerate(range(1, 9, 1)):
        k = k * 4
        svd_zip_img = np.zeros_like(img, dtype=np.float32)

        for channel in range(c):
            # calculate_svd(img[:, :, 0])
            U, singular_value, V_T = np.linalg.svd(img[:, :, channel])
            sigma_matrix = get_signal_matrix(singular_value, k)
            svd_zip_img[:, :, channel] = U[:, :k] @ sigma_matrix @ V_T[:k, :]

            # break

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

    plt.savefig("./svd_subtle_contrast_image.png")
    plt.show()
        # cv.imshow("img", np.hstack([img, zip_img]))
        #
        # cv.imshow("B", svd_zip_img)
        # cv.waitKey(0)





if __name__ == '__main__':
    main()
    # test()