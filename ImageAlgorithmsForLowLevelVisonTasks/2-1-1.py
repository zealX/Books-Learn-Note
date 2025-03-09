import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def build_gabor_filter(ksize=31, sigma=4., theta=np.pi/4, lambd=10., gamma=0.5, psi=0):
    gabor = cv2.getGaborKernel(
        (ksize, ksize),
        sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F
    )
    return gabor


def easy_gabor_filter(img_path: str):
    # Gabor参数
    ksize = 31  # 滤波器尺寸
    sigma = 4.0  # 高斯包络标准差
    theta = np.pi / 4  # 方向
    lambd = 10.0  # 波长（lambda）
    gamma = 0.5  # 长宽比
    psi = 0  # 相位

    kernel = build_gabor_filter(ksize, sigma, theta, lambd, gamma, psi)

    # visual kernel
    plt.imshow(kernel, cmap='gray')
    plt.title('Gabor Kernel')
    plt.axis('off')
    plt.show()

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # apply
    filtered_img = cv2.filter2D(img, cv2.CV_8UC3, kernel)

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Gabor filtered')
    plt.imshow(filtered_img, cmap='gray')
    plt.axis('off')

    plt.show()


def gabor_filter_application(img_path: str, mode='sketch'):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if mode == 'sketch':
        ksize = 21  # 滤波器尺寸
        sigma = 3.0  # 高斯包络标准差
        theta = 45 * np.pi / 180  # 方向
        lambd = 10.0  # 波长（lambda）
        gamma = 0.5  # 长宽比
        psi = 90 * np.pi / 180  # 相位
        kernel = build_gabor_filter(ksize, sigma, theta, lambd, gamma, psi)
        filtered_img = cv2.filter2D(img, cv2.CV_8UC3, kernel)
    else:
        # 浮雕效果
        ksize = 15  # 滤波器尺寸
        sigma = 5.0  # 高斯包络标准差
        theta = 0  # 方向
        lambd = 10.0  # 波长（lambda）
        gamma = 0.8  # 长宽比
        psi = np.pi / 2  # 相位
        kernel = build_gabor_filter(ksize, sigma, theta, lambd, gamma, psi)
        filtered_img = cv2.filter2D(img, cv2.CV_8UC3, kernel)
        filtered_img = cv2.addWeighted(img, 0.5, filtered_img, 0.5, 0)

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Gabor filtered')
    plt.imshow(filtered_img, cmap='gray')
    plt.axis('off')

    plt.show()


def gabor_sketch(img_path):
    # 转换为灰度图
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # 创建多方向Gabor滤波器组
    kernels = []
    for theta in np.arange(0, np.pi, np.pi / 4):  # 4个方向
        kernel = cv2.getGaborKernel((21, 21), sigma=3, theta=theta,
                                    lambd=10, gamma=0.5, psi=np.pi / 2)
        kernels.append(kernel)

    # 多方向响应叠加
    combined = np.zeros_like(gray, dtype=np.float32)
    for k in kernels:
        filtered = cv2.filter2D(gray, cv2.CV_32F, k)
        combined += np.abs(filtered)

    # 归一化并反相
    combined = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX)
    sketch = 255 - combined.astype(np.uint8)

    # 三通道化以保持彩色输出
    res = cv2.merge([sketch, sketch, sketch])
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(gray, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Gabor filtered')
    plt.imshow(sketch, cmap='gray')
    plt.axis('off')

    plt.show()


if __name__ == '__main__':
    gabor_filter_application('fbb.png', 'HAHA')
