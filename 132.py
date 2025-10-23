import socket
import numpy as np
import cv2
import time

UDP_IP = "0.0.0.0"
UDP_PORT = 12345
output_file = "received_img.raw"

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65535)
sock.bind((UDP_IP, UDP_PORT))

print(f"UDP服务器正在监听 {UDP_PORT} 端口...")

chunks = {}
expected_chunks = None

# 统计FPS
frame_count = 0
fps = 0.0
start_time = time.time()

while True:
    data, addr = sock.recvfrom(65535)  # 接收最大UDP包
    if not data:
        continue

    header = data[:8]
    seq_num = int.from_bytes(header[:4], byteorder='big')
    total_chunks = int.from_bytes(header[4:], byteorder='big')
    payload = data[8:]

    if expected_chunks is None:
        expected_chunks = total_chunks

    chunks[seq_num] = payload

    if len(chunks) == expected_chunks:
        # 拼接数据
        with open(output_file, "wb") as f:
            for i in range(expected_chunks):
                f.write(chunks[i])
        print(f"文件接收完毕，已保存为：{output_file}")

        # 读取文件并解码为16bit图像
        with open(output_file, "rb") as f:
            raw_data = f.read()

        # 确定尺寸
        height = 1280
        width = 1088

        expected_size = height * width * 2  # 2字节/像素

        if len(raw_data) != expected_size:
            print(f"警告：文件大小 {len(raw_data)} bytes 不等于期望大小 {expected_size} bytes")
        else:
            # 统计帧率
            frame_count += 1
            curr_time = time.time()
            elapsed_time = curr_time - start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = curr_time

            # 转换为16位图像
            img = np.frombuffer(raw_data, dtype=np.uint16)
            img = img.reshape((height, width))

            # 归一化
            img = img.astype(np.float32) / 1024.0


            # 显示帧率
            text = f"FPS: {fps:.2f}"
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2, cv2.LINE_AA)

            # 显示
            cv2.imshow("Received Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 重置，准备接收下一张图
        chunks.clear()
        expected_chunks = None

cv2.destroyAllWindows()
