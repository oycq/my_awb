# cam_client.py
import socket
import numpy as np
import threading
import queue

class VideoCapture:
    def __init__(self, udp_port=12345, height=1280, width=1088):
        self.udp_ip = "0.0.0.0"
        self.udp_port = udp_port
        self.height = height
        self.width = width
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65535 * 4)  # 增大接收缓冲区，减少丢包
        self.sock.bind((self.udp_ip, self.udp_port))
        self.queue = queue.Queue(maxsize=1)  # 限制队列大小为1，只保留最新帧，减少延迟
        self.chunks = {}
        self.expected_chunks = None
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()

    def _receive_loop(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(65535)
                if not data:
                    continue
                header = data[:8]
                seq_num = int.from_bytes(header[:4], byteorder='big')
                total_chunks = int.from_bytes(header[4:], byteorder='big')
                payload = data[8:]

                # 新增: 当收到 seq_num == 0 时，重置残留状态（假设新帧从0开始）
                if seq_num == 0 and self.expected_chunks is not None:
                    self.chunks.clear()
                    self.expected_chunks = None

                if self.expected_chunks is None:
                    self.expected_chunks = total_chunks
                self.chunks[seq_num] = payload
                if len(self.chunks) == self.expected_chunks:
                    expected_size = self.height * self.width * 2
                    # 预分配 bytearray 优化拼接效率
                    raw_data = bytearray(expected_size)
                    offset = 0
                    for i in range(self.expected_chunks):
                        chunk = self.chunks[i]
                        chunk_len = len(chunk)
                        raw_data[offset:offset + chunk_len] = chunk
                        offset += chunk_len
                    if offset != expected_size:
                        print(f"警告：数据大小 {offset} bytes 不等于期望大小 {expected_size} bytes")
                    else:
                        # 转换为16位图像
                        img = np.frombuffer(raw_data, dtype=np.uint16).reshape((self.height, self.width))
                        # 归一化到0-1 float32
                        img = img.astype(np.float32) / 1024.0
                        # 如果队列满，丢弃旧帧，只放最新
                        if not self.queue.empty():
                            try:
                                self.queue.get_nowait()
                            except queue.Empty:
                                pass
                        self.queue.put(img)
                    # 重置
                    self.chunks.clear()
                    self.expected_chunks = None
            except Exception as e:
                if self.running:
                    print(f"接收错误: {e}")

    def read(self):
        # 阻塞式读取
        return self.queue.get()

    def release(self):
        self.running = False
        self.sock.close()