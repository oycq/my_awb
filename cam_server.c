#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/time.h>

#define DEST_IP "192.168.0.100"
#define DEST_PORT 12345
#define CHUNK_SIZE 1400
#define EXPECTED_SIZE 1280*1088*2
#define SENSOR_FILE "sensor2.raw"

int main() {
    int sockfd;
    struct sockaddr_in dest_addr;

    // 创建UDP套接字
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket");
        exit(EXIT_FAILURE);
    }

    memset(&dest_addr, 0, sizeof(dest_addr));
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(DEST_PORT);
    dest_addr.sin_addr.s_addr = inet_addr(DEST_IP);

    while (1) {
        FILE *file = fopen("/tmp/" SENSOR_FILE, "rb");
        if (!file) {
            usleep(10000);  // 调到10ms，避免太频繁检查
            continue;
        }

        // 检查文件大小
        if (fseek(file, 0, SEEK_END) != 0) {
            fclose(file);
            usleep(10000);
            continue;
        }

        long file_size = ftell(file);
        if (file_size != EXPECTED_SIZE) {
            fclose(file);
            usleep(10000);
            continue;
        }

        rewind(file);

        uint32_t total_chunks = (uint32_t)((file_size + CHUNK_SIZE - 1) / CHUNK_SIZE);

        // 读取整个文件到内存
        uint8_t *file_data = (uint8_t *)malloc(file_size);
        if (!file_data) {
            fclose(file);
            usleep(10000);
            continue;
        }

        size_t read_bytes = fread(file_data, 1, file_size, file);
        fclose(file);

        if (read_bytes != file_size) {
            free(file_data);
            usleep(10000);
            continue;
        }

        // 发送整幅图像（拆分成chunk发送）
        struct timeval start_time, end_time;
        gettimeofday(&start_time, NULL);

        for (uint32_t seq_num = 0; seq_num < total_chunks; seq_num++) {
            size_t offset = seq_num * CHUNK_SIZE;
            size_t chunk_size = CHUNK_SIZE;
            if (offset + chunk_size > file_size) {
                chunk_size = file_size - offset;
            }

            uint8_t send_buffer[CHUNK_SIZE + 8];  // 够大

            // 4字节序号
            send_buffer[0] = (seq_num >> 24) & 0xFF;
            send_buffer[1] = (seq_num >> 16) & 0xFF;
            send_buffer[2] = (seq_num >> 8) & 0xFF;
            send_buffer[3] = seq_num & 0xFF;

            // 4字节总分片数
            send_buffer[4] = (total_chunks >> 24) & 0xFF;
            send_buffer[5] = (total_chunks >> 16) & 0xFF;
            send_buffer[6] = (total_chunks >> 8) & 0xFF;
            send_buffer[7] = total_chunks & 0xFF;

            // **关键修复：复制 payload 到 buffer[8:]**
            memcpy(send_buffer + 8, file_data + offset, chunk_size);

            // 发送 header + payload
            sendto(sockfd, send_buffer, chunk_size + 8, 0,
                   (const struct sockaddr *)&dest_addr, sizeof(dest_addr));
        }

        gettimeofday(&end_time, NULL);

        long seconds = end_time.tv_sec - start_time.tv_sec;
        long useconds = end_time.tv_usec - start_time.tv_usec;
        long msec = (seconds * 1000) + (useconds / 1000);

        //printf("transmission_time_ms: %ld\n", msec);

        free(file_data);

        // 删除文件（修复文件名）
        remove("/tmp/" SENSOR_FILE);
    }

    close(sockfd);
    return 0;
}