#include <stdio.h>
#include <cstdlib>
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#include <stdlib.h>
#include <opencv/cv.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#define unsigned char uchar

int block_size = 3;

// 2차원 double 배열을 할당하는 함수
double** d_alloc(int rows, int cols) {
    double** array = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        array[i] = (double*)malloc(cols * sizeof(double));
    }
    return array;
}

// 이미지 메모리를 할당하고 2D uchar 배열을 반환하는 함수
uchar** uc_alloc(int size_x, int size_y) {
    uchar** m = (uchar**)calloc(size_y, sizeof(uchar*));
    for (int i = 0; i < size_y; i++) {
        m[i] = (uchar*)calloc(size_x, sizeof(uchar));
    }
    return m;
}

// 이진 파일에서 2D uchar 배열에 데이터를 읽는 함수
void read_ucmatrix(int size_x, int size_y, uchar** ucmatrix, const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (f == NULL) {
        printf("%s File open Error!\n", filename);
        exit(0);
    }
    for (int i = 0; i < size_y; i++) {
        fread(ucmatrix[i], sizeof(uchar), size_x, f);
    }
    fclose(f);
}

// 2D uchar 배열의 데이터를 이진 파일로 쓰는 함수
void write_ucmatrix(int size_x, int size_y, uchar** ucmatrix, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (f == NULL) {
        printf("%s File open Error!\n", filename);
        exit(0);
    }
    for (int i = 0; i < size_y; i++) {
        fwrite(ucmatrix[i], sizeof(uchar), size_x, f);
    }
    fclose(f);
}

// uc_alloc을 통해 할당된 메모리를 해제하는 함수
void uc_free(int rows, uchar** matrix) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// 2D double 배열 할당 메모리를 해제하는 함수
void d_free(int rows, int cols, double** array) {
    for (int i = 0; i < rows; i++) {
        free(array[i]); // 각 행에 대한 메모리 해제
    }
    free(array); // 배열 자체에 대한 메모리 해제
}

// 가우시안 필터링을 위한 컨볼루션 함수
void convolution(double** h, int F_length, int size_x, int size_y, uchar** image1, uchar** image2) {
    int margin = F_length / 2;
    for (int i = 0; i < size_y; i++) {
        for (int j = 0; j < size_x; j++) {
            double sum = 0.0;
            for (int y = 0; y < F_length; y++) {
                int indexY = i - margin + y;
                if (indexY < 0) indexY = -indexY;
                else if (indexY >= size_y) indexY = 2 * size_y - indexY - 1;
                for (int x = 0; x < F_length; x++) {
                    int indexX = j - margin + x;
                    if (indexX < 0) indexX = -indexX;
                    else if (indexX >= size_x) indexX = 2 * size_x - indexX - 1;
                    sum += h[y][x] * (double)image1[indexY][indexX];
                }
            }
            image2[i][j] = (uchar)(sum < 0 ? 0 : sum > 255 ? 255 : sum);
        }
    }
}

// FFT 연산을 위해 배열 재배열하는 함수
int rearrange(double* X, int N) {
    int i, j, * power_of_2, * pos, stage, num_of_stages = 0;
    double temp;

    for (i = N; i > 1; i >>= 1, num_of_stages++);
    if ((power_of_2 = (int*)malloc(sizeof(int) * num_of_stages)) == NULL)
        return -1;
    if ((pos = (int*)malloc(sizeof(int) * N)) == NULL)
        return -1;

    power_of_2[0] = 1;
    for (stage = 1; stage < num_of_stages; stage++)
        power_of_2[stage] = power_of_2[stage - 1] << 1;

    for (i = 1; i < N - 1; i++)
        pos[i] = 0;
    for (i = 1; i < N - 1; i++)
    {
        if (!pos[i])
        {
            for (j = 0; j < num_of_stages; j++)
            {
                if (i & power_of_2[j])
                    pos[i] += power_of_2[num_of_stages - 1 - j];
            }
            temp = X[i];
            X[i] = X[pos[i]];
            X[pos[i]] = temp;
            pos[pos[i]] = 1;
        }
    }
    free(power_of_2);
    free(pos);
    return 0;
}

// 1D FFT 연산을 수행하는 함수
void fft(double* X_re, double* X_im, int N)
{
    double X_temp_re, X_temp_im;
    double phase;
    int num_of_stages = 0, num_of_elements, num_of_sections, size_of_butterfly;
    int i, j, stage, m1, m2;
    for (i = N; i > 1; i >>= 1, num_of_stages++);
    num_of_elements = N;
    num_of_sections = 1;
    size_of_butterfly = N >> 1;
    for (stage = 0; stage < num_of_stages; stage++) {
        m1 = 0;
        m2 = size_of_butterfly;
        for (i = 0; i < num_of_sections; i++) {
            for (j = 0; j < size_of_butterfly; j++, m1++, m2++) {
                X_temp_re = X_re[m1] - X_re[m2];
                X_temp_im = X_im[m1] - X_im[m2];
                X_re[m1] = X_re[m1] + X_re[m2];
                X_im[m1] = X_im[m1] + X_im[m2];
                phase = -2.0 * M_PI * j / num_of_elements;
                X_re[m2] = X_temp_re * cos(phase) - X_temp_im * sin(phase);
                X_im[m2] = X_temp_re * sin(phase) + X_temp_im * cos(phase);
            }
            m1 += size_of_butterfly;
            m2 += size_of_butterfly;
        }
        num_of_elements >>= 1;
        num_of_sections <<= 1;
        size_of_butterfly >>= 1;
    }
    rearrange(X_re, N);
    rearrange(X_im, N);
}

// 2D FFT 연산을 수행하는 함수
int fft_2d(double** X_re, double** X_im, int N)
{
    int i, j;
    double* temp_re, * temp_im;

    if ((temp_re = (double*)malloc(sizeof(double) * N)) == NULL)
        return -1;
    if ((temp_im = (double*)malloc(sizeof(double) * N)) == NULL)
        return -1;

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            temp_re[j] = X_re[i][j] * pow(-1, j);
            temp_im[j] = X_im[i][j] * pow(-1, j);
        }
        fft(temp_re, temp_im, N);

        for (j = 0; j < N; j++) {
            X_re[i][j] = temp_re[j];
            X_im[i][j] = temp_im[j];
        }
    }
    for (j = 0; j < N; j++) {
        for (i = 0; i < N; i++) {
            temp_re[i] = X_re[i][j] * pow(-1, i);
            temp_im[i] = X_im[i][j] * pow(-1, i);
        }
        fft(temp_re, temp_im, N);
        for (i = 0; i < N; i++) {
            X_re[i][j] = temp_re[i] / N;
            X_im[i][j] = temp_im[i] / N;
        }
    }

    free(temp_re);
    free(temp_im);
    return 0;
}

// 이미지에 가우시안 마스크를 적용하는 함수
void applyGaussianFilter(uchar** image, int width, int height, double** mask, int maskSize) {
    int halfMaskSize = maskSize / 2;
    double** temp = d_alloc(width, height);

    for (int i = halfMaskSize; i < height - halfMaskSize; i++) {
        for (int j = halfMaskSize; j < width - halfMaskSize; j++) {
            double sum = 0.0;
            for (int mi = -halfMaskSize; mi <= halfMaskSize; mi++) {
                for (int mj = -halfMaskSize; mj <= halfMaskSize; mj++) {
                    sum += image[i + mi][j + mj] * mask[mi + halfMaskSize][mj + halfMaskSize];
                }
            }
            temp[i][j] = sum;
        }
    }

    // 복사
    for (int i = halfMaskSize; i < height - halfMaskSize; i++) {
        for (int j = halfMaskSize; j < width - halfMaskSize; j++) {
            image[i][j] = (uchar)(temp[i][j] + 0.5); // 반올림 후 uchar로 변환
        }
    }

    d_free(width, height, temp);
}

// 2D IFFT 연산을 수행하는 함수
void ifft_2d(double** X_re, double** X_im, int N)
{
    int i, j;
    double** temp_re, ** temp_im;

    if ((temp_re = (double**)malloc(sizeof(double*) * N)) == NULL)
        return;
    if ((temp_im = (double**)malloc(sizeof(double*) * N)) == NULL)
        return;

    for (i = 0; i < N; i++) {
        temp_re[i] = (double*)malloc(sizeof(double) * N);
        temp_im[i] = (double*)malloc(sizeof(double) * N);
    }

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            temp_re[j][i] = X_re[i][j];
            temp_im[j][i] = X_im[i][j];
        }
    }

    fft_2d(temp_re, temp_im, N);

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            X_re[i][j] = temp_re[j][i];
            X_im[i][j] = temp_im[j][i];
        }
    }

    for (i = 0; i < N; i++) {
        free(temp_re[i]);
        free(temp_im[i]);
    }
    free(temp_re);
    free(temp_im);
}

double** sobelYMask, ** sobelXMask, ** PrewittMask, ** RobertsMask, ** LaplaceMask, ** Laplace2Mask;
int mask_size = 3;
void make_Mask(int mask_size, double** Mask, int checkMask)
{
    int i, j;
    double Laplace1Mask[3][3] = { 0,-1,0,-1,4,-1,0,-1,0 };
    double Laplace2Mask[3][3] = { -1,-1,-1,-1,8,-1,-1,-1,-1 };
    switch (checkMask)
    {
    case 0:
        for (i = 0; i < mask_size; i++)
            for (j = 0; j < mask_size; j++)
                Mask[i][j] = Laplace1Mask[i][j];
        break;
    case 1:
        for (i = 0; i < mask_size; i++)
            for (j = 0; j < mask_size; j++)
                Mask[i][j] = Laplace2Mask[i][j];
        break;
    default:
        printf("Mask Number is wrong \n");
        exit(1);
    }
}

void laplacianSharpening(uchar** inImg, uchar** outImg, int Row, int Col, int checkMask)
{
    double** laplaceMask = d_alloc(mask_size, mask_size);
    make_Mask(mask_size, laplaceMask, checkMask);

    convolution(laplaceMask, mask_size, Col, Row, inImg, outImg);

    // Free the allocated memory for the Laplacian mask
    for (int i = 0; i < mask_size; i++) {
        free(laplaceMask[i]);
    }
    free(laplaceMask);
}


// 패딩을 추가하는 함수
uchar** add_padding(uchar** input, int original_rows, int original_cols, int new_rows, int new_cols) {
    uchar** padded_image = uc_alloc(new_cols, new_rows);

    for (int i = 0; i < new_rows; i++) {
        for (int j = 0; j < new_cols; j++) {
            if (i < original_rows && j < original_cols) {
                padded_image[i][j] = input[i][j];
            }
            else {
                padded_image[i][j] = 0; // 패딩 영역은 0(검은색)으로 채웁니다.
            }
        }
    }

    return padded_image;
}

// 패딩을 제거하는 함수
uchar** remove_padding(uchar** input, int original_rows, int original_cols, int padded_rows, int padded_cols) {
    uchar** cropped_image = uc_alloc(original_cols, original_rows);

    for (int i = 0; i < original_rows; i++) {
        for (int j = 0; j < original_cols; j++) {
            cropped_image[i][j] = input[i][j];
        }
    }

    return cropped_image;
}

int main(int argc, char* argv[]) {
    if (argc != 6)
    {
        printf("\n. Usage : %s inImage COL ROW outImage flag \n", argv[0]);
        exit(1);
    }
 
    int checkMask;
    int flag = atoi(argv[5]);

    if (flag == 0) {
        int size_x = atoi(argv[2]);
        int size_y = atoi(argv[3]);

        double** gaussMask = d_alloc(block_size, block_size);
        gaussMask[0][0] = 1 / 16.;
        gaussMask[0][1] = 2 / 16.;
        gaussMask[0][2] = 1 / 16.;
        gaussMask[1][0] = 2 / 16.;
        gaussMask[1][1] = 4 / 16.;
        gaussMask[1][2] = 2 / 16.;
        gaussMask[2][0] = 1 / 16.;
        gaussMask[2][1] = 2 / 16.;
        gaussMask[2][2] = 1 / 16.;

        uchar** inImg = uc_alloc(size_x, size_y);
        uchar** outImg = uc_alloc(size_x, size_y);

        read_ucmatrix(size_x, size_y, inImg, argv[1]);

        convolution(gaussMask, block_size, size_x, size_y, inImg, outImg);

        write_ucmatrix(size_x, size_y, outImg, argv[4]);

        uc_free(size_y, inImg);
        uc_free(size_y, outImg);

        for (int i = 0; i < block_size; i++) {
            free(gaussMask[i]);
        }
        free(gaussMask);

        return 0;
    }
    else if (flag == 1) {
        const char* inputFileName = argv[1];
        const char* outputFileName = argv[4];

        int originalWidth = atoi(argv[2]);
        int originalHeight = atoi(argv[3]);
        int extendedWidth = 1024;
        int extendedHeight = 1024;

        uchar** originalImage = NULL;
        uchar** extendedImage = NULL;

        // 이미지 파일에서 데이터 읽기
        FILE* inputFile = fopen(inputFileName, "rb");
        if (!inputFile) {
            printf("Error opening input file: %s\n", inputFileName);
            return 1;
        }

        // 메모리 할당
        originalImage = uc_alloc(originalWidth, originalHeight);

        if (!originalImage) {
            printf("Memory allocation error for originalImage\n");
            fclose(inputFile);
            return 1;
        }

        // 이미지 데이터 읽기
        read_ucmatrix(originalWidth, originalHeight, originalImage, inputFileName);

        fclose(inputFile);

        // 가우시안 마스크 생성
        int maskSize = 3;  // 3x3 마스크
        double** gaussMask = d_alloc(maskSize, maskSize);
        gaussMask[0][0] = 1 / 16.0;
        gaussMask[0][1] = 2 / 16.0;
        gaussMask[0][2] = 1 / 16.0;
        gaussMask[1][0] = 2 / 16.0;
        gaussMask[1][1] = 4 / 16.0;
        gaussMask[1][2] = 2 / 16.0;
        gaussMask[2][0] = 1 / 16.0;
        gaussMask[2][1] = 2 / 16.0;
        gaussMask[2][2] = 1 / 16.0;



        // 이미지 크기를 2의 제곱수로 조정
        extendedImage = uc_alloc(extendedWidth, extendedHeight);

        for (int i = 0; i < extendedHeight; i++) {
            for (int j = 0; j < extendedWidth; j++) {
                if (i < originalHeight && j < originalWidth) {
                    extendedImage[i][j] = originalImage[i][j];
                }
                else {
                    extendedImage[i][j] = 0;
                }
            }
        }

        // FFT 필터링
        int N = extendedHeight; // 큰 이미지의 높이를 사용
        double** imageInReal = d_alloc(N, N);
        double** imageInImag = d_alloc(N, N);

        // 이미지 데이터를 double 형식의 배열로 복사
        for (int i = 0; i < extendedHeight; i++) {
            for (int j = 0; j < extendedWidth; j++) {
                imageInReal[i][j] = (double)extendedImage[i][j];
                imageInImag[i][j] = 0.0;  // 초기값 설정
            }
        }

        // 2D FFT 수행
        fft_2d(imageInReal, imageInImag, N);
           

        // 이미지에 가우시안 필터 적용
        applyGaussianFilter(extendedImage, extendedWidth, extendedHeight, gaussMask, maskSize);

        // 2D IFFT 수행
        ifft_2d(imageInReal, imageInImag, N);

        // 결과 이미지를 원래 크기로 다시 축소
        for (int i = 0; i < originalHeight; i++) {
            for (int j = 0; j < originalWidth; j++) {
                originalImage[i][j] = extendedImage[i][j];
            }
        }

        // 결과 이미지를 저장
        FILE* outputFile = fopen(outputFileName, "wb");
        
        // 결과 이미지 데이터 저장
        write_ucmatrix(originalWidth, originalHeight, originalImage, outputFileName);

        fclose(outputFile);

        return 0;
    }
    else if (flag == 2) {
        int Row = atoi(argv[3]);
        int Col = atoi(argv[2]);


        uchar**  inImg = uc_alloc(Col, Row);
        uchar**  outImg = uc_alloc(Col, Row);

        // 이미지 파일 읽기
        read_ucmatrix(Col, Row, inImg, argv[1]);

        printf("Start Image Filtering\n");


        printf("Enter the edge mask number (0 :Laplace1Mask, 1 :Laplace2Mask): ");
        scanf_s("%d", &checkMask);

        // 샤프닝 필터링 (Laplacian Sharpening) 추가
        laplacianSharpening(inImg, outImg, Row, Col, checkMask);


        printf("Finished Image Filtering\n");


        // 결과 이미지 파일 쓰기
        write_ucmatrix(Col, Row, outImg, argv[4]);


        return 0;
    }
    else if (flag == 3) {
        const char* input_filename = argv[1]; // 이미지 파일 이름
        const char* output_filename = argv[4]; // 출력 파일 이름
        int size_x = atoi(argv[2]);
        int size_y = atoi(argv[3]);


        printf("Enter the edge mask number (0 :Laplace1Mask, 1 :Laplace2Mask): ");
        scanf_s("%d", &checkMask);


        // 이미지 데이터 로드
        uchar** inImg = uc_alloc(size_x, size_y);
        read_ucmatrix(size_x, size_y, inImg, input_filename);

        // 이미지에 패딩 추가 (1024x1024로 만듦)
        uchar** paddedImg = add_padding(inImg, size_y, size_x, 1024, 1024);



        // FFT를 위한 메모리 할당 (패딩된 이미지 크기에 맞춤)
        double** X_re = d_alloc(1024, 1024);
        double** X_im = d_alloc(1024, 1024);

        // 패딩된 이미지 데이터를 FFT 입력으로 복사 (허수부는 0으로 초기화)
        for (int i = 0; i < 1024; i++) {
            for (int j = 0; j < 1024; j++) {
                X_re[i][j] = (i < size_y&& j < size_x) ? (double)paddedImg[i][j] : 0.0;
                X_im[i][j] = 0.0;
            }
        }

        // FFT 수행
        fft_2d(X_re, X_im, 1024);

        // 라플라시안 필터 적용 (패딩된 이미지에 적용)
        uchar** outImg = uc_alloc(1024, 1024);
        laplacianSharpening(paddedImg, outImg, 1024, 1024, checkMask);

        // 역 FFT 수행
        ifft_2d(X_re, X_im, 1024);

        // 결과 이미지에서 패딩 제거 (원래 크기로 되돌림)
        uchar** resultImg = remove_padding(outImg, size_y, size_x, 1024, 1024);



        // 결과 이미지 저장
        write_ucmatrix(size_x, size_y, outImg, output_filename);

        // 메모리 해제
        d_free(size_y, size_x, X_re);
        d_free(size_y, size_x, X_im);

        return 0;
    }
}