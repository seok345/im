import cv2
import os
import glob
import numpy as np  # 혹시 모를 배열 처리 오류를 위해 numpy 임포트


def convert_to_grayscale_and_save(input_dir, output_dir):
    """
    지정된 폴더의 모든 컬러 이미지들을 흑백으로 변환하여 다른 폴더에 저장합니다.
    """
    # 출력 폴더 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"출력 폴더 '{output_dir}'를 생성했습니다.")

    # 지원되는 이미지 확장자 리스트
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_count = 0

    print(f"'{input_dir}' 폴더에서 이미지를 검색합니다...")

    # 모든 확장자의 이미지 파일 목록을 가져옵니다.
    for ext in image_extensions:
        # glob.glob은 파일 경로 목록을 리스트로 반환합니다.
        for filepath in glob.glob(os.path.join(input_dir, ext)):
            filename = os.path.basename(filepath)

            # 컬러 이미지 로드 (BGR 포맷)
            color_image = cv2.imread(filepath, cv2.IMREAD_COLOR)

            if color_image is None:
                print(f"경고: 파일을 로드할 수 없습니다: {filename}. 파일이 손상되었거나 경로 문제가 있을 수 있습니다.")
                continue

            # 컬러 이미지를 흑백으로 변환 (1채널)
            grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            # 흑백 이미지 저장 경로 설정 및 저장
            output_filepath = os.path.join(output_dir, filename)
            cv2.imwrite(output_filepath, grayscale_image)

            image_count += 1
            # print(f"변환 완료: {filename}")

    print(f"\n✅ 총 {image_count}개의 이미지를 흑백으로 변환하여 '{output_dir}'에 저장했습니다.")


if __name__ == '__main__':
    # --- 설정 ---
    COLOR_IMAGE_DIR = './color_images'  # 원본 컬러 이미지를 넣어둘 폴더
    GRAYSCALE_IMAGE_DIR = './grayscale_images'  # 흑백 이미지를 저장할 폴더

    # 흑백 변환 전에 원본 이미지 폴더가 존재하는지 확인
    if not os.path.exists(COLOR_IMAGE_DIR):
        os.makedirs(COLOR_IMAGE_DIR)
        print(f"'{COLOR_IMAGE_DIR}' 폴더를 생성했습니다. 여기에 학습에 사용할 컬러 이미지를 넣고 다시 실행해주세요.")
    else:
        convert_to_grayscale_and_save(COLOR_IMAGE_DIR, GRAYSCALE_IMAGE_DIR)