# 딥러닝 관련 스터디 모음 

# DNN 
- DNN_관련 기초 이론, 코드 모음 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/crimama/DL_study/blob/main/material/DNNbasic.ipynb)
- DNN 기본 템플릿
- 콜백
- 오버피팅 조정 레이어 
- 제너레이터

# Functional.Api
- Functional.api (다중입력, 다중출력) 예제 정리 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/crimama/DL_study/blob/main/material/Functional.api_basic.ipynb)
  - Sequentail 방식의 모델이 아닌 방식 
  - 다중입력, 다중 출력 CNN 가능 <--- 모델 부문 활용 
  - 해당 파일에서 모델 부문만 사용하면 됨 

# U-Net 
- UNET 영상 분할 기초 모음 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/crimama/DL_study/blob/main/material/UNET.ipynb)

  - UNET 영상 분할 - 이진 분류  
  - MNET 영상 분할 - 이진 분류 
  - UNET 영상 분할 - 컬러 이진 분류 
  - UNET 영상 분할 - 다중 분류 
- UNET 실습 (MRI 사진) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/crimama/DL_study/blob/main/unet_practice_mri_images.ipynb)


# 오토인코더 
- 오토인코더 내용 및 템플릿 정리 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/crimama/DL_study/blob/main/material/22.01.10_AutoEncoder_basic.ipynb)
- 오토 인코더 : 차원을 축소하는 인코더, 다시 복원 시키는 디코더를 통해 이상 탐지, 노이즈 제거, 해상도 향상 
- Super Resolution(이건 템플릿만, 추후 스터디 more) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/crimama/DL_study/blob/main/material/22.01.10_Super%20Resolution_example-edsr.ipynb)
   - 이미지의 해상도를 높이는 데 사용 
   - 새로운 데이터를 학습 해 사용할 수 있지만 기존에 학습 된 weight를 불러 와 해상도 향상 가능
   - 학습 된 weight를 사용할 경우 해당 템플릿의 Demo 파트 사용 

# RNN 
- RNN 기본 및 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/crimama/DL_study/blob/main/material/22.01.10_RNN_basic.ipynb)

- Simplest RNN Template [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/crimama/DL_study/blob/main/material/22.01.10_basic_sequence_data_predict.ipynb)

- RNN pracice [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/crimama/DL_study/blob/main/material/22.01.10_practice_RNN_Timeseries_regression.ipynb)


# 딥러닝  응용 - 영상, 이미지 데이터 

## 포즈 추출 
  - 포즈 추출  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/crimama/DL_study/blob/main/material/22.01.06_pose_extraction.ipynb)

  - 웹캠 포즈 추출 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/crimama/DL_study/blob/main/material/22.01.06_pose_extraction_webcam.ipynb)
  - 사람의 포즈를 인식 해서 skeleton 형태로 그림 
  - 이미 학습 된 weight로도 충분 하므로 이미지만 넣어서 그대로 사용 가능 
  - 전신 일 때는 인식률이 좋지만, 상반신, 하반신 잘려 있으면 인식률이 떨어짐 



