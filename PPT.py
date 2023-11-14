import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
from sklearn import datasets, model_selection
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# 학습 이미지 경로
train_path = 'Trainning/Cataract/'

# 분류 Classes 명 (초기, 미성숙, 성숙, 무증상)
dirs = ['Early', 'Immature', 'Maturity', 'None']

# 데이터와 라벨 리스트
data = []
label = []

# 학습 데이터 불러오기
for i, d in enumerate(dirs):
  files = os.listdir(train_path+d)
  # 읽어드린 이미지 파일 카운팅 변수
  img_count = 0
  for f in files:
    try :
      img = Image.open(train_path + d + '/' + f, 'r')
      # 이미지를 128, 128로 크기 조정
      resize_img = img.resize((128, 128))

      # 정규화 과정
      # 이미지를 RGB 컬러로 각각 쪼갠다.
      r, g, b = resize_img.split()
      # 각 쪼갠 이미지를 255로 나눠서 0~1 사이의 값이 나오도록 정규화
      r_resize_img = np.asarray(np.float32(r) / 255.0)
      b_resize_img = np.asarray(np.float32(g) / 255.0)
      g_resize_img = np.asarray(np.float32(b) / 255.0)
      # 0~1의 값을 가진 데이터로 3채널 데이터 구성
      rgb_resize_img = np.asarray([r_resize_img, b_resize_img, g_resize_img])
      # 정규화 과정으로 가공된 데이터 저장
      data.append(rgb_resize_img)

      # 라벨추가 초기: 0, 미성숙: 1, 성숙 : 2, 무증상: 3)
      label.append(i)
      # 하나의 이미지를 가공하고 학습 데이터셋에 추가할때마다 Print로 출력
      img_count = img_count+1
      print(d+" img count : " + str(img_count))

    except :
      # 예외발생 , Json 파일 무시
      print("Error or JSON FILE")

    # 각 Classes의 학습할 이미지의 수 설정
    if(img_count == 100) : break

# 데이터와 라벨을 각각 적합한 자료형의 어레이로 구성
data = np.array(data, dtype='float32')
label = np.array(label, dtype='int64')

# 학습데이터와 검증데이터 추출
train_X, test_X, train_Y, test_Y = model_selection.train_test_split(data, label, test_size=0.1)

# Torch에 사용하기 위해 데이터 가공
train_X = torch.from_numpy(train_X).float()
train_Y = torch.from_numpy(train_Y).long()

test_X = torch.from_numpy(test_X).float()
test_Y = torch.from_numpy(test_Y).long()

# 학습 데이터 셋 설정 ( Batch Size 8 , 랜덤 추출 On)
train = TensorDataset(train_X, train_Y)
train_loader = DataLoader(train, batch_size=8, shuffle=True)

# 신경망 구성
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    # 합성곱층
    # 입력 채널 수, 출력 채널 수, 필터 크기
    self.conv1 = nn.Conv2d(3, 10, 5)
    self.conv2 = nn.Conv2d(10, 20, 5)

    # 전결합층
    # 29=(((((128-5)+1)/2)-5)+1)/2
    self.fc1 = nn.Linear(20 * 29 * 29, 50)
    self.fc2 = nn.Linear(50, 4)

  def forward(self, x):
    # 풀링층
    x = F.max_pool2d(F.relu(self.conv1(x)), 2) # 풀링 영역 크기
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = x.view(-1, 20 * 29 * 29)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.log_softmax(x)

# 모델 인스턴스 생성
model = Net()

# 손실함수 호출
criterion = nn.CrossEntropyLoss()

# 옵티마이져 객체 생성 , 학습률 0.001
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Train Start")

# 학습시작 epoch
for epoch in range(5):
  total_loss = 0
  for train_x, train_y in train_loader:
    train_x, train_y = Variable(train_x), Variable(train_y)
    optimizer.zero_grad()
    output = model(train_x)
    loss = criterion(output, train_y)
    loss.backward()
    optimizer.step()
    total_loss += loss.data.item()
    # 학습 1회마다 정확도 및 손실 계산 출력
  if (epoch+1) % 1 == 0:
    test_x, test_y = Variable(test_X), Variable(test_Y)
    result = torch.max(model(test_x).data, 1)[1]
    accuracy = sum(test_y.data.numpy() == result.numpy()) / len(test_y.data.numpy())
    print(epoch + 1,accuracy, total_loss)

# 모델 저장
# model - epoch - img_cnt 형태로 모델명 구성
torch.save(model.state_dict(),'model-5-100')
test_x, test_y = Variable(test_X), Variable(test_Y)
ㅊㅊ