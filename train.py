import os
import torch
import torch.nn as nn
import numpy as np
import random
import timm
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pydicom
import cv2

def seed_everything(seed=777):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()  # 시드 고정

image_size = 256

class MultiTaskEfficientNet(nn.Module):
    def __init__(self, base_model_name='efficientnet_b0', num_classes=1, grid_size=7, num_boxes=3):
        super(MultiTaskEfficientNet, self).__init__()
        self.num_boxes = num_boxes
        self.base_model = timm.create_model(base_model_name, pretrained=False, num_classes=0)
        self.classification_head = nn.Linear(self.base_model.num_features, num_classes)
        self.bbox_head = nn.Linear(self.base_model.num_features, grid_size * grid_size * num_boxes * 3)  # objectness, x_center, y_center
        
    def forward(self, x):
        features = self.base_model(x)
        classification_output = self.classification_head(features)
        bbox_output = self.bbox_head(features)
        bbox_output = bbox_output.view(-1, 7, 7, self.num_boxes * 3)  # (batch_size, grid_size, grid_size, num_boxes * 3)
        return classification_output, bbox_output

# Example usage
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),  # Resize the image to 528x528
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(                   # 정규화
        mean=[0.5, 0.5, 0.5],                # 각 채널의 평균
        std=[0.5, 0.5, 0.5]                  # 각 채널의 표준편차
    ),
])

class MultiTaskSwinV2(nn.Module):
    def __init__(self, base_model_name='gagan3012/swinv2-base-512', num_classes=1, grid_size=7, num_boxes=3):
        super(MultiTaskSwinV2, self).__init__()
        # Load the base model from Hugging Face
        self.num_boxes = num_boxes
        self.base_model = AutoModel.from_pretrained(base_model_name)
        
        # Get the hidden size of the last layer (this can be done based on model config or directly accessing the layer)
        hidden_size = self.base_model.config.hidden_size
        
        # Define the classification and bounding box heads
        self.classification_head = nn.Linear(hidden_size, num_classes)
        self.bbox_head = nn.Linear(hidden_size, grid_size * grid_size * num_boxes * 3)  # objectness, x_center, y_center

    def forward(self, x):
        # Extract features from the base model
        features = self.base_model(x).last_hidden_state[:, 0, :]  # Assuming the first token's output (like [CLS] token)
        
        # Classification and bounding box outputs
        classification_output = self.classification_head(features)
        bbox_output = self.bbox_head(features)
        
        # Reshape the bounding box output
        bbox_output = bbox_output.view(-1, 7, 7, 2 * 3)  # (batch_size, grid_size, grid_size, num_boxes * 3)
        
        return classification_output, bbox_output
    
# class MultiTaskSwinV2(nn.Module):
#     def __init__(self, base_model_name='swinv2_base_window16_256', num_classes=1, grid_size=7, num_boxes=3):
#         super(MultiTaskSwinV2, self).__init__()
#         self.num_boxes = num_boxes
#         self.base_model = timm.create_model(base_model_name, pretrained=False, num_classes=0)
#         self.classification_head = nn.Linear(self.base_model.num_features, num_classes)
#         self.bbox_head = nn.Linear(self.base_model.num_features, grid_size * grid_size * num_boxes * 3)  # objectness, x_center, y_center

#     def forward(self, x):
#         features = self.base_model(x)
#         classification_output = self.classification_head(features)
#         bbox_output = self.bbox_head(features)
#         bbox_output = bbox_output.view(-1, 7, 7, self.num_boxes * 3)  # (batch_size, grid_size, grid_size, num_boxes * 3)
#         return classification_output, bbox_output

# # Example usage
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),  # Resize the image to 528x528
#     transforms.ToTensor(),  # Convert image to tensor
# ])

# class MultiTaskConvNeXt(nn.Module):
#     def __init__(self, base_model_name='convnext_large', num_classes=1, grid_size=7, num_boxes=3):
#         super(MultiTaskConvNeXt, self).__init__()
#         self.num_boxes = num_boxes
#         self.base_model = timm.create_model(base_model_name, pretrained=True, num_classes=0)
#         self.classification_head = nn.Linear(self.base_model.num_features, num_classes)
#         self.bbox_head = nn.Linear(self.base_model.num_features, grid_size * grid_size * num_boxes * 3)  # objectness, x_center, y_center

#     def forward(self, x):
#         features = self.base_model(x)
#         classification_output = self.classification_head(features)
#         bbox_output = self.bbox_head(features)
#         bbox_output = bbox_output.view(-1, 7, 7, self.num_boxes * 3)  # (batch_size, grid_size, grid_size, num_boxes * 3)
#         return classification_output, bbox_output
    
# # Example usage
# transform = transforms.Compose([
#     transforms.Resize((528, 528)),  # Resize the image to 528x528
#     transforms.ToTensor(),  # Convert image to tensor
#     transforms.Normalize(                   # 정규화
#         mean=[0.5, 0.5, 0.5],                # 각 채널의 평균
#         std=[0.5, 0.5, 0.5]                  # 각 채널의 표준편차
#     ),
# ])

class MedicalAnnotatedDataset(Dataset):
    def __init__(
        self,
        root_dir,
        transform=None,
        grid_size=7,
        num_boxes=3,
        clahe_clip=2.0,
        clahe_grid=(8, 8)
    ):
        """
        Args:
            root_dir (str): Dataset의 루트 디렉토리 경로.
            transform (callable, optional): 샘플에 적용될 Optional transform.
            grid_size (int): YOLO 스타일 그리드의 크기 (기본값: 7x7).
            num_boxes (int): 그리드 셀당 예측할 바운딩 박스 개수.
            clahe_clip (float): CLAHE 클립 리밋.
            clahe_grid (tuple): CLAHE 타일 그리드 크기.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.samples = []

        # 클래스 폴더 순회
        for class_name in ['normal', 'stenosis']:
            class_dir = os.path.join(self.root_dir, class_name)
            for subdir, dirs, _ in os.walk(class_dir):
                for dir_name in dirs:
                    full_dir = os.path.join(subdir, dir_name)
                    for fname in os.listdir(full_dir):
                        if not fname.lower().endswith('.dcm'):
                            continue
                        dcm_path = os.path.join(full_dir, fname)
                        if class_name == 'stenosis':
                            txt_path = dcm_path[:-4] + '.txt'
                            if os.path.isfile(txt_path):
                                self.samples.append((dcm_path, class_name, txt_path))
                        else:
                            self.samples.append((dcm_path, class_name, None))

        self.label_map = {'normal': 0, 'stenosis': 1}

        # CLAHE 객체
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dcm_path, label, txt_path = self.samples[idx]

        # DICOM 읽기 및 픽셀 추출
        ds = pydicom.dcmread(dcm_path)
        arr = ds.pixel_array.astype(np.float32)
        # 정규화 0-255
        arr -= arr.min()
        maxv = arr.max() if arr.max() > 0 else 1.0
        arr = (arr / maxv * 255.0).astype(np.uint8)

        # CLAHE 적용
        arr = self.clahe.apply(arr)

        # PIL 이미지 변환 및 RGB 복제
        image = Image.fromarray(arr).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # 분류 라벨
        cls_lbl = self.label_map[label]

        # 그리드 라벨 초기화
        grid_lbl = np.zeros((self.grid_size, self.grid_size, self.num_boxes, 3), dtype=np.float32)

        # bounding box 정보 채우기
        if label == 'stenosis' and txt_path:
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 3:
                        continue
                    x_ctr, y_ctr = float(parts[1]), float(parts[2])

                    gx = min(int(x_ctr * self.grid_size), self.grid_size - 1)
                    gy = min(int(y_ctr * self.grid_size), self.grid_size - 1)

                    rel_x = min(x_ctr * self.grid_size - gx, 0.999)
                    rel_y = min(y_ctr * self.grid_size - gy, 0.999)

                    # 빈 박스 슬롯 찾기
                    for b in range(self.num_boxes):
                        if grid_lbl[gy, gx, b, 0] == 0:
                            grid_lbl[gy, gx, b, 0] = 1.0
                            grid_lbl[gy, gx, b, 1] = rel_x
                            grid_lbl[gy, gx, b, 2] = rel_y
                            break

        return image, cls_lbl, torch.from_numpy(grid_lbl)

# Create dataset instances
train_dataset = MedicalAnnotatedDataset(root_dir='./classification_dcm/train', transform=transform)
val_dataset = MedicalAnnotatedDataset(root_dir='./classification_dcm/val', transform=transform)

# custom_collate_fn을 사용하는 DataLoader 설정
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

def bbox_loss_fn(pred_bboxes, true_bboxes, grid_size=7, lambda_noobj=0.1):
    """
    YOLO 기반 바운딩 박스 손실 함수.
    pred_bboxes: 예측된 바운딩 박스 (batch_size, grid_size, grid_size, num_boxes * 3)
    true_bboxes: 실제 바운딩 박스 라벨 (batch_size, grid_size, grid_size, num_boxes, 3)
    """
    batch_size = pred_bboxes.size(0)
    num_boxes = pred_bboxes.size(-1) // 3  # 바운딩 박스의 개수 추출

    # 예측된 바운딩 박스 리셰이프
    pred_bboxes = pred_bboxes.view(batch_size, grid_size, grid_size, num_boxes, 3)

    # 객체 존재 여부에 대한 마스크 생성
    objectness_mask = (true_bboxes[..., 0] == 1).float()  # 객체가 있는 경우
    no_objectness_mask = (true_bboxes[..., 0] == 0).float()  # 객체가 없는 경우

    # MSE 손실 함수 정의
    mse_loss = nn.MSELoss(reduction='none')
    # BCEWithLogitsLoss 정의 (sigmoid 활성화 포함)
    bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    # 좌표 손실 계산 (객체가 있는 경우에만)
    coord_loss = mse_loss(pred_bboxes[..., 1:], true_bboxes[..., 1:])  # x_center, y_center 비교
    coord_loss = coord_loss.sum(dim=-1)  # 좌표 손실 합산
    coord_loss = (coord_loss * objectness_mask).sum()  # 객체가 있는 경우에만 손실 합산

    # 객체성(objectness) 손실 계산
    objectness_loss = bce_loss(pred_bboxes[..., 0], true_bboxes[..., 0])  # BCEWithLogitsLoss 사용
    objectness_loss_obj = (objectness_loss * objectness_mask).sum()  # 객체가 있는 경우의 손실
    objectness_loss_noobj = (objectness_loss * no_objectness_mask).sum() * lambda_noobj  # 객체가 없는 경우의 손실에 가중치 적용

    # 전체 객체성 손실 합산
    total_objectness_loss = objectness_loss_obj + objectness_loss_noobj

    # 전체 손실 계산
    total_loss = (coord_loss + total_objectness_loss) / batch_size
    return total_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 모델 로드
# model = MultiTaskEfficientNet()
model = MultiTaskSwinV2()
# model = MultiTaskConvNeXt()
model.to(device)

# 최적화기와 손실 함수
optimizer = torch.optim.RAdam(model.parameters(), lr=0.001)
classification_criterion = nn.BCEWithLogitsLoss()

# best acc epcoh: 28 (acc 9470 bbox loss 1.3)
# best bbox loss epoch: 5 (acc 9193 bbox 6031)


## edited version
# best acc epoch: 20 (acc 9485)
# best bbox epoch: 6 (7769)

## convNeXT
# Epoch [77/300], Classification Loss: 0.0038, BBox Loss: 0.0449, Val Classification Loss: 0.3217, Val BBox Loss: 1.8982, Val Accuracy: 0.9460
def train(model, num_epochs, train_loader, val_loader, classification_criterion, optimizer, device, save_path):
    # 로그 파일 경로 설정
    log_file_path = os.path.join(save_path, 'log.txt')
    
    # 경로가 존재하지 않으면 생성
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 로그 파일을 쓰기 모드로 열기 (기존 내용 덮어쓰기)
    with open(log_file_path, 'w') as log_file:
        log_file.write('Epoch,Train Classification Loss,Train BBox Loss,Val Classification Loss,Val BBox Loss,Val Accuracy\n')

    for epoch in range(num_epochs):
        model.train()
        total_classification_loss = 0
        total_bbox_loss = 0

        train_loader_tqdm = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]', unit='batch')

        for images, labels, grid_bboxes in train_loader_tqdm:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            grid_bboxes = grid_bboxes.to(device)  # 바운딩 박스도 장치로 이동

            optimizer.zero_grad()
            classification_output, bbox_output = model(images)
            
            # 분류 손실 계산
            classification_loss = classification_criterion(classification_output, labels)

            # 바운딩 박스 손실 계산
            bbox_loss = bbox_loss_fn(bbox_output, grid_bboxes)

            # 가중치 적용
            total_loss = classification_weight * classification_loss + bbox_weight * bbox_loss

            total_loss.backward()
            optimizer.step()
            total_classification_loss += classification_loss.item()
            total_bbox_loss += bbox_loss.item()

            train_loader_tqdm.set_postfix(classification_loss=total_classification_loss / len(train_loader),
                                          bbox_loss=total_bbox_loss / len(train_loader))

        # 평가 함수 호출
        val_classification_loss, val_bbox_loss, val_acc = evaluate(model, val_loader, classification_criterion, device)
        train_classification_loss = total_classification_loss / len(train_loader)
        train_bbox_loss = total_bbox_loss / len(train_loader)
        
        # 에폭 결과 출력
        print(f'Epoch [{epoch+1}/{num_epochs}], Classification Loss: {train_classification_loss:.4f}, '
              f'BBox Loss: {train_bbox_loss:.4f}, '
              f'Val Classification Loss: {val_classification_loss:.4f}, Val BBox Loss: {val_bbox_loss:.4f}, '
              f'Val Accuracy: {val_acc:.4f}')
        
        # 로그 파일에 기록
        with open(log_file_path, 'a') as log_file:
            log_file.write(f'{epoch+1},{train_classification_loss:.4f},{train_bbox_loss:.4f},'
                           f'{val_classification_loss:.4f},{val_bbox_loss:.4f},{val_acc:.4f}\n')

        # 모델 가중치 저장
        torch.save(model.state_dict(), os.path.join(save_path, f'model_epoch_{epoch+1}.pth'))

def evaluate(model, dataloader, classification_criterion, device):
    model.eval()
    total_classification_loss = 0
    total_bbox_loss = 0
    correct_predictions = 0

    dataloader_tqdm = tqdm(dataloader, desc='Evaluating', unit='batch')
    
    with torch.no_grad():
        for images, labels, grid_bboxes in dataloader_tqdm:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            grid_bboxes = grid_bboxes.to(device)
            
            classification_output, bbox_output = model(images)
            
            # 분류 손실 계산
            classification_loss = classification_criterion(classification_output, labels)

            # 바운딩 박스 손실 계산
            bbox_loss = bbox_loss_fn(bbox_output, grid_bboxes)
            
            total_classification_loss += classification_loss.item()
            total_bbox_loss += bbox_loss.item()

            # 분류 정확도 계산
            predictions = torch.round(torch.sigmoid(classification_output))
            correct_predictions += (predictions == labels).sum().item()

            dataloader_tqdm.set_postfix(classification_loss=total_classification_loss / len(dataloader),
                                        bbox_loss=total_bbox_loss / len(dataloader))

    # 평균 손실 및 정확도 반환
    avg_classification_loss = total_classification_loss / len(dataloader)
    avg_bbox_loss = total_bbox_loss / len(dataloader)
    accuracy = correct_predictions / len(dataloader.dataset)
    return avg_classification_loss, avg_bbox_loss, accuracy

# 가중치 설정
classification_weight = 1.0  # 분류 손실의 가중치
bbox_weight = 1  # 바운딩 박스 손실의 가중치

# 저장할 경로 지정 (예: 'saved_models' 폴더)
# save_path = './weights/edited_dataset_eff_b2'
save_path = f'./weights/dcm_swinv2_sc_{image_size}'

# 훈련 실행
num_epochs = 300
train(model, num_epochs, train_loader, val_loader, classification_criterion, optimizer, device, save_path)