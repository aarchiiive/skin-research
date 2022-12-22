from train import train

if __name__ == "__main__":
    """모델 학습

    Args:
        dataset_path (_type_): train, test, label 폴더가 있는 상위 폴더 이름을 입력해주세요 ex) "data"
        save_path (_type_): weights와 log.txt를 저장할 경로를 입력해주세요 ex) "lab01", "lab14" 
        model_name (_type_): 학습시킬 모델명을 입력해주세요(아래의 model_list 참고) ex) "resnet50", "efficientnet_v2_l"
        num_epochs (int, optional): 총 학습시킬 epoch. Defaults to 25.
        input_size (int, optional): 이미지의 사이즈. Defaults to 600.
        class_names (list, optional): 분류시킬 class의 이름들. Defaults to ["level 0", "level 1", "level 2", "level 3"].
        optimize (str, optional): _description_. Defaults to "adam".
        learning_rate (float, optional): learning rate. Defaults to 0.0001.
        weight_decay (float, optional): weight decay (parameter in Adam). Defaults to 0.0005.
        drop_rate (float, optional): 모델의 마지막 fc layer에서 적용할 dropout 비율. Defaults to 0.2.
        batch_size (int, optional):  batch size -> gpu 성능에 따라서 늘리거나 줄여주세요. Defaults to 8.
        num_workers (int, optional): cpu 코어 성능에 따라서 늘리거나 줄여주세요. Defaults to 10.
        resume (bool, optional): 학습을 재개하고 싶다면 True를 넣어주세요 -> save_path에 있는 last.pt를 불러옴. Defaults to False.
        use_wandb (bool, optional): wandb를 사용한다면 True로 설정. Defaults to False.
        project_name (_type_, optional): wandb에 사용할 프로젝트 이름. Defaults to None.
        start (int, optional): _description_. Defaults to 0.
        
    [model list]
    - resnet50
    - resnet101
    - resnet152
    - resnext50
    - resnext101
    - densenet121
    - densenet161
    - densenet169
    - densenet201
    - efficientnet_b5
    - efficientnet_b6
    - efficientnet_b7
    - efficientnet_v2_s
    - efficientnet_v2_m
    - efficientnet_v2_l
    - xception
    - xception41
    - xception41p
    - xception65
    - xception65p
    - xception71s
    
    조정해야할 parameter는 num_epochs, batch_size, num_workers입니다.
    CUDA memroy error가 발생하거나 CPU/RAM 관련 error가 발생하면 batch_size, num_workers를 낮춰보는 것을 추천드립니다.
    -> ex) batch_size=4, num_workers=4
    num_epochs는 25를 추천드리나 시간적 여유가 되신다면 30~40으로 설정하여도 괜찮습니다.
    num_epochs를 연구하시는 분들이 서로 상의 후에 동일하게 맞추는 것이 좋습니다.
    
    모델 성능에 영향이 있는 parameter는 learning_rate, drop_rate입니다.
    현재 default 값들도 좋은 성능을 보여주고 있습니다!
    이는 교수님과 연구하시는 분들 서로 회의 후 문제가 있을 경우 변경하시는 것을 추천 드립니다.
    """
    
    """
    dataset_path와 save_path는 폴더명 하나만 입력해주시면 됩니다.
    아래는 샘플 코드입니다.
    wandb를 사용할 경우 use_wandb를 True로, 사용하실 project_name을 설정해주세요.
    """
    train("data", "lab01", "resnet50", batch_size=2, num_workers=8, use_wandb=True, project_name="skin-training")
    train("data", "lab02", "densenet161", batch_size=16, num_workers=16, num_epochs=50, use_wandb=True, project_name="skin-training")
    train("data", "lab03", "resnext50", batch_size=16, num_workers=32) # without wandb
    train("data", "lab04", "xception41", batch_size=8, num_workers=16, use_wandb=True, project_name="skin-training")
