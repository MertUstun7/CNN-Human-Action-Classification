# CNN-Human-Action-Classification

# GİRİŞ
Pytorch tabanlı insan eylemleri sınıflandırmaya yönelik bir CNN modeli oluşturan notebook hazırladım.
Öncelikle CSV'den etiketleri okuyup label2idx haritası oluşturdum ve veri setini train/validation/test ayrımı yaparak hazırladım.
Hem klasik hem de güçlendirilmiş bir şekilde farklı data augmentation yöntemleri kullanarak özel Dataset ve DataLoader tanımladım.
Model tarafında, çok katmanlı bir EnhancedCNN tasarladım içine Residualblock +SE attention, BatchNorm, Dropout ardından global average + max pooling ve tam bağlantılı sınıflandırıcı yer aldı.
Eğitimde AdamW optimizer, label smoothing içeren CrossEntropy loss kullandım. Sınıf dengesizliğini önlemeye yönelik class-weighted loss, Mixup/Cutmix, AMP, CosineAnnealing tabanlı çğrenme oranı scheduler'ları, EMA ve son aşamada SWA tekniklerini entegre ettim.
Grad-CAM görselleştirmesi ve accuracy/loss grafiklerini kullandım.
Son aşamada ise değerlendirmede accuracy, confusion matrix ve TTA yöntemlerini kullandım.

# Metrikler
CNN modelini eğitimini tamamladığımda yaklaşık olarak %62'lik bir doğrulama sonucu elde ettim. Aşağıda modelde elde ettiğim sonuçlara yer verilmiştir.

<img width="489" height="512" alt="Ekran görüntüsü 2025-09-25 195828" src="https://github.com/user-attachments/assets/24821d33-2cbf-4878-bb8c-a9f926a11877" />

<img width="484" height="507" alt="Ekran görüntüsü 2025-09-25 195842" src="https://github.com/user-attachments/assets/c1e30dfe-cbf5-4ad6-9c7f-b38d70de4283" />

<img width="489" height="511" alt="Ekran görüntüsü 2025-09-25 195836" src="https://github.com/user-attachments/assets/0a19f3ab-b536-4c60-9ea1-f48973400c58" />

İlerleyen süreçte model başarısını bir üst seviye taşımak için  çalışmalarımı sürdürmeye devam edeceğim.

Eğitim sürecinde Meet Nagadia tarafından hazırlanan Human Action Recognition (HAR) Dataset kullanılmıştır.
https://www.kaggle.com/datasets/meetnagadia/human-action-recognition-har-dataset
