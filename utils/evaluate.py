from sklearn.metrics import average_precision_score, log_loss

def calculate_ctr(gt):
<<<<<<< HEAD
  positive = len([x for x in gt if x == 1])
  ctr = positive/float(len(gt)) # label에서 전체 positive 비율
  return ctr

def compute_rce(pred, gt):
    cross_entropy = log_loss(gt, pred) # label과 예측값의 cross entropy loss
    data_ctr = calculate_ctr(gt) # label에서 전체 positive 비율
    strawman_cross_entropy = log_loss(gt, [data_ctr for _ in range(len(gt))]) 
    return (1.0 - cross_entropy/strawman_cross_entropy)*100.0
=======
    positive = len([x for x in gt if x == 1])
    ctr = positive/float(len(gt))
    return ctr

def compute_rce(pred, gt):
    cross_entropy = log_loss(gt, pred)
    data_ctr = calculate_ctr(gt)
    strawman_cross_entropy = log_loss(gt, [data_ctr for _ in range(len(gt))])
    return (1.0 - cross_entropy/strawman_cross_entropy)*100.0
>>>>>>> 8e2cc1bb3a0e7a4c33d308448c2297e256cfeac8
