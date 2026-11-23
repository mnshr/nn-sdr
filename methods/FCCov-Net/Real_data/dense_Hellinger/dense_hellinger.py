import argparse
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent

parent_dir_str = str(parent_dir)
if parent_dir_str not in sys.path:
    sys.path.append(parent_dir_str)

from support_function import *
from mynet import *

emotion = {'HA':0, 'SA':1, 'SU':2, 'AN':3, 'DI':4, 'FE':5, 'NE':6}

def get_label(emo, label, distribution, mode=0):
    n = label.shape[0]
    if mode==0:  
        single_label = np.zeros(n)
        for i in range(n):
            single_label[i] = emo[label[i]]
        return single_label.astype(np.int8)
    elif mode==1:   
        multi_label = np.zeros((n,7))
        r = 0.2     
        for i, dis in enumerate(distribution):
            multi_label[i,:6] = (dis > r).astype(np.int8)
            if label[i] == 'NE':
                multi_label[i,6] = 1
            else:
                multi_label[i,6] = 0
        return multi_label
    
def parse_args():
    parser = argparse.ArgumentParser(description="Run JAFFE experiment under the Hellinger distance.")
    parser.add_argument("--d", type=int, default=1, help="Structural dimension.")
    parser.add_argument("--bl", type=float, default=2e-6, help="Regularization parameter blambda.")
    return parser.parse_args()


def main(args=None):
    if args is None:
        args = parse_args()

    directory = str(current_dir.parent / "data" / "jaffe") + "/"
    data = np.load(f'{directory}data.npy')
    label = np.load(f'{directory}label.npy')
    counts = np.load(f'{directory}counts.npy')


    wire_data = data.reshape(213, -1)/255.0
    data = data[:, np.newaxis, :, :]/255.0


    dis_sum = counts.sum(axis=1)[:,np.newaxis]
    dis = np.tile(dis_sum, (1,6))
    distribution = counts/dis


    target = get_label(emotion, label, distribution, 0)
    multi_label = get_label(emotion, label, distribution, 1)

    dist_mode = 3 # Hellinger
    device = 'cuda:2'
    arch = 'DenseNet'   # LeNet
    iter_num = 500
    batch_size = 64
    d = args.d
    bl = args.bl

    rep = 5
    acc = np.zeros((2,4))
    mse_bp = np.zeros((2,4))
    mse_knn = np.zeros((2,4))
    for i in range(rep):

        m_tr_data, m_te_data, w_tr_data, w_te_data, tr_label, te_label, tr_dis, te_dis = train_test_split(
        data, wire_data, target, distribution, test_size=0.2)
        
        mccov_net = CustomNet(architecture=arch, optimizer_name='Adam', iter_num=iter_num, batch_size=batch_size, 
                    blambda=bl, dist_mode=dist_mode, device=device, output_dim=d)
        mccov_net.train_step(m_tr_data, tr_dis, mccov_net.mccov_loss)

        x_train = mccov_net.transform(m_tr_data)
        x_test = mccov_net.transform(m_te_data)
        
        D = distance_matrix(tr_dis, dist_mode) 
        cv_wire = True
        eps, gamma = param(w_tr_data, D, cv_wire)
        train_res, test_res = wire(w_tr_data, w_te_data, D, d, eps, gamma)
        
        acc += classification(x_train, x_test, train_res, test_res, tr_label, te_label)  
        mse_bp += LDL_BP(x_train, x_test, train_res, test_res, tr_dis, te_dis, d, num_epochs=100, batch_size = 32)
        mse_knn += LDL_KNN(x_train, x_test, train_res, test_res, tr_dis, te_dis, k = 50, bandwidth = 10)
        
    acc /= rep
    mse_bp /= rep
    mse_knn /= rep

    print("Classification accuracy:")           # 0-4 represents [Linear Regression, KNN, Random Forest, SVM]
    print(f"MCCOV{acc[0][:1] + acc[0][2:]};")   # Do Not print KNN result       
    print(f"WIRE{acc[1][:1] + acc[1][2:]}")
    print("MSE of KNN (Frechet Regression):")
    print(f"MCCOV{mse_knn[0]};")
    print(f"WIRE{mse_knn[1]}")

if __name__ == "__main__":
    main()
