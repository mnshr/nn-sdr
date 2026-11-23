import os
import matplotlib.pyplot as plt
from support_function import *
from mynet import *

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'figures')
os.makedirs(output_dir, exist_ok=True)

# Load visualization data
pos = './data/3d_graph2/'
vis_tr_data = np.load(f"{pos}m_tr_data.npy")
vis_tr_label = np.load(f"{pos}tr_label.npy")

# Load trained model
dist_mode = 3 # Hellinger
device = 'cuda:2'
arch = 'DenseNet'   # LeNet
iter_num = 500
batch_size = 64
d = 3
bl = 2e-6

vis_model = CustomNet(architecture=arch, optimizer_name='Adam', iter_num=iter_num, batch_size=batch_size, lr=1e-3,
                  blambda=bl, dist_mode=dist_mode, device=device, output_dim=d)

vis_model.load_state_dict(torch.load('./data/3d_graph2/3d_dense_2e-6.pth'))

x_train = vis_model.transform(vis_tr_data)
y_train = vis_tr_label

def draw_3d(angle):
    output_path = os.path.join(output_dir, f'3d_view_angle{angle}.png')
    if angle == 1:
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        markers = ['o', 's', '^', 'D', 'v', 'P', '*']
        anti_emotion = {0:'HA', 1:'SA', 2:'SU', 3:'AN', 4:'DI', 5:'FE', 6:'NE'}

        # Create a 3D figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        # ax.set_axis_off()

        # Scatter plot grouped by class labels
        for i in range(7):  # assume six classes
            indices = y_train == i
            ax.scatter(x_train[indices, 0], x_train[indices, 1], x_train[indices, 2], 
                    c=colors[i], marker=markers[i], label=f'Class {anti_emotion[i]}', s=20, alpha=0.6)
        # Update viewing angle
        ax.view_init(elev=5, azim=0)

        # Show legend
        ax.legend()

        # Render plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {output_path}")
        plt.close(fig)

    elif angle == 2:
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        markers = ['o', 's', '^', 'D', 'v', 'P', '*']
        anti_emotion = {0:'HA', 1:'SA', 2:'SU', 3:'AN', 4:'DI', 5:'FE', 6:'NE'}

        # Create a 3D figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        # ax.set_axis_off()

        # Scatter plot grouped by class labels
        for i in range(7):  # assume six classes
            indices = y_train == i
            ax.scatter(x_train[indices, 0], x_train[indices, 1], x_train[indices, 2], 
                    c=colors[i], marker=markers[i], label=f'Class {anti_emotion[i]}', s=20, alpha=0.6)
        # Update viewing angle
        ax.view_init(elev=5, azim=90)

        # Show legend
        ax.legend()

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {output_path}")
        plt.close(fig)

    elif angle == 3:
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        markers = ['o', 's', '^', 'D', 'v', 'P', '*']
        anti_emotion = {0:'HA', 1:'SA', 2:'SU', 3:'AN', 4:'DI', 5:'FE', 6:'NE'}

        # Create a 3D figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        # ax.set_axis_off()

        # Scatter plot grouped by class labels
        for i in range(7):  # assume six classes
            indices = y_train == i
            ax.scatter(x_train[indices, 0], x_train[indices, 1], x_train[indices, 2], 
                    c=colors[i], marker=markers[i], label=f'Class {anti_emotion[i]}', s=20, alpha=0.6)
        # Update viewing angle
        ax.view_init(elev=90, azim=5)
            
        # Show legend
        ax.legend()

        # Render plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {output_path}")
        plt.close(fig)

def main():
    draw_3d(angle=1)
    draw_3d(angle=2)
    draw_3d(angle=3)

if __name__ == "__main__":
    main()
