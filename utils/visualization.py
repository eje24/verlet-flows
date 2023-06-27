import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def graph_frame(receptor, ligand):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    receptor_edges = torch.tensor([[0, 1], [0, 2], [0, 3]])
    ligand_edges = torch.tensor([[0, 1], [0, 2], [0, 3]])

    for idx, node in enumerate(torch.cat([receptor,ligand])):
        ax.scatter(node[0], node[1], node[2], color='black', label='1')
        ax.text(node[0], node[1], node[2], str(idx % 4))

    for edge in receptor_edges:
        x = [receptor[edge[0]][0], receptor[edge[1]][0]]
        y = [receptor[edge[0]][1], receptor[edge[1]][1]]
        z = [receptor[edge[0]][2], receptor[edge[1]][2]]
        ax.plot(x, y, z, color='blue')

    for edge in ligand_edges:
        x = [ligand[edge[0]][0], ligand[edge[1]][0]]
        y = [ligand[edge[0]][1], ligand[edge[1]][1]]
        z = [ligand[edge[0]][2], ligand[edge[1]][2]]
        ax.plot(x, y, z, color='green')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()