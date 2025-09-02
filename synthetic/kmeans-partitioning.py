# imports
import numpy as np
from plant import SyntheticSystem
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from sklearn.cluster import KMeans

# 4D feature embedding (encodes position and displacement vector)
def make_features(P, U, sx, sy, lambda_pos, lambda_dir):
    S = np.array([1.0/sx, 1.0/sy], float)
    P_scaled = P * S # removes dependence on choice of sx, sy
    return np.hstack([np.sqrt(lambda_pos)*P_scaled, np.sqrt(lambda_dir)*U])

def run_kmeans(X, K, rng=0):
    km = KMeans(
        n_clusters=K,
        init='k-means++',
        n_init='auto',
        max_iter=300,
        algorithm='elkan',
        random_state=rng
    )
    labels = km.fit_predict(X)
    centers = km.cluster_centers_
    return labels, centers

# Voronoi cell grid on rectangular region
def voronoi_finite_polygons_2d(vor, bbox):
    # convert infinite Voronoi regions to finite polygons clipped to region
    # region = (xmin, xmax, ymin, ymax)
    
    from collections import defaultdict
    xmin, xmax, ymin, ymax = bbox
    center = vor.points.mean(axis=0)
    radius = 10*(max(xmax-xmin, ymax-ymin))

    # construct a map: point -> ridge vertices
    all_ridges = defaultdict(list)
    for (p, q), rv in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges[p].append((q, rv))
        all_ridges[q].append((p, rv))

    regions = []
    for p, region_idx in enumerate(vor.point_region):
        verts = vor.regions[region_idx]
        if len(verts) == 0 or -1 in verts:
            # reconstruct a finite region
            ridges = all_ridges[p]
            new_verts = []
            for q, rv in ridges:
                v0, v1 = rv
                if v0 >= 0 and v1 >= 0:
                    new_verts += [vor.vertices[v0], vor.vertices[v1]]
                else:
                    # vertex at infinity, need to extend
                    v = vor.vertices[v0 if v0 >= 0 else v1]
                    t = vor.points[q] - vor.points[p]
                    t = np.array([t[1], -t[0]])  # perpendicular
                    t /= np.linalg.norm(t)
                    # direction: away from center
                    if np.dot(v - center, t) < 0:
                        t = -t
                    far = v + t*radius
                    new_verts += [v, far]
            # order and unique
            new_verts = np.array(new_verts)
            # compute centroid, sort by angle
            c = new_verts.mean(0)
            angles = np.arctan2(new_verts[:,1]-c[1], new_verts[:,0]-c[0])
            new_verts = new_verts[np.argsort(angles)]
        else:
            new_verts = vor.vertices[verts]

        # clip to bbox (Sutherland–Hodgman algorithm)
        poly = new_verts
        for edge in [(xmin,1,lambda p: p[0]>=xmin),
                     (xmax,-1,lambda p: p[0]<=xmax),
                     (ymin,2,lambda p: p[1]>=ymin),
                     (ymax,-2,lambda p: p[1]<=ymax)]:
            bound, sign, inside = edge
            clipped = []
            for a,b in zip(poly, np.roll(poly, -1, axis=0)):
                ina, inb = inside(a), inside(b)
                if ina and inb:
                    clipped.append(b)
                elif ina and not inb:
                    # a -> intersection
                    if sign in (1,-1): t = (bound - a[0])/(b[0]-a[0])
                    else:               t = (bound - a[1])/(b[1]-a[1])
                    clipped.append(a + t*(b-a))
                elif (not ina) and inb:
                    if sign in (1,-1): t = (bound - a[0])/(b[0]-a[0])
                    else:               t = (bound - a[1])/(b[1]-a[1])
                    clipped.append(a + t*(b-a))
                    clipped.append(b)
            poly = np.array(clipped) if len(clipped)>0 else np.zeros((0,2))
        regions.append(poly)
    return regions, np.arange(len(vor.points))

# partitioning pipeline
def partition(K, sx, sy, x_space, y_space, lambda_pos, lambda_dir, rng=0):

    # k-means clustering in embedding space
    X = make_features(P, U, sx, sy, lambda_pos, lambda_dir)
    labels, _ = run_kmeans(X, K, rng=rng)

    # create Voronoi cells
    sites = np.vstack([P[labels==k].mean(0) for k in range(K)])
    mean_dirs = np.vstack([U[labels==k].mean(0) for k in range(K)])
    norms = np.linalg.norm(mean_dirs, axis=1, keepdims=True)
    mean_dirs = np.divide(mean_dirs, np.maximum(norms, 1e-12))

    vor = Voronoi(sites)
    bbox = (x_space[0], x_space[1], y_space[0], y_space[1])
    cells, order = voronoi_finite_polygons_2d(vor, bbox) # clip Voronoi cells to region
    # cells[i] corresponds to sites[order[i]]; order is 0..K-1

    return P, U, labels, sites, cells, mean_dirs

# alignment quanitification (angular variance)
def circular_mean(U):
    
    n = U.shape[0]
    s = U.sum(axis=0) # resultant vector
    R = float(np.linalg.norm(s) / n) # mean resultant length in [0, 1]

    return R

# visualization function
def plot_partition(P, U, labels, sites, cells, mean_dirs, save_path, quiver_skip=3):
    K = len(sites)
    cmap = plt.cm.get_cmap('tab20', K)
    plt.figure(figsize=(7,6))

    # filled convex Voronoi cells (background)
    for k, poly in enumerate(cells):
        if len(poly) >= 3:
            plt.fill(poly[:,0], poly[:,1], alpha=0.20, color=cmap(k),
                     edgecolor=None, linewidth=0.0, zorder=0)

    # quiver arrows for the original vector field
    idx = np.arange(len(P))
    keep = idx[(idx % quiver_skip) == 0]
    colors = [cmap(labels[i]) for i in keep]
    plt.quiver(P[keep,0], P[keep,1], U[keep,0], U[keep,1],
               angles='xy', scale_units='xy', scale=4,
               color=colors, width=0.002, zorder=1)

    # cell boundaries on top
    for k, poly in enumerate(cells):
        if len(poly) >= 2:
            plt.plot(np.r_[poly[:,0], poly[0,0]],
                     np.r_[poly[:,1], poly[0,1]],
                     'k-', linewidth=1.0, zorder=2)

    # site points and mean direction arrows (comment out for larger K)
    # plt.scatter(sites[:,0], sites[:,1], c='k', s=25, zorder=3)
    # for k in range(K):
    #     d = mean_dirs[k]
    #     plt.arrow(sites[k,0], sites[k,1], d[0], d[1],
    #               head_width=0.1, length_includes_head=True, color='k', zorder=3)

    # overlay vector similarity quantification (min=0, max=1) on each cell
    for k in range(K):
        Uk = U[labels==k]
        R = circular_mean(Uk)
        R_trunc = np.floor(R * 1e4) / 1e4 # truncate to 4 decimal places
        
        cx = float(cells[k][:, 0].mean())
        cy = float(cells[k][:, 1].mean())

        plt.text(
            cx, cy,
            f"R={R_trunc:.4f}",
            ha="center", va="center",
            fontsize=2, zorder=3,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.1)
        )

    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x'); plt.ylabel('y')
    plt.title('Voronoi partition of displacement field (synthetic benchmark)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()


if __name__ == '__main__':

    # initialize system
    args = {
        'system_args': {
            'bias': [0.0, 0.0],
            'cov_lb': [[0.5, 0.0],[0.0, 0.5]],
            'cov_ub': [[2.0, 0.0],[0.0, 2.0]],
            'init_state': [0.0, 0.0],
            'goal_state': [10.0, 10.0],
            'gain': 0.5,
            'max_step': 0.7,
            'success_dist': 2.0,
            'time_limit': np.inf,
            'barricades': []
        },
        'sx': 0.1,
        'x_space': [0.0, 12.0],
        'sy': 0.1,
        'y_space': [0.0, 12.0],
    }

    sx = args['sx']; sy = args['sy']
    x_space = args['x_space']; y_space = args['y_space']
    clsys = SyntheticSystem(args['system_args'])

    # sample points and corresponding displacement vectors uniformly from space
    P = []
    U = []
    for i in range(int(x_space[0] / sx), int(x_space[1] / sx) + 1):
        for j in range(int(y_space[0] / sy), int(y_space[1] / sy) + 1):
            v = clsys.pseudostep_2([i * sx, j * sy])
            norm = np.linalg.norm(v)

            if norm == 0: # skip zero vectors
                continue
            u = v / norm

            P.append([i * sx, j * sy])
            U.append(u)
    P = np.array(P, float)
    U = np.array(U, float)

    # run partitioning
    K = 196
    P, U, labels, sites, cells, mean_dirs = partition(
        K, sx, sy, x_space, y_space, lambda_pos=1.0, lambda_dir=5.0, rng=0
    )

    plot_partition(P, U, labels, sites, cells, mean_dirs, save_path='partition.png')
    print('Saved partition.png')