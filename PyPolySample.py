import random
import numpy as np
from shapely.affinity import affine_transform
from shapely.geometry import Point, Polygon
from shapely.ops import triangulate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class PolySample():

    def __init__(self, vertices = [(0,0),(1,0),(1,1),(0,1)]):
        polygon = Polygon(vertices)

        # setup polygon and triangulate it
        self.vertices = vertices
        self.areas = []
        self.transforms = []
        self.triangles = []
        for t in triangulate(polygon):
            self.areas.append(t.area)
            (x0, y0), (x1, y1), (x2, y2), _ = t.exterior.coords
            self.triangles.append([[x0, x1, x2, x0], [y0, y1, y2, y0]])
            #print((x0, y0), (x1, y1), (x2, y2))
            self.transforms.append([x2 - x0, x1 - x0, y2 - y0, y1 - y0, x0, y0])

        #print("Triangulations:", transforms)
        #print("Areas of triangulation:", areas)

    def random_points_in_polygon(self, k):
        """
        Return list of k points chosen uniformly at random inside the polygon.
        https://codereview.stackexchange.com/questions/69833/generate-sample-coordinates-inside-a-polygon
        """
        points = []
        for transform in random.choices(self.transforms, weights=self.areas, k=k):
            x, y = [random.random() for _ in range(2)]
            if x + y > 1:
                p = Point(1 - x, 1 - y)
            else:
                p = Point(x, y)

            transformed = list(affine_transform(p, transform).coords)
            #points.append(affine_transform(p, transform))
            points.append(transformed[0])

        return points


    def custom_plasma_sampling(self, k, tau_lo = 2.44, tau_hi = 25, Dyss_lo = -1.5, Dyss_hi = 1.5):
        """
        Turns out for linear dynamics model sampling for the PlasmaRL application,
        uniform sampling within the polygon NOT the best method

        For example:
            * d(dy/dt|t=0)/da is small for low values of a, but becomes quite large
              as a -> 1
            * Therefore, if we sample uniformly in a space, we are automatically
              biased to sample the fast dynamics more often

        Therefore, instead of sampling in model parameter space, we should
        sample in terms of the quantities we actually care about such 
        as the time constant and gain
        """

        points = []

        for _ in range(k):

            tau = np.random.uniform(tau_lo,tau_hi)
            a = 1 + np.log(0.3679)/tau

            delX = np.random.uniform(Dyss_lo/0.673,Dyss_hi/0.673)
            b = -1*(7.4315+delX)*(a-1)/1.14

            points.append((a,b))

        return points


    def plot_sampling(self, fname, k):
        points = self.random_points_in_polygon(k)

        x = [point[0] for point in points]
        y = [point[1] for point in points]

        fig = plt.figure()
        plt.scatter(x, y, alpha=0.3, color='black')

        #for triangle in self.triangles:
        #    print(triangle[0], triangle[1])
        #    plt.plot(triangle[0], triangle[1])

        plt.savefig(fname)

    def plot_plasma_sampling(self, fname, k):
        points = self.custom_plasma_sampling(k)

        x = [point[0] for point in points]
        y = [point[1] for point in points]

        fig = plt.figure()
        plt.scatter(x, y, alpha=0.3, color='black')

        #for triangle in self.triangles:
        #    print(triangle[0], triangle[1])
        #    plt.plot(triangle[0], triangle[1])

        plt.savefig(fname)

    def plot_dynamics(self, fname, k):
        
        points = self.random_points_in_polygon(k)
        a = [point[0] for point in points]
        b = [point[1] for point in points]

        for i in range(len(a)):
            traj_t = []
            traj_y = []
            x=0
            for t in range(100):
                x = a[i]*x+b[i]*1.14
                traj_t.append(t)
                traj_y.append(x*0.673)
            plt.plot(traj_t, traj_y, color='blue')

        plt.savefig(fname)
#polygon = Polygon([(0.8, 1.8),(0.96, 0.35),(0.96, 0.15),(0.8, 0.78)])
#vertices = [(0.8, 1.8),(0.96, 0.35),(0.96, 0.15),(0.8, 0.78)]
#polygon = PolySample(vertices)
#polygon.plot_sampling(5000)
#points, triangles =random_points_in_polygon(polygon, 10000)




