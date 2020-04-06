import numpy as np


def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data-  numpy array of points
    :param generator: random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being index to a sample
             which is chosen as centroid.
    '''
    # TODO:
    # implement the Kmeans++ algorithm of how to choose the centers according to the lecture and notebook
    # Choose 1st center randomly and use Euclidean distance to calculate other centers.
    #raise Exception(
    #         'Implement get_k_means_plus_plus_center_indices function in Kmeans.py')


    centers = []
    centers.append(generator.randint(n))
    for i in range(1, n_cluster):
        eu_distance = []
        for j in x:
            nearest_center , nearest_distance = get_nearest_center(j, centers)
            eu_distance.append(nearest_distance)
        max_prob = 0
        index = 0
        for k, distance in enumerate(eu_distance):
            prob = distance / sum(eu_distance)
            if prob > max_prob:
                max_prob = prob
                index = k
        centers.append(index)


    # DO NOT CHANGE CODE BELOW THIS LINE

    print("[+] returning center for [{}, {}] points: {}".format(n, len(x), centers))
    return centers
def get_nearest_center(point, centers):
    index = 0
    min_distance = float("inf")
    for i, center in enumerate(centers):
        eu_distance = sum((point - center)**2)
        if eu_distance < min_distance:
            min_distance = eu_distance
            index = i
    return centers[index], min_distance


def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)




class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a length (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates a Int)
            Note: Number of iterations is the number of time you update the assignment
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        #raise Exception(
        #     'Implement fit function in KMeans class')
        index = self.generator.choice(N, self.n_cluster, replace = True)
        centroids = np.array([x[i] for i in index])
        #y = np.zeros(N, dtype=int)
        J = 10**10

        iter = 0
        while iter < self.max_iter:
            # Compute membership
            distance = np.sum(((x - np.expand_dims(centroids, axis=1)) ** 2), axis=2)
            y = np.argmin(distance, axis=0)
            # Compute distortion
            J_new = np.sum([np.sum((x[y == k] - centroids[k]) ** 2) for k in range(self.n_cluster)])
            if np.absolute(J - J_new) <= self.e:
                self.max_iter += 1
                break
            J = J_new
            # Compute means
            centroids = np.array([np.mean(x[y == k], axis=0) for k in range(self.n_cluster)])
            iter += 1

        # DO NOT CHANGE CODE BELOW THIS LINE
        return centroids, y, self.max_iter

        


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)

            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (N,) numpy array)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        #raise Exception(
        #     'Implement fit function in KMeansClassifier class')
        k_means = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e, generator=self.generator)
        centroids, membership, max_iter = k_means.fit(x, centroid_func)
        votes = [{} for k in range(self.n_cluster)]
        for Y , cluster in zip(y, membership):
            if Y in votes[cluster].keys():
                votes[cluster][Y] += 1
            else:
                votes[cluster][Y] = 1
        labels = []
        for i in votes:
            if not i:
                max_label = 0
            else:
                max_label = max(i, key = i.get)
            labels.append(max_label)
        centroid_labels = np.array(labels)

        
        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        # raise Exception(
        #     'Implement predict function in KMeansClassifier class')
        labels = []
        for i in range(N):
            distance = [(x[i] - self.centroids[n]).dot((x[i] - self.centroids[n]).T) for n in range(self.n_cluster)]
            index = np.argmin(distance)
            temp = self.centroid_labels[index]
            labels.append(temp)

        # DO NOT CHANGE CODE BELOW THIS LINE
        return np.array(labels)
        

def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors

        Return new image from the image by replacing each RGB value in image with nearest code vectors (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    # TODO
    # - comment/remove the exception
    # - implement the function

    # DONOT CHANGE CODE ABOVE THIS LINE
    # raise Exception(
    #         'Implement transform_image function')
    pixel = image.shape[0]
    RGB   = image.shape[1]
    dimension = (pixel, RGB, 3)
    new_im = np.zeros(dimension)
    for i in range(pixel):
        for j in range(RGB):
            distance = [(image[i,j] - code_vectors[k]).dot((image[i,j] - code_vectors[k]).T) for k in range(code_vectors.shape[0])]
            index = np.argmin(distance)
            new_im[i, j ] = code_vectors[index]

    # DONOT CHANGE CODE BELOW THIS LINE
    return new_im

